import torch
import torch.nn as nn

from neuralop.models import FNO
from neuralop.models.spectral_convolution import FactorizedSpectralConv

from torch_geometric.nn import NNConv
from .net_utils import MLP

from .neighbor_ops import NeighborSearchLayer, NeighborMLPConvLayer

import torch.nn.functional as F

class Projection(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, n_dim=2, non_linearity=F.gelu):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = in_channels if hidden_channels is None else hidden_channels 
        self.non_linearity = non_linearity
        Conv = getattr(nn, f'Conv{n_dim}d')
        self.fc1 = Conv(in_channels, hidden_channels, 1)
        self.fc2 = Conv(hidden_channels, out_channels, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.non_linearity(x)
        x = self.fc2(x)
        return x
    
class SDFFNOGNO(FNO):
    def __init__(
        self,
        n_modes,
        hidden_channels,
        in_channels=4,
        out_channels=1,
        lifting_channels=64,
        projection_channels=64,
        n_layers=4,
        interp_mode="interp_before",
        incremental_n_modes=None,
        use_mlp=False,
        mlp=None,
        non_linearity=torch.nn.functional.gelu,
        norm=None,
        preactivation=False,
        fno_skip="linear",
        mlp_skip="soft-gating",
        separable=False,
        factorization=None,
        rank=1,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="one-sided",
        fft_norm="forward",
        SpectralConv=FactorizedSpectralConv,
        resolution=[64, 64, 64],
        r=0.05,
        gno_implementation="torch_scatter",
        **kwargs
    ):
        FNO.__init__(
            self,
            n_modes,
            hidden_channels,
            in_channels,
            out_channels,
            lifting_channels,
            projection_channels,
            n_layers,
            incremental_n_modes,
            use_mlp,
            mlp,
            non_linearity,
            norm,
            preactivation,
            fno_skip,
            mlp_skip,
            separable,
            factorization,
            rank,
            joint_factorization,
            fixed_rank_modes,
            implementation,
            decomposition_kwargs,
            domain_padding,
            domain_padding_mode,
            fft_norm,
            SpectralConv,
            **kwargs
        )

        self.interp_mode = interp_mode
        self.r = r
        self.resolution = resolution[0]
        self.gno_implementation = gno_implementation

        if self.interp_mode == "interp_after" or self.interp_mode == "interp_before":
            self.interp_f = lambda f, vert: torch.nn.functional.grid_sample(
                f, vert, align_corners=False
            )

        if self.interp_mode == "interp_before":
            self.projection = Projection(
                in_channels=self.hidden_channels,
                out_channels=out_channels,
                hidden_channels=projection_channels,
                non_linearity=non_linearity,
                n_dim=1,
            )
        self.device_indicator_param = nn.Parameter(torch.empty(0))

        self.vert_mlp = MLP([3, self.hidden_channels], torch.nn.GELU)

        #### using PYG
        if self.gno_implementation == "PyG":
            kernel = MLP(
                [(self.resolution + 3) * 2, 128, self.hidden_channels],
                torch.nn.GELU,
                diagonal=True,
            )
            self.graph_conv = NNConv(
                self.hidden_channels, self.hidden_channels, kernel, aggr="mean"
            )
        #### using Chris'
        elif self.gno_implementation == "torch_scatter":
            self.nsearch = NeighborSearchLayer(self.r)
            self.graph_conv = NeighborMLPConvLayer(
                in_channels=self.hidden_channels,
                hidden_dim=self.hidden_channels,
                out_channels=self.hidden_channels,
            )
        else:
            raise ValueError("Network not supported")

    def get_graph(self, x_in, x_out, f_in, f_out):
        x_in = x_in.squeeze()
        x_out = x_out.squeeze()
        N_in = x_in.shape[0]
        pwd = torch.cdist(x_in, x_out).squeeze()
        edge_index = torch.stack(torch.where(pwd <= self.r))
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=x_in.device)
        edge_attr_x = torch.cat([x_in[edge_index[0].T], x_out[edge_index[1].T]], dim=-1)
        edge_attr_f = torch.cat([f_in[edge_index[0].T], f_out[edge_index[1].T]], dim=-1)
        edge_attr = torch.cat([edge_attr_x, edge_attr_f], dim=-1)
        edge_index[1, :] = edge_index[1, :] + N_in
        return edge_index.detach(), edge_attr.detach()

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat(
            [batchsize, 1, size_y, size_z, 1]
        )
        gridy = torch.linspace(0, 1, size_y, dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat(
            [batchsize, size_x, 1, size_z, 1]
        )
        gridz = torch.linspace(0, 1, size_z, dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat(
            [batchsize, size_x, size_y, 1, 1]
        )
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device

    def forward(self, grid_sdf, vert):
        """TFNO's forward pass"""
        vert = vert.unsqueeze(2).unsqueeze(2)

        grid_sdf = self.lifting(grid_sdf)

        if self.domain_padding is not None:
            grid_sdf = self.domain_padding.pad(grid_sdf)

        for layer_idx in range(self.n_layers):
            grid_sdf = self.fno_blocks(grid_sdf, layer_idx)

        if self.domain_padding is not None:
            grid_sdf = self.domain_padding.unpad(grid_sdf)

        ### using linear interpolation
        # vert_sdf = self.interp_f(grid_sdf, vert).squeeze(3).squeeze(3)

        ### using GNO
        vert = vert / 2 + 0.5
        grid = self.get_grid(
            shape=(1, self.resolution, self.resolution, self.resolution),
            device=grid_sdf.device,
        ).flatten(start_dim=0, end_dim=3)
        vert_sdf = self.vert_mlp(vert).squeeze()
        grid_sdf = grid_sdf.permute(0, 2, 3, 4, 1).flatten(start_dim=0, end_dim=3)

        #### using PYG
        if self.gno_implementation == "PyG":
            edge_index, edge_attr = self.get_graph(grid, vert, grid_sdf, vert_sdf)
            sdf = torch.cat([grid_sdf, vert_sdf], dim=0)
            vert_sdf = (
                self.graph_conv(sdf, edge_index, edge_attr).permute(1, 0).unsqueeze(0)
            )
            vert_sdf = vert_sdf[..., self.resolution**3 :]
        #### using Chris'
        elif self.gno_implementation == "torch_scatter":
            neighbors = self.nsearch(grid.squeeze(), vert.squeeze())
            vert_sdf = self.graph_conv(
                inp_features=grid_sdf, outp_features=vert_sdf, neighbors=neighbors
            )
            vert_sdf = vert_sdf.permute(1, 0).unsqueeze(0)
        else:
            raise ValueError("Network not supported")

        vert_sdf = self.projection(vert_sdf).squeeze(2)
        return vert_sdf

    def data_dict_to_input(self, data_dict):
        grid_sdf = torch.cat(
            (data_dict["sdf_query_points"], data_dict["sdf"].unsqueeze(1)), dim=1
        ).to(self.device)
        vert = data_dict["vertices"].to(self.device)
        return grid_sdf, vert

    @torch.no_grad()
    def eval_dict(self, data_dict, loss_fn=None, decode_fn=None, **kwargs):
        input_grid_features, output_points = self.data_dict_to_input(data_dict)
        pred_out = self(input_grid_features, output_points)
        if loss_fn is None:
            loss_fn = self.loss
        gt_out = data_dict["pressure"].to(self.device).reshape(-1, 1)
        out_dict = {"l2": loss_fn(pred_out, gt_out)}
        if decode_fn is not None:
            pred_out = decode_fn(pred_out)
            gt_out = decode_fn(gt_out)
            out_dict["l2_decoded"] = loss_fn(pred_out, gt_out)
        return out_dict

    def loss_dict(self, data_dict, loss_fn=None):
        grid_sdf, vert = self.data_dict_to_input(data_dict)
        vert_sdf = self(grid_sdf, vert)
        if loss_fn is None:
            loss_fn = self.loss
        return {"loss": loss_fn(vert_sdf, data_dict["pressure"].to(self.device))}
