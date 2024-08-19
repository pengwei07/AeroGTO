import torch
import torch.nn as nn
from .base_model import BaseModel

from neuralop.models import FNO

from .neighbor_ops import NeighborSearchLayer, NeighborMLPConvLayer, NeighborMLPConvLayerLinear
from .net_utils import PositionalEmbedding, AdaIN, MLP

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
    
class FNOGNO(BaseModel):
    def __init__(
            self,
            in_channels=4,
            out_channels=1,
            fno_modes=(32, 32, 32),
            fno_hidden_channels=86,
            fno_domain_padding=0.125,
            fno_norm="group_norm",
            fno_factorization="tucker",
            fno_rank=0.4,
            adain_embed_dim=64,
            coord_embed_dim=16,
            radius=0.055,
            linear_kernel=True,
            max_in_points=None,
    ):
        super().__init__()

        if fno_norm == "ada_in":
            init_norm = 'group_norm'
        else:
            init_norm = fno_norm

        self.linear_kernel = linear_kernel
        self.max_in_points = max_in_points

        self.fno = FNO(
            fno_modes,
            hidden_channels=fno_hidden_channels,
            in_channels=in_channels,
            out_channels=1,
            use_mlp=True,
            mlp={"expansion": 1.0, "dropout": 0},
            domain_padding=fno_domain_padding,
            factorization=fno_factorization,
            norm=init_norm,
            rank=fno_rank,
        )

        if fno_norm == "ada_in":
            self.adain_pos_embed = PositionalEmbedding(adain_embed_dim)
            self.fno.fno_blocks.norm = nn.ModuleList(
                AdaIN(adain_embed_dim, fno_hidden_channels)
                for _ in range(self.fno.fno_blocks.n_norms * self.fno.fno_blocks.convs.n_layers)
            )
            self.use_adain = True
        else:
            self.use_adain = False

        self.nb_search_out = NeighborSearchLayer(radius)
        self.pos_embed = PositionalEmbedding(coord_embed_dim)

        kernel_in_dim = 6 * coord_embed_dim
        kernel_in_dim += 0 if self.linear_kernel else fno_hidden_channels

        kernel = MLP([kernel_in_dim, 512, 256, fno_hidden_channels],
                     nn.GELU)

        if self.linear_kernel:
            self.gno = NeighborMLPConvLayerLinear(mlp=kernel)
        else:
            self.gno = NeighborMLPConvLayer(mlp=kernel)

        self.projection = Projection(
            in_channels=fno_hidden_channels,
            out_channels=out_channels,
            hidden_channels=256,
            non_linearity=nn.functional.gelu,
            n_dim=1,
        )

    # x_in : (n_in, 3)
    # x_out : (n_x, n_y, n_z, 3)
    # df : (1, n_x, n_y, n_z)

    # u : (n_in, out_channels)
    def forward(self, x_in, x_out, df):
        out_to_in_nb = self.nb_search_out(x_out.view(-1, 3), x_in)

        n_out = x_out.view(-1, 3).shape[0]
        x_out_embed = self.pos_embed(
            x_out.reshape(-1, )
        ).reshape(
            (n_out, -1)
        )

        # Latent space and distance
        x_out = torch.cat(
            (df, x_out.permute(3, 0, 1, 2)), dim=0
        ).unsqueeze(
            0
        )  # (1, 12, n_x, n_y, n_z)

        # Apply FNO blocks
        x_out = self.fno.lifting(x_out)
        if self.fno.domain_padding is not None:
            x_out = self.fno.domain_padding.pad(x_out)

        for layer_idx in range(self.fno.n_layers):
            x_out = self.fno.fno_blocks(x_out, layer_idx)

        if self.fno.domain_padding is not None:
            x_out = self.fno.domain_padding.unpad(x_out)
        # x_out: (1, fno_hidden_channels, n_x, n_y, n_z)

        x_out = x_out.squeeze(0).permute(1, 2, 3, 0).reshape(-1, self.fno.hidden_channels)
        # x_out: (n_x*n_y*n_z, fno_hidden_channels)

        n_in = x_in.shape[0]
        x_in_embed = self.pos_embed(
            x_in.reshape(-1, )
        ).reshape(
            (n_in, -1)
        )

        if self.linear_kernel:
            x_out = self.gno(x_out_embed, out_to_in_nb, x_out, x_in_embed)
        else:
            x_out = torch.cat((x_out_embed, x_out), dim=1)
            # x_out: (n_x*n_y*n_z, fno_hidden_channels + 3*coord_embed_dim)
            x_out = self.gno(x_out, out_to_in_nb, x_in_embed)

            # x_out: (n_in, fno_hidden_channels)

        x_out = x_out.unsqueeze(0).permute(0, 2, 1)  # (1, fno_hidden_channels, n_in)

        # Project pointwise to out channels
        x_out = self.projection(x_out).squeeze(0).permute(1, 0)  # (n_in, out_channels)

        return x_out

    # Batch size 1 is assumed

    def data_dict_to_input(self, data_dict):
        x_in = data_dict["vertices"].squeeze(0)  # (n_in, 3)
        x_out = (
            data_dict["sdf_query_points"].squeeze(0).permute(1, 2, 3, 0)
        )  # (n_x, n_y, n_z, 3)
        df = data_dict["sdf"]  # (1, n_x, n_y, n_z)

        if self.use_adain:
            vel = torch.tensor([data_dict['info'][0]['velocity']]).view(-1, ).to(self.device)
            vel_embed = self.adain_pos_embed(vel)
            for norm in self.fno.fno_blocks.norm:
                norm.update_embeddding(vel_embed)

        x_in, x_out, df = (
            x_in.to(self.device),
            x_out.to(self.device),
            df.to(self.device),
        )

        return x_in, x_out, df

    @torch.no_grad()
    def eval_dict(self, data_dict, loss_fn=None, decode_fn=None, **kwargs):
        x_in, x_out, df = self.data_dict_to_input(data_dict)

        if self.max_in_points is not None:
            r = min(self.max_in_points, x_in.shape[0])
            pred_chunks = []
            x_in_chunks = torch.split(x_in, r, dim=0)
            for j in range(len(x_in_chunks)):
                pred_chunks.append(self(x_in_chunks[j], x_out, df))
            pred = torch.cat(tuple(pred_chunks), dim=0)
        else:
            pred = self(x_in, x_out, df)

        pred = pred.reshape(1, -1)

        if loss_fn is None:
            loss_fn = self.loss
        truth = data_dict["pressure"].to(self.device).reshape(1, -1)
        out_dict = {"l2": loss_fn(pred, truth)}

        if decode_fn is not None:
            pred = decode_fn(pred)
            truth = decode_fn(truth)
            out_dict["l2_decoded"] = loss_fn(pred, truth)

            torch.save(pred.view(-1, ).cpu().detach(), 'pred_car_' + str(kwargs['ind']).zfill(3) + '.pt')

        return out_dict

    def loss_dict(self, data_dict, loss_fn=None):
        x_in, x_out, df = self.data_dict_to_input(data_dict)

        if self.max_in_points is not None:
            r = min(self.max_in_points, x_in.shape[0])
            indices = torch.randperm(x_in.shape[0])[:r]
            x_in = x_in[indices, ...]

        pred = self(x_in, x_out, df)

        if loss_fn is None:
            loss_fn = self.loss

        if self.max_in_points is not None:
            truth = data_dict["pressure"][indices].view(1, -1).to(self.device)
        else:
            truth = data_dict["pressure"].view(1, -1).to(self.device)

        return {
            "loss": loss_fn(
                pred.view(1, -1), truth
            )
        }


class FNOGNOAhmed(BaseModel):
    def __init__(
            self,
            in_channels=5,
            out_channels=1,
            fno_modes=(32, 32, 32),
            fno_hidden_channels=86,
            fno_domain_padding=0.125,
            fno_norm="ada_in",
            fno_factorization="tucker",
            fno_rank=0.4,
            adain_embed_dim=64,
            coord_embed_dim=16,
            radius=0.033,
            linear_kernel=True,
            max_in_points=5000,
    ):
        super().__init__()

        if fno_norm == "ada_in":
            init_norm = 'group_norm'
        else:
            init_norm = fno_norm

        self.linear_kernel = linear_kernel
        self.max_in_points = max_in_points

        self.fno = FNO(
            fno_modes,
            hidden_channels=fno_hidden_channels,
            in_channels=in_channels,
            out_channels=1,
            use_mlp=True,
            mlp={"expansion": 1.0, "dropout": 0},
            domain_padding=fno_domain_padding,
            factorization=fno_factorization,
            norm=init_norm,
            rank=fno_rank,
        )

        if fno_norm == "ada_in":
            self.adain_pos_embed = PositionalEmbedding(adain_embed_dim)
            self.fno.fno_blocks.norm = nn.ModuleList(
                AdaIN(adain_embed_dim, fno_hidden_channels)
                for _ in range(self.fno.fno_blocks.n_norms * self.fno.fno_blocks.convs.n_layers)
            )
            self.use_adain = True
        else:
            self.use_adain = False

        self.nb_search_out = NeighborSearchLayer(radius)
        self.pos_embed = PositionalEmbedding(coord_embed_dim)

        kernel_in_dim = 6 * coord_embed_dim
        kernel_in_dim += 0 if self.linear_kernel else fno_hidden_channels

        kernel = MLP([kernel_in_dim, 512, 256, fno_hidden_channels],
                     nn.GELU)

        if self.linear_kernel:
            self.gno = NeighborMLPConvLayerLinear(mlp=kernel)
        else:
            self.gno = NeighborMLPConvLayer(mlp=kernel)

        self.projection = Projection(
            in_channels=fno_hidden_channels,
            out_channels=out_channels,
            hidden_channels=256,
            non_linearity=nn.functional.gelu,
            n_dim=1,
        )

    # x_in : (n_in, 3)
    # x_out : (n_x, n_y, n_z, 3)
    # df : (1, n_x, n_y, n_z)

    # u : (n_in, out_channels)
    def forward(self, x_in, x_out, df):
        out_to_in_nb = self.nb_search_out(x_out.view(-1, 3), x_in)

        n_out = x_out.view(-1, 3).shape[0]
        x_out_embed = self.pos_embed(
            x_out.reshape(-1, )
        ).reshape(
            (n_out, -1)
        )

        # Latent space and distance
        x_out = torch.cat(
            (df, x_out.permute(3, 0, 1, 2)), dim=0
        ).unsqueeze(
            0
        )  # (1, 12, n_x, n_y, n_z)

        # Apply FNO blocks
        x_out = self.fno.lifting(x_out)
        if self.fno.domain_padding is not None:
            x_out = self.fno.domain_padding.pad(x_out)

        for layer_idx in range(self.fno.n_layers):
            x_out = self.fno.fno_blocks(x_out, layer_idx)

        if self.fno.domain_padding is not None:
            x_out = self.fno.domain_padding.unpad(x_out)
        # x_out: (1, fno_hidden_channels, n_x, n_y, n_z)

        x_out = x_out.squeeze(0).permute(1, 2, 3, 0).reshape(-1, self.fno.hidden_channels)
        # x_out: (n_x*n_y*n_z, fno_hidden_channels)

        n_in = x_in.shape[0]
        x_in_embed = self.pos_embed(
            x_in.reshape(-1, )
        ).reshape(
            (n_in, -1)
        )

        if self.linear_kernel:
            x_out = self.gno(x_out_embed, out_to_in_nb, x_out, x_in_embed)
        else:
            x_out = torch.cat((x_out_embed, x_out), dim=1)
            # x_out: (n_x*n_y*n_z, fno_hidden_channels + 3*coord_embed_dim)
            x_out = self.gno(x_out, out_to_in_nb, x_in_embed)

            # x_out: (n_in, fno_hidden_channels)

        x_out = x_out.unsqueeze(0).permute(0, 2, 1)  # (1, fno_hidden_channels, n_in)

        # Project pointwise to out channels
        x_out = self.projection(x_out).squeeze(0).permute(1, 0)  # (n_in, out_channels)

        return x_out

    # Batch size 1 is assumed

    def data_dict_to_input(self, data_dict):
        x_in = data_dict["centroids"][0]  # (n_in, 3)
        x_out = (
            data_dict["df_query_points"].squeeze(0).permute(1, 2, 3, 0)
        )  # (n_x, n_y, n_z, 3)
        df = data_dict["df"]  # (1, n_x, n_y, n_z)

        # info_fields = torch.cat([
        #    v*torch.ones_like(df) for _, v in data_dict['info'][0].items()
        # ], dim=0)

        info_fields = data_dict['info'][0]['velocity'] * torch.ones_like(df)

        df = torch.cat((
            df, info_fields
        ), dim=0)

        if self.use_adain:
            vel = torch.tensor([data_dict['info'][0]['velocity']]).view(-1, ).to(self.device)
            vel_embed = self.adain_pos_embed(vel)
            for norm in self.fno.fno_blocks.norm:
                norm.update_embeddding(vel_embed)

        x_in, x_out, df = (
            x_in.to(self.device),
            x_out.to(self.device),
            df.to(self.device),
        )

        return x_in, x_out, df

    @torch.no_grad()
    def eval_dict(self, data_dict, loss_fn=None, decode_fn=None, **kwargs):
        x_in, x_out, df = self.data_dict_to_input(data_dict)

        if self.max_in_points is not None:
            r = min(self.max_in_points, x_in.shape[0])
            pred_chunks = []
            x_in_chunks = torch.split(x_in, r, dim=0)
            for j in range(len(x_in_chunks)):
                pred_chunks.append(self(x_in_chunks[j], x_out, df))
            pred = torch.cat(tuple(pred_chunks), dim=0)
        else:
            pred = self(x_in, x_out, df)

        pred = pred.reshape(1, -1)

        if loss_fn is None:
            loss_fn = self.loss
        truth = data_dict["pressure"][0].to(self.device).reshape(1, -1)
        out_dict = {"l2": loss_fn(pred, truth)}

        if decode_fn is not None:
            pred = decode_fn(pred)
            truth = decode_fn(truth)
            out_dict["l2_decoded"] = loss_fn(pred, truth)

            # torch.save(pred.view(-1, ).cpu().detach(), 'pred_ahmed_' + str(kwargs['ind']).zfill(3) + '.pt')
        return out_dict

    def loss_dict(self, data_dict, loss_fn=None):
        x_in, x_out, df = self.data_dict_to_input(data_dict)

        if self.max_in_points is not None:
            r = min(self.max_in_points, x_in.shape[0])
            indices = torch.randperm(x_in.shape[0])[:r]
            x_in = x_in[indices, ...]

        pred = self(x_in, x_out, df)

        if loss_fn is None:
            loss_fn = self.loss

        if self.max_in_points is not None:
            truth = data_dict["pressure"][0][indices].view(1, -1).to(self.device)
        else:
            truth = data_dict["pressure"][0].view(1, -1).to(self.device)

        # truth = data_dict["pressure"][0][indices].view(1, -1).to(self.device)
        return {
            "loss": loss_fn(
                pred.view(1, -1), truth
            )
        }
