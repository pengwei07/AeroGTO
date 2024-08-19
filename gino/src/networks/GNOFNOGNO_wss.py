import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from .neighbor_ops import NeighborSearchLayer, NeighborMLPConvLayer, NeighborMLPConvLayerLinear, NeighborMLPConvLayerWeighted

from neuralop.models import FNO

from .net_utils import PositionalEmbedding, AdaIN, MLP

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

class GNOFNOGNO(BaseModel):
    def __init__(
            self,
            radius_in=0.05,
            radius_out=0.05,
            embed_dim=64,
            # hidden_channels=(86, 86),
            hidden_channels=(32, 32),
            in_channels=1,
            out_channels=1,
            # fno_modes=(32, 32, 32),
            fno_modes=(16, 16, 16),
            #fno_hidden_channels=86,
            #fno_out_channels=86,
            fno_hidden_channels=32,
            fno_out_channels=32,
            fno_domain_padding=0.125,
            fno_norm="group_norm",
            fno_factorization="tucker",
            fno_rank=0.4,
            linear_kernel=True,
            weighted_kernel=True,
    ):
        super().__init__()
        self.weighted_kernel = weighted_kernel
        self.nb_search_in = NeighborSearchLayer(radius_in)
        self.nb_search_out = NeighborSearchLayer(radius_out)
        self.pos_embed = PositionalEmbedding(embed_dim)
        self.df_embed = MLP(
            [in_channels, embed_dim, 3 * embed_dim], torch.nn.GELU
        )
        self.linear_kernel = linear_kernel

        kernel1 = MLP(
            [10 * embed_dim, 512, 256, hidden_channels[0]], torch.nn.GELU
        )
        self.gno1 = NeighborMLPConvLayerWeighted(mlp=kernel1)

        if linear_kernel == False:
            kernel2 = MLP(
                [fno_out_channels + 4 * embed_dim, 512, 256, hidden_channels[1]], torch.nn.GELU
            )
            self.gno2 = NeighborMLPConvLayer(mlp=kernel2)
        else:
            kernel2 = MLP(
                [7 * embed_dim, 512, 256, hidden_channels[1]], torch.nn.GELU
            )
            self.gno2 = NeighborMLPConvLayerLinear(mlp=kernel2)

        self.fno = FNO(
            fno_modes,
            hidden_channels=fno_hidden_channels,
            in_channels=hidden_channels[0] + 3 + in_channels,
            out_channels=fno_out_channels,
            use_mlp=True,
            mlp={"expansion": 1.0, "dropout": 0},
            domain_padding=fno_domain_padding,
            factorization=fno_factorization,
            norm=fno_norm,
            rank=fno_rank,
        )

        self.projection = Projection(
            in_channels=hidden_channels[1],
            out_channels=out_channels,
            hidden_channels=256,
            non_linearity=nn.functional.gelu,
            n_dim=1,
        )

    # x_in : (n_in, 3)
    # x_out : (n_x, n_y, n_z, 3)
    # df : (in_channels, n_x, n_y, n_z)
    # ara : (n_in)

    # u : (n_in, out_channels)
    def forward(self, x_in, x_out, df, x_eval=None, area_in=None, area_eval=None):
        
        # setting the device
        new_new_device = x_in.device

        # manifold to latent neighborhood
        in_to_out_nb = self.nb_search_in(x_in, x_out.view(-1, 3))

        # latent to manifold neighborhood
        if x_eval is not None:
            out_to_in_nb = self.nb_search_out(x_out.view(-1, 3), x_eval)
        else:
            out_to_in_nb = self.nb_search_out(x_out.view(-1, 3), x_in)

        # Embed manifold coordinates
        resolution = df.shape[-1]  # 16,32,64
        n_in = x_in.shape[0]
        if area_in is None or self.weighted_kernel is False:
            area_in = torch.ones((n_in,), device = new_new_device)
        x_in = torch.cat([x_in, area_in.unsqueeze(-1)], dim=-1)
        x_in_embed = self.pos_embed(
            x_in.reshape(-1, )
        ).reshape(
            (n_in, -1)
        )  # (n_in, 4*embed_dim)

        if x_eval is not None:
            n_eval = x_eval.shape[0]
            if area_eval is None or self.weighted_kernel is False:
                area_eval = torch.ones((n_eval,), device = new_new_device)
            x_eval = torch.cat([x_eval, area_eval.unsqueeze(-1)], dim=-1)
            x_eval_embed = self.pos_embed(
                x_eval.reshape(-1, )
            ).reshape(
                (n_eval, -1)
            )  # (n_eval, 4*embed_dim)

        # Embed latent space coordinates
        x_out_embed = self.pos_embed(
            x_out.reshape(-1, )
        ).reshape(
            (resolution ** 3, -1)
        )  # (n_x*n_y*n_z, 3*embed_dim)

        # Embed latent space features
        df_embed = self.df_embed(df.permute(1, 2, 3, 0)).reshape(
            (resolution ** 3, -1)
        )  # (n_x*n_y*n_z, 3*embed_dim)
        grid_embed = torch.cat([x_out_embed, df_embed], dim=-1)  # (n_x*n_y*n_z, 6*embed_dim)

        # GNO : project to latent space
        u = self.gno1(
            x_in_embed, in_to_out_nb, grid_embed, area_in
        )  # (n_x*n_y*n_z, hidden_channels[0])
        u = (
            u.reshape(resolution, resolution, resolution, -1).permute(3, 0, 1, 2).unsqueeze(0)
        )  # (1, hidden_channels[0], n_x, n_y, n_z)

        # Add positional embedding and distance information
        u = torch.cat(
            (x_out.permute(3, 0, 1, 2).unsqueeze(0), df.unsqueeze(0), u), dim=1
        )  # (1, 3+in_channels+hidden_channels[0], n_x, n_y, n_z)

        # FNO on latent space
        u = self.fno(u)  # (1, fno_out_channels, n_x, n_y, n_z)
        u = (
            u.squeeze().permute(1, 2, 3, 0).reshape(resolution ** 3, -1)
        )  # (n_x*n_y*n_z, fno_out_channels)

        # GNO : project to manifold
        if self.linear_kernel == False:
            if x_eval is not None:
                u = self.gno2(u, out_to_in_nb, x_eval_embed)  # (n_eval, hidden_channels[1])
            else:
                u = self.gno2(u, out_to_in_nb, x_in_embed)  # (n_in, hidden_channels[1])
        else:
            if x_eval is not None:
                u = self.gno2(
                    x_in=x_out_embed,
                    neighbors=out_to_in_nb,
                    in_features=u,
                    x_out=x_eval_embed,
                )
            else:
                u = self.gno2(
                    x_in=x_out_embed,
                    neighbors=out_to_in_nb,
                    in_features=u,
                    x_out=x_in_embed,
                )
        u = u.unsqueeze(0).permute(0, 2, 1)  # (1, hidden_channels[1], n_in/n_eval)

        # Pointwise projection to out channels
        u = self.projection(u).squeeze(0).permute(1, 0)  # (n_in/n_eval, out_channels)

        return u

class GNOFNOGNOAhmed_wss(GNOFNOGNO):
    def __init__(
            self,
            radius_in=0.05,
            radius_out=0.05,
            embed_dim=16,
            # hidden_channels=(86, 86),
            hidden_channels=(16, 16),
            in_channels=2,
            out_channels=3,
            # fno_modes=(32, 32, 32),
            fno_modes=(16, 16, 16),
            #fno_hidden_channels=86,
            #fno_out_channels=86,
            fno_hidden_channels=16,
            fno_out_channels=16,
            fno_domain_padding=0.125,
            fno_norm="ada_in",
            adain_embed_dim=64,
            fno_factorization="tucker",
            fno_rank=0.4,
            linear_kernel=True,
            weighted_kernel=True,
            max_in_points=5000,
            subsample_train=1,
            subsample_eval=1,
    ):
        if fno_norm == "ada_in":
            init_norm = 'group_norm'
        else:
            init_norm = fno_norm

        self.max_in_points = max_in_points
        self.subsample_train = subsample_train
        self.subsample_eval = subsample_eval

        super().__init__(
            radius_in=radius_in,
            radius_out=radius_out,
            embed_dim=embed_dim,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            fno_modes=fno_modes,
            fno_hidden_channels=fno_hidden_channels,
            fno_out_channels=fno_out_channels,
            fno_domain_padding=fno_domain_padding,
            fno_norm=init_norm,
            fno_factorization=fno_factorization,
            fno_rank=fno_rank,
            linear_kernel=linear_kernel,
            weighted_kernel=weighted_kernel,
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
 
    def data_dict_to_input(self, data_dict):
        # setting
        # new_device = data_dict.device

        x_in = data_dict["centroids"][0]  # (n_in, 3)
        new_device = x_in.device

        x_out = (
            data_dict["df_query_points"].squeeze(0).permute(1, 2, 3, 0)
        )  # (n_x, n_y, n_z, 3)
        df = data_dict["df"]  # (1, n_x, n_y, n_z)
        area = data_dict["areas"][0] # (n_in, 3)

        # info_fields = torch.cat([
        #    v*torch.ones_like(df) for _, v in data_dict['info'][0].items()
        # ], dim=0) # (8, n_x, n_y, n_z)

        info_fields = data_dict['info'][0]['velocity'] * torch.ones_like(df)

        df = torch.cat((
            df, info_fields
        ), dim=0)  # (2, n_x, n_y, n_z)

        if self.use_adain:
            vel = torch.tensor([data_dict['info'][0]['velocity']]).view(-1, ).to(new_device)
            vel_embed = self.adain_pos_embed(vel)
            for norm in self.fno.fno_blocks.norm:
                norm.update_embeddding(vel_embed)

        x_in, x_out, df, area = (
            x_in.to(new_device),
            x_out.to(new_device),
            df.to(new_device),
            area.to(new_device),
        )
        return x_in, x_out, df, area
    
    def data_dict_to_input_test(self, data_dict, device_test):
        # setting
        # new_device = data_dict.device

        x_in = data_dict["centroids"][0]  # (n_in, 3)
        new_device = device_test

        x_out = (
            data_dict["df_query_points"].squeeze(0).permute(1, 2, 3, 0)
        )  # (n_x, n_y, n_z, 3)
        df = data_dict["df"]  # (1, n_x, n_y, n_z)
        area = data_dict["areas"][0] # (n_in, 3)

        # info_fields = torch.cat([
        #    v*torch.ones_like(df) for _, v in data_dict['info'][0].items()
        # ], dim=0) # (8, n_x, n_y, n_z)

        info_fields = data_dict['info'][0]['velocity'] * torch.ones_like(df)

        df = torch.cat((
            df, info_fields
        ), dim=0)  # (2, n_x, n_y, n_z)

        if self.use_adain:
            vel = torch.tensor([data_dict['info'][0]['velocity']]).view(-1, ).to(new_device)
            vel_embed = self.adain_pos_embed(vel)
            for norm in self.fno.fno_blocks.norm:
                norm.update_embeddding(vel_embed)

        x_in, x_out, df, area = (
            x_in.to(new_device),
            x_out.to(new_device),
            df.to(new_device),
            area.to(new_device),
        )
        return x_in, x_out, df, area
    
    # now exists errors
    @torch.no_grad()
    def eval_dict(self, device, data_dict, loss_fn=None, decode_fn=None, **kwargs):
        
        x_in, x_out, df, area = self.data_dict_to_input_test(data_dict, device)

        x_in = x_in[::self.subsample_eval, ...]
        area = area[::self.subsample_eval]
        
        if self.max_in_points is not None:
            r = min(self.max_in_points, x_in.shape[0])
            pred_chunks = []
            x_in_chunks = torch.split(x_in, r, dim=0)
            area_chunks = torch.split(area, r, dim=0)
            for j in range(len(x_in_chunks)):
                pred_index_j = super().forward(x_in, x_out, df, x_in_chunks[j], area_in=area, area_eval=area_chunks[j])
                pred_chunks.append(pred_index_j)
            pred = torch.cat(tuple(pred_chunks), dim=0)
        else:
            pred = self(x_in, x_out, df, area=area)

        # pred = pred.reshape(1, -1)
        if loss_fn is None:
            loss_fn = self.loss
        # check data loader
        truth = data_dict["wss"][0].to(device)[::self.subsample_eval, ...]

        pred, truth = pred.permute(1, 0), truth.permute(1, 0)
        
        out_dict = {"l2": loss_fn(pred, truth)}
        if decode_fn is not None:
            pred_decode = decode_fn(pred)
            truth_decode = decode_fn(truth)
            out_dict["l2_decoded"] = loss_fn(pred_decode, truth_decode)
            '''
            # print("data_dict['dragWeight'][0].shape, pred.shape, truth.shape", data_dict['dragWeight'][0].shape, pred.shape, truth.shape)
            # data_dict['dragWeight'][0].shape, pred.shape, truth.shape torch.Size([127889]) torch.Size([127889, 3]) torch.Size([127889, 3])

            #drag_pred = torch.sum(data_dict['dragWeight'][0].cuda()*pred)
            #drag_truth = torch.sum(data_dict['dragWeight'][0].cuda()*truth)

            #out_dict['drag_pred']  = drag_pred
            #out_dict['drag_truth']  =  drag_truth
            '''
        return out_dict, truth_decode, pred_decode
    
    def forward(self, data_dict):
        # new_device = data_dict.device
        x_in, x_out, df, area = self.data_dict_to_input(data_dict)
        x_in = x_in[::self.subsample_train, ...]
        area = area[::self.subsample_train]
        r = min(self.max_in_points, x_in.shape[0])
        indices = torch.randperm(x_in.shape[0])[:r]
        
        # use GNOFNOGNO's forward()
        pred = super().forward(x_in, x_out, df, x_in[indices, ...], area, area[indices])

        truth = data_dict["wss"][0][::self.subsample_train]
        truth = truth[indices].to(x_in.device)
        
        # (5000,3)
        # permute
        # print(pred.shape, truth.shape)  torch.Size([5000, 3]) torch.Size([5000, 3])

        return pred.permute(1, 0), truth.permute(1, 0)
    
