import torch
import torch.nn as nn
from .base_model import BaseModel

from neuralop.models import FNO

import matplotlib.pyplot as plt
from matplotlib import cm

from .net_utils import PositionalEmbedding, AdaIN
from src.utils.visualization import fig_to_numpy

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
    
class FNOInterp(BaseModel):
    def __init__(
        self,
        in_channels=4,
        out_channels=1,
        fno_modes=(32, 32, 32),
        fno_hidden_channels=80,
        fno_domain_padding=0.125,
        fno_norm="group_norm",
        fno_factorization="tucker",
        fno_rank=0.4,
    ):
        super().__init__()

        self.fno = FNO(
            fno_modes,
            hidden_channels=fno_hidden_channels,
            in_channels=in_channels,
            out_channels=1,
            use_mlp=True,
            mlp={"expansion": 1.0, "dropout": 0},
            domain_padding=fno_domain_padding,
            factorization=fno_factorization,
            norm=fno_norm,
            rank=fno_rank,
        )

        self.interp_f = lambda f, coord: torch.nn.functional.grid_sample(
            f, coord, align_corners=False
        )

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
        # Latent space and distance
        x_out = torch.cat((df, x_out.permute(3, 0, 1, 2)), dim=0).unsqueeze(
            0
        )  # (1, 4, n_x, n_y, n_z)

        # Apply FNO blocks
        x_out = self.fno.lifting(x_out)
        if self.fno.domain_padding is not None:
            x_out = self.fno.domain_padding.pad(x_out)

        for layer_idx in range(self.fno.n_layers):
            x_out = self.fno.fno_blocks(x_out, layer_idx)

        if self.fno.domain_padding is not None:
            x_out = self.fno.domain_padding.unpad(x_out)

        # Interpolate to manifold points
        x_in = x_in.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, n_in, 3)
        x_out = self.interp_f(x_out, x_in)  # (1, fno_hidden_channels, 1, 1, n_in)
        x_out = x_out.squeeze(2).squeeze(2)  # (1, fno_hidden_channels, n_in)

        # Project pointwise to out channels
        x_out = self.projection(x_out).squeeze(0).permute(1, 0)  # (n_in, out_channels)

        return x_out

    # Batch size 1 is assumed

    def data_dict_to_input(self, data_dict):
        x_in = data_dict["vertices"].squeeze(0)  # (n_in, 3)
        x_out = (
            data_dict["sdf_query_points"].squeeze(0).permute(1, 2, 3, 0)
        )  # (n_x, n_y, n_z, 3)
        df = data_dict["sdf"]  # (n_x, n_y, n_z)

        x_in, x_out, df = (
            x_in.to(self.device),
            x_out.to(self.device),
            df.to(self.device),
        )

        return x_in, x_out, df

    @torch.no_grad()
    def eval_dict(self, data_dict, loss_fn=None, decode_fn=None, **kwargs):
        x_in, x_out, df = self.data_dict_to_input(data_dict)
        pred = self(x_in, x_out, df).reshape(1, -1)
        if loss_fn is None:
            loss_fn = self.loss
        truth = data_dict["pressure"].to(self.device).reshape(1, -1)
        out_dict = {"l2": loss_fn(pred, truth)}

        if decode_fn is not None:
            pred = decode_fn(pred)
            truth = decode_fn(truth)
            out_dict["l2_decoded"] = loss_fn(pred, truth)
        return out_dict

    def loss_dict(self, data_dict, loss_fn=None):
        x_in, x_out, df = self.data_dict_to_input(data_dict)
        pred = self(x_in, x_out, df)

        if loss_fn is None:
            loss_fn = self.loss
        return {
            "loss": loss_fn(
                pred.view(1, -1), data_dict["pressure"].view(1, -1).to(self.device)
            )
        }


class FNOInterpAhmed(FNOInterp):
    def __init__(
        self,
        in_channels=12,
        out_channels=1,
        fno_modes=(32, 32, 32),
        fno_hidden_channels=80,
        fno_domain_padding=0.125,
        fno_norm="ada_in",
        fno_factorization="tucker",
        fno_rank=0.4,
        embed_dim=256,
        subsample_train=1,
        subsample_eval=1,
    ):
        self.subsample_train = subsample_train
        self.subsample_eval = subsample_eval
        if fno_norm == "ada_in":
            init_norm = "group_norm"
        else:
            init_norm = fno_norm

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            fno_modes=fno_modes,
            fno_hidden_channels=fno_hidden_channels,
            fno_domain_padding=fno_domain_padding,
            fno_norm=init_norm,
            fno_factorization=fno_factorization,
            fno_rank=fno_rank,
        )

        if fno_norm == "ada_in":
            self.pos_embed = PositionalEmbedding(embed_dim)
            self.fno.fno_blocks.norm = nn.ModuleList(
                AdaIN(embed_dim, fno_hidden_channels)
                for _ in range(
                    self.fno.fno_blocks.n_norms * self.fno.fno_blocks.convs.n_layers
                )
            )
            self.use_adain = True
        else:
            self.use_adain = False

    # Batch size 1 is assumed

    def data_dict_to_input(self, data_dict):
        x_in = data_dict["centroids"][0]  # (n_in, 3)
        x_out = (
            data_dict["df_query_points"].squeeze(0).permute(1, 2, 3, 0)
        )  # (n_x, n_y, n_z, 3)
        df = data_dict["df"]  # (1, n_x, n_y, n_z)

        info_fields = torch.cat(
            [v * torch.ones_like(df) for _, v in data_dict["info"][0].items()], dim=0
        )

        df = torch.cat((df, info_fields), dim=0)

        if self.use_adain:
            vel = (
                torch.tensor([data_dict["info"][0]["velocity"]])
                .view(
                    -1,
                )
                .to(self.device)
            )
            vel_embed = self.pos_embed(vel)
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
        x_in = x_in[::self.subsample_eval, ...]
        pred = self(x_in, x_out, df).reshape(1, -1)
        if loss_fn is None:
            loss_fn = self.loss
        truth = data_dict["pressure"][0][::self.subsample_eval, ...].to(self.device).reshape(1, -1)
        out_dict = {"l2": loss_fn(pred, truth)}

        if decode_fn is not None:
            pred = decode_fn(pred)
            truth = decode_fn(truth)
            out_dict["l2_decoded"] = loss_fn(pred, truth)
        return out_dict

    @torch.no_grad()
    def image_dict(self, data_dict, loss_fn=None, decode_fn=None, **kwargs):
        x_in, x_out, df = self.data_dict_to_input(data_dict)
        pred = self(x_in, x_out, df).reshape(1, -1)
        truth = data_dict["pressure"][0].to(self.device).reshape(1, -1)
        if decode_fn is not None:
            pred = decode_fn(pred)
            truth = decode_fn(truth)
        # Visualize truth, pred, and error

        ### plot
        x_in = x_in.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        truth = truth.numpy()
        error = truth - error
        x = x_in[:, 0]
        y = x_in[:, 1]
        z = x_in[:, 2]

        fig = plt.figure(figsize=(24, 8))
        ax1 = fig.add_subplot(131, projection="3d")
        sc1 = ax1.scatter(x, y, z, c=truth, cmap=cm.viridis)
        plt.colorbar(sc1, ax=ax1, shrink=0.5).set_label("Truth")

        ax2 = fig.add_subplot(132, projection="3d")
        sc2 = ax2.scatter(x, y, z, c=pred, cmap=cm.viridis)
        plt.colorbar(sc2, ax=ax2, shrink=0.5).set_label("Prediction")

        ax3 = fig.add_subplot(133, projection="3d")
        sc3 = ax3.scatter(x, y, z, c=error, cmap=cm.viridis)
        plt.colorbar(sc3, ax=ax3, shrink=0.5).set_label("Error")

        img = fig_to_numpy(fig)
        plt.close(fig)

        out_dict = {"img": img}
        return out_dict

    def loss_dict(self, data_dict, loss_fn=None):
        x_in, x_out, df = self.data_dict_to_input(data_dict)
        x_in = x_in[::self.subsample_eval, ...]
        pred = self(x_in, x_out, df)

        if loss_fn is None:
            loss_fn = self.loss
        return {
            "loss": loss_fn(
                pred.view(1, -1), data_dict["pressure"][0][::self.subsample_eval, ...].view(1, -1).to(self.device)
            )
        }
