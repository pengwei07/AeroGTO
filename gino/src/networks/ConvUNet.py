from typing import List, Tuple, Union
from torchtyping import TensorType
import matplotlib.pyplot as plt
from matplotlib import cm

import torch
import torch.nn as nn

from .base_model import BaseModel
from src.utils.visualization import fig_to_numpy


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet3D(BaseModel):
    def __init__(self, in_channels, out_channels, depth=2, base_filters=64):
        super(UNet3D, self).__init__()
        self.depth = depth
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Encoder layers
        for i in range(depth):
            in_ch = in_channels if i == 0 else base_filters * 2 ** (i - 1)
            out_ch = base_filters * 2**i
            self.encoder.append(DoubleConv(in_ch, out_ch))
            self.encoder.append(nn.MaxPool3d(kernel_size=2, stride=2))

        # Middle layers
        self.middle = DoubleConv(
            base_filters * 2 ** (depth - 1), base_filters * 2**depth
        )

        # Decoder layers
        for i in range(depth, 0, -1):
            in_ch = base_filters * 2**i
            out_ch = base_filters * 2 ** (i - 1)
            self.decoder.append(
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(in_ch, out_ch))

        # Output layer
        self.output = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def forward(self, x: TensorType["batch", "in_channels", "x", "y", "z"]):
        enc_outputs = []

        # Encoder
        for i in range(0, self.depth * 2, 2):
            x = self.encoder[i](x)
            enc_outputs.append(x)
            x = self.encoder[i + 1](x)

        # Middle
        x = self.middle(x)

        # Decoder
        for i in range(0, self.depth * 2, 2):
            x = self.decoder[i](x)
            x = torch.cat((x, enc_outputs[-(i // 2) - 1]), dim=1)
            x = self.decoder[i + 1](x)

        # Output
        x = self.output(x)
        return x


# Extend the UNet3D class to support extracting features from a specified point coordinates.
class UNet3DWithSamplePoints(UNet3D):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        unet_out_channels: int = 32,
        depth: int = 2,
        base_filters: int = 64,
        use_position_input: bool = False,
    ):
        super(UNet3DWithSamplePoints, self).__init__(
            in_channels, unet_out_channels, depth, base_filters
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(unet_out_channels, unet_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(unet_out_channels, out_channels),
        )
        self.use_position_input = use_position_input

    def forward(
        self,
        x: TensorType["batch", "in_channels", "x", "y", "z"],
        output_points: TensorType["batch", "num_points", 3],
    ):
        # Use super class to get the output of the UNet3D
        x = super(UNet3DWithSamplePoints, self).forward(x)
        # sample_points to 5D by adding two dimensions
        output_points = output_points.unsqueeze(2).unsqueeze(2)
        x = torch.nn.functional.grid_sample(x, output_points, align_corners=False)
        # Remove the two dimensions
        x = x.squeeze(3).squeeze(3)
        # (BxCxN) -> (BxNxC)
        x = x.permute(0, 2, 1)
        x = self.final_mlp(x)
        return x

    def data_dict_to_input(self, data_dict):
        input_grid_features = data_dict["sdf"].unsqueeze(1).to(self.device)
        if self.use_position_input:
            grid_points = data_dict["sdf_query_points"].to(self.device)
            input_grid_features = torch.cat((input_grid_features, grid_points), dim=1)
        output_points = data_dict["vertices"].to(self.device)
        return input_grid_features, output_points

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

    def loss_dict(self, data_dict, loss_fn=None, **kwargs):
        input_grid_features, output_points = self.data_dict_to_input(data_dict)
        pred_pressure = self(input_grid_features, output_points)
        gt_pressure = data_dict["pressure"].to(self.device).reshape(-1, 1)
        if loss_fn is None:
            loss_fn = self.loss
        return {"loss": loss_fn(pred_pressure, gt_pressure)}

    @torch.no_grad()
    def image_dict(self, data_dict, loss_fn=None, decode_fn=None, **kwargs):
        input_grid_features, output_points = self.data_dict_to_input(data_dict)
        pred = self(input_grid_features, output_points)
        truth = data_dict["pressure"].to(self.device)
        if decode_fn is not None:
            pred = decode_fn(pred)
            truth = decode_fn(truth)

        ### plot
        out_dict = {}
        vertices = data_dict["vertices"]
        for b in range(len(data_dict["vertices"])):
            x, y, z = vertices[b].T.cpu().numpy()
            pred_b = pred[b].squeeze().cpu().numpy()
            truth_b = truth[b].cpu().numpy()
            error_b = truth_b - pred_b

            fig = plt.figure(figsize=(24, 8))
            ax1 = fig.add_subplot(131, projection="3d")
            sc1 = ax1.scatter(x, y, z, c=truth_b, cmap=cm.viridis)
            plt.colorbar(sc1, ax=ax1, shrink=0.5).set_label("Truth")

            ax2 = fig.add_subplot(132, projection="3d")
            sc2 = ax2.scatter(x, y, z, c=pred_b, cmap=cm.viridis)
            plt.colorbar(sc2, ax=ax2, shrink=0.5).set_label("Prediction")

            ax3 = fig.add_subplot(133, projection="3d")
            sc3 = ax3.scatter(x, y, z, c=error_b, cmap=cm.viridis)
            plt.colorbar(sc3, ax=ax3, shrink=0.5).set_label("Error")

            img = fig_to_numpy(fig)

            out_dict[f"image_{b}"] = img

        plt.close(fig)
        return out_dict
