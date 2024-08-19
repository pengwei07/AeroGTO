import torch
import torch.nn as nn

from .neighbor_ops import NeighborSearchLayer, NeighborMLPConvLayer


# A simple network that only takes input position
class SmallContConvWithMLPKernel(torch.nn.Module):
    def __init__(self, in_channel=3, width=32, out_channel=1, radius=0.1):
        super().__init__()

        self.nsearch = NeighborSearchLayer(radius)
        self.conv1 = NeighborMLPConvLayer(
            in_channels=in_channel, hidden_dim=width, out_channels=width
        )
        self.conv2 = NeighborMLPConvLayer(
            in_channels=width, hidden_dim=width, out_channels=width
        )
        self.to_output = nn.Sequential(nn.GELU(), nn.Linear(width, out_channel))
        self.device_indicator_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device

    def forward(self, x_in):
        """
        Only takes poisitions, but need a separate features in the future
        Assume that the batch size is 1.
        """
        x_in = x_in.squeeze()
        neighbors = self.nsearch(x_in, x_in)
        x = self.conv1(x_in, neighbors)
        x = self.conv2(x, neighbors)
        x = self.to_output(x)
        return x

    def data_dict_to_input(self, data_dict, **kwargs):
        vert = data_dict["vertices"].to(self.device)
        return vert

    @torch.no_grad()
    def eval_dict(self, data_dict, loss_fn=None, decode_fn=None):
        vert = self.data_dict_to_input(data_dict)
        batch_size = vert.shape[0]
        pred_out = self(vert)
        pred_out = pred_out.reshape(batch_size, -1)
        if loss_fn is None:
            loss_fn = self.loss
        gt_out = data_dict["pressure"].to(self.device).reshape(batch_size, -1)
        out_dict = {"l2": loss_fn(pred_out, gt_out)}
        if decode_fn is not None:
            pred_out = decode_fn(pred_out)
            gt_out = decode_fn(gt_out)
            loss_decoded = loss_fn(pred_out, gt_out)
            out_dict["l2_decoded"] = loss_decoded
        return out_dict

    def loss_dict(self, data_dict, loss_fn=None, **kwargs):
        vert = self.data_dict_to_input(data_dict)
        batch_size = vert.shape[0]
        pred_pressure = self(vert)
        pred_pressure = pred_pressure.reshape(batch_size, -1)
        if loss_fn is None:
            loss_fn = self.loss
        gt_pressure = data_dict["pressure"].to(self.device).reshape(batch_size, -1)
        return {"loss": loss_fn(pred_pressure, gt_pressure)}
