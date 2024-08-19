import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv
from .base_model import BaseModel
from .net_utils import MLP

class GNO(BaseModel):
    def __init__(self, in_channel=3, width=32, mid_width=64, out_channel=3, r=0.1):
        super().__init__()

        kernel1 = MLP(
            [in_channel * 2, mid_width // 2, mid_width, width**2], torch.nn.GELU
        )
        kernel2 = MLP(
            [in_channel * 2, mid_width // 2, mid_width, width**2], torch.nn.GELU
        )
        kernel3 = MLP(
            [in_channel * 2, mid_width // 2, mid_width, width**2], torch.nn.GELU
        )
        kernel4 = MLP(
            [in_channel * 2, mid_width // 2, mid_width, width**2], torch.nn.GELU
        )
        self.conv1 = NNConv(width, width, kernel1, aggr="mean")
        self.conv2 = NNConv(width, width, kernel2, aggr="mean")
        self.conv3 = NNConv(width, width, kernel3, aggr="mean")
        self.conv4 = NNConv(width, width, kernel4, aggr="mean")
        self.to_output = torch.nn.Linear(width, out_channel)

        self.linear = nn.Linear(in_channel, width)
        self.activation = F.gelu
        self.r = r

    def get_graph(self, x_in, x_out=None):
        if x_out is None:
            x_in = x_in.squeeze()
            pwd = torch.cdist(x_in, x_in).squeeze()
            edge_index = torch.stack(torch.where(pwd <= self.r))
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=x_in.device)
            edge_attr = torch.cat([x_in[edge_index[0].T], x_in[edge_index[1].T]], dim=-1)
        else:
            x_in = x_in.squeeze()
            x_out = x_out.squeeze()
            N_in = x_in.shape[0]
            pwd = torch.cdist(x_in, x_out).squeeze()
            edge_index = torch.stack(torch.where(pwd <= self.r))
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=x_in.device)
            edge_attr = torch.cat([x_in[edge_index[0].T], x_out[edge_index[1].T]], dim=-1)
            edge_index[1, :] = edge_index[1, :] + N_in
        return edge_index.detach(), edge_attr.detach()

    def forward(self, u_in, x_in=None, x_out=None):
        """
        u_in: (N_in, C)
        x_in: (N_in, d) or None
        When both x_in and x_out are None, the first input is just vectices and there's no feature associated to the vertices.
        Synthesize features.
        x_out: (N_out, d) or None
        """
        if x_in is None:
            x_in = u_in
            u_in = self.linear(x_in)
        if x_out is None:
            edge_index, edge_attr = self.get_graph(x_in)
            u = u_in
        else:
            N_in = x_in.shape[1]
            edge_index, edge_attr = self.get_graph(x_in, x_out)
            u_out = self.linear(x_out)  # x_out (B, N_out, C)
            u = torch.cat([u_in, u_out], dim=1)  # x_out (B, N_in + N_out, C)

        u = u.squeeze()
        u = self.conv1(u, edge_index, edge_attr)
        u = self.activation(u)
        u = self.conv2(u, edge_index, edge_attr)
        u = self.activation(u)
        u = self.conv3(u, edge_index, edge_attr)
        u = self.activation(u)
        u = self.conv4(u, edge_index, edge_attr)
        u = u.unsqueeze(0)

        if x_out is not None:
            u = u[:, N_in:, :]

        u_out = self.to_output(u)
        return u_out

    def data_dict_to_input(self, data_dict, **kwargs):
        vert = data_dict["vertices"].to(self.device)
        return vert

    @torch.no_grad()
    def eval_dict(self, data_dict, loss_fn=None, decode_fn=None):
        vert = self.data_dict_to_input(data_dict)
        pred_out = self(vert)
        gt_out = data_dict["pressure"].to(self.device)
        out_dict = {"l2": loss_fn(pred_out, gt_out)}
        if decode_fn is not None:
            pred_out = decode_fn(pred_out)
            gt_out = decode_fn(gt_out)
            out_dict["l2_decoded"] = loss_fn(pred_out, gt_out)
        return out_dict

    def loss_dict(self, data_dict, loss_fn=None):
        vert = self.data_dict_to_input(data_dict)
        pressure = self(vert)
        if loss_fn is None:
            loss_fn = self.loss
        return {"loss": loss_fn(pressure, data_dict["pressure"].to(self.device))}
