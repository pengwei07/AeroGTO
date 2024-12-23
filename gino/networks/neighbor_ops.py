# from typing import Optional
# import unittest
# from torchtyping import TensorType

# import torch

# from torch_scatter import segment_csr
# from .net_utils import MLP

# # import open3d.ml.torch as ml3d
# # NeighborSearchReturnType = ml3d.python.return_types.fixed_radius_search


# # class NeighborSearchLayer(torch.nn.Module):
# #     def __init__(self, radius: float):
# #         super().__init__()
# #         self.radius = radius
# #         self.nsearch = ml3d.layers.FixedRadiusSearch()

# #     def forward(
# #         self, inp_positions: TensorType["N", 3], out_positions: TensorType["M", 3]
# #     ) -> NeighborSearchReturnType:
# #         neighbors = self.nsearch(inp_positions, out_positions, self.radius)
# #         return neighbors

# class NeighborSearchLayer(torch.nn.Module):
#     def __init__(self, radius):
#         super().__init__()
#         self.radius = radius

#     def forward(self, points, queries):
#         device = points.device
#         # 计算queries和points之间的距离
#         d = torch.cdist(queries, points)
#         # 找到距离小于等于radius的点对的索引
#         valid_pairs = torch.where(d <= self.radius)

#         # 初始化neighbors_index
#         neighbors_index = valid_pairs[1]

#         # 计算每个query点的邻居数量
#         count_num = torch.zeros(queries.shape[0], device=device).scatter_add_(0, valid_pairs[0],
#                                                                               torch.ones_like(valid_pairs[0],
#                                                                                               device=device,
#                                                                                               dtype=torch.float))

#         # 计算neighbors_row_splits
#         neighbors_row_splits = torch.cumsum(torch.cat((torch.zeros(1, device=device), count_num)), dim=0).to(
#             torch.int64)

#         return [neighbors_index, neighbors_row_splits]
    
    
    


# class NeighborPoolingLayer(torch.nn.Module):
#     def __init__(self, reduction="mean"):
#         super().__init__()
#         self.reduction = reduction

#     def forward(
#         self, in_features: TensorType["N", "C"], neighbors: NeighborSearchReturnType
#     ) -> TensorType["M", "C"]:
#         """
#         inp_positions: [N,3]
#         out_positions: [M,3]
#         inp_features: [N,C]
#         neighbors: ml3d.layers.FixedRadiusSearchResult. If None, will be computed. For the same inp_positions and out_positions, this can be reused.
#         """
#         rep_features = in_features[neighbors.neighbors_index.long()]
#         out_features = segment_csr(
#             rep_features, neighbors.neighbors_row_splits, reduce=self.reduction
#         )
#         return out_features


# class NeighborMLPConvLayer(torch.nn.Module):
#     def __init__(
#         self, mlp=None, in_channels=8, hidden_dim=32, out_channels=32, reduction="mean"
#     ):
#         super().__init__()
#         self.reduction = reduction
#         if mlp is None:
#             mlp = MLP(
#             [2 * in_channels, hidden_dim,out_channels], torch.nn.GELU
#             )
#         self.mlp = mlp

#     def forward(
#         self,
#         in_features: TensorType["N", "C_in"],
#         neighbors: NeighborSearchReturnType,
#         out_features: Optional[TensorType["M", "C_in"]] = None,
#     ) -> TensorType["M", "C_out"]:
#         """
#         inp_features: [N,C]
#         outp_features: [M,C]
#         neighbors: ml3d.layers.FixedRadiusSearchResult.
#         """
#         if out_features is None:
#             out_features = in_features

#         assert in_features.shape[1] + out_features.shape[1] == self.mlp.layers[0].in_features
#         rep_features = in_features[neighbors.neighbors_index.long()]
#         rs = neighbors.neighbors_row_splits
#         num_reps = rs[1:] - rs[:-1]
#         # repeat the self features using num_reps
#         self_features = torch.repeat_interleave(out_features, num_reps, dim=0)
#         agg_features = torch.cat([rep_features, self_features], dim=1)
#         rep_features = self.mlp(agg_features)
#         out_features = segment_csr(
#             rep_features, neighbors.neighbors_row_splits, reduce=self.reduction
#         )
#         return out_features

# class NeighborMLPConvLayerWeighted(torch.nn.Module):
#     def __init__(
#         self, mlp=None, in_channels=8, hidden_dim=32, out_channels=32, reduction="mean"
#     ):
#         super().__init__()
#         self.reduction = reduction
#         if mlp is None:
#             mlp = MLP(
#             [2 * in_channels, hidden_dim,out_channels], torch.nn.GELU
#             )
#         self.mlp = mlp
    
#     ########################################################
#     # @torch.no_grad()
#     ########################################################
#     def forward(
#         self,
#         in_features: TensorType["N", "C_in"],
#         neighbors: NeighborSearchReturnType,
#         out_features: Optional[TensorType["M", "C_in"]] = None,
#         in_weights:  Optional[TensorType["N"]] = None,
#     ) -> TensorType["M", "C_out"]:
#         """
#         in_features: [N,C]
#         out_features: [M,C]
#         in_weights: [N]
#         neighbors: ml3d.layers.FixedRadiusSearchResult.
#         """
#         if out_features is None:
#             out_features = in_features

#         assert in_features.shape[1] + out_features.shape[1] == self.mlp.layers[0].in_features
#         rep_features = in_features[neighbors.neighbors_index.long()]
#         if in_weights is None:
#             rep_weights = 1
#         else:
#             rep_weights = in_weights[neighbors.neighbors_index.long()].unsqueeze(-1)
#         rs = neighbors.neighbors_row_splits
#         num_reps = rs[1:] - rs[:-1]
#         # repeat the self features using num_reps
#         self_features = torch.repeat_interleave(out_features, num_reps, dim=0)
#         agg_features = torch.cat([rep_features, self_features], dim=1)
#         rep_features = rep_weights * self.mlp(agg_features)
#         out_features = segment_csr(
#             rep_features, neighbors.neighbors_row_splits, reduce=self.reduction
#         )
#         return out_features

# class NeighborMLPConvLayerLinear(torch.nn.Module):
#     def __init__(
#             self, mlp=None, in_channels=8, hidden_dim=32, out_channels=32, reduction="mean"
#     ):
#         super().__init__()
#         # if linear_kernel=False
#         #   out_features = sum k([in_features, out_features])
#         # if linear_kernel=True
#         #   out_features = sum k([x_in, x_out]) * in_features

#         self.reduction = reduction
#         if mlp is None:
#             mlp = MLP(
#                 [2 * in_channels, hidden_dim, out_channels], torch.nn.GELU
#             )
#         self.mlp = mlp

#     ########################################################
#     # @torch.no_grad()
#     ########################################################
#     def forward(
#             self,
#             x_in: TensorType["N", "3"],
#             neighbors: NeighborSearchReturnType,
#             in_features: TensorType["N", "C"],
#             x_out: Optional[TensorType["M", "3"]] = None,
#     ) -> TensorType["M", "C_out"]:
#         """
#         inp_features: [N,C]
#         outp_features: [M,C]
#         neighbors: ml3d.layers.FixedRadiusSearchResult.
#         """
#         if x_out is None:
#             x_out = x_in

#         assert x_in.shape[1] + x_out.shape[1] == self.mlp.layers[0].in_features
#         rep_features = x_in[neighbors.neighbors_index.long()]
#         in_features = in_features[neighbors.neighbors_index.long()]
#         rs = neighbors.neighbors_row_splits
#         num_reps = rs[1:] - rs[:-1]
#         # repeat the self features using num_reps
#         self_features = torch.repeat_interleave(x_out, num_reps, dim=0)
#         agg_features = torch.cat([rep_features, self_features], dim=1)
#         rep_features = self.mlp(agg_features) # (N, C)
#         rep_features = rep_features*in_features # (N, C) * (N, C) -> (N, C)

#         out_features = segment_csr(
#             rep_features, neighbors.neighbors_row_splits, reduce=self.reduction
#         )
#         return out_features



# class TestNeighborSearch(unittest.TestCase):
#     def setUp(self) -> None:
#         self.N = 10000
#         self.device = "cuda:0"
#         return super().setUp()

#     def test_neighbor_search(self):
#         inp_positions = torch.randn([self.N, 3]).to(self.device) * 10
#         inp_features = torch.randn([self.N, 8]).to(self.device)
#         out_positions = inp_positions

#         neighbors = NeighborSearchLayer(1.2)(inp_positions, out_positions)
#         pool = NeighborPoolingLayer(reduction="mean")
#         out_features = pool(inp_features, neighbors)

#     def test_mlp_conv(self):
#         out_N = 1000
#         radius = 1.2
#         in_positions = torch.randn([self.N, 3]).to(self.device) * 10
#         out_positions = torch.randn([out_N, 3]).to(self.device) * 10
#         in_features = torch.randn([self.N, 8]).to(self.device)
#         out_features = torch.randn([out_N, 8]).to(self.device)

#         neighbors = NeighborSearchLayer(radius)(in_positions, out_positions)
#         conv = NeighborMLPConvLayer(reduction="mean").to(self.device)
#         out_features = conv(in_features, neighbors, out_features=out_features)


# if __name__ == "__main__":
#     unittest.main()



from typing import Optional
from torchtyping import TensorType

import torch
from torch_scatter import segment_csr
from .net_utils import MLP


class NeighborSearchLayer(torch.nn.Module):
    def __init__(self, radius):
        super().__init__()
        self.radius = radius

    def forward(self, points, queries):
        device = points.device
        # Calculate the distance between queries and points
        d = torch.cdist(queries, points)
        # Find the index of a pair of points whose distance is less than or equal to radius
        valid_pairs = torch.where(d <= self.radius)
        # Initialize the neighbors_index
        neighbors_index = valid_pairs[1]
        # The number of neighbors of each query point is counted
        count_num = torch.zeros(queries.shape[0], device=device).scatter_add_(0, valid_pairs[0],
                                                                              torch.ones_like(valid_pairs[0],
                                                                                              device=device,
                                                                                              dtype=torch.float))

        # 计算neighbors_row_splits
        neighbors_row_splits = torch.cumsum(torch.cat((torch.zeros(1, device=device), count_num)), dim=0).to(
            torch.int64)

        return [neighbors_index, neighbors_row_splits]


# class NeighborPoolingLayer(torch.nn.Module):
#     def __init__(self, reduction="mean"):
#         super().__init__()
#         self.reduction = reduction
#     def forward(
#         self, in_features: TensorType["N", "C"], neighbors: list
#     ) -> TensorType["M", "C"]:
#         """
#         inp_positions: [N,3]
#         out_positions: [M,3]
#         inp_features: [N,C]
#         neighbors: list
#         """
#         rep_features = in_features[neighbors[0]]
#         out_features = segment_csr(
#             rep_features, neighbors[1], reduce=self.reduction
#         )
#         return out_features


class NeighborMLPConvLayer(torch.nn.Module):
    def __init__(
        self, mlp=None, in_channels=3, hidden_dim=32, out_channels=32, reduction="mean"
    ):
        super().__init__()
        self.reduction = reduction
        if mlp is None:
            mlp = MLP(
            [2 * in_channels, hidden_dim,out_channels], torch.nn.GELU
            )
        self.mlp = mlp

    def forward(
        self,
        in_features: TensorType["B", "N", "C_in"],
        neighbors: list,
        out_features: Optional[TensorType["B", "M", "C_in"]] = None,
    ) -> TensorType["B", "M", "C_out"]:
        """
        inp_features: [N,C]
        outp_features: [M,C]
        neighbors: list.
        """
        if out_features is None:
            out_features = in_features

        assert in_features.shape[2] + out_features.shape[2] == self.mlp.layers[0].in_features
        rep_features = in_features[:,neighbors[0]]
        rs = neighbors[1]
        num_reps = rs[1:] - rs[:-1]
        # repeat the self features using num_reps
        self_features = torch.repeat_interleave(out_features, num_reps, dim=1)
        agg_features = torch.cat([rep_features, self_features], dim=2)
        rep_features = self.mlp(agg_features)

        rep_features = rep_features.permute(1, 0, 2)
        out_features = segment_csr(
            rep_features, neighbors[1], reduce=self.reduction
        )
        out_features = out_features.permute(1, 0, 2)
        return out_features

class NeighborMLPConvLayerWeighted(torch.nn.Module):
    def __init__(
        self, mlp=None, in_channels=8, hidden_dim=32, out_channels=32, reduction="mean"
    ):
        super().__init__()
        self.reduction = reduction
        if mlp is None:
            mlp = MLP(
            [2 * in_channels, hidden_dim,out_channels], torch.nn.GELU
            )
        self.mlp = mlp
    
    ########################################################
    # @torch.no_grad()
    ########################################################
    def forward(
        self,
        in_features: TensorType["N", "C_in"],
        neighbors: list,
        out_features: Optional[TensorType["B", "M", "C_in"]] = None,
        in_weights:  Optional[TensorType["N"]] = None,
    ) -> TensorType["B", "M", "C_out"]:
        """
        in_features: [N,C]
        out_features: [M,C]
        in_weights: [N]
        neighbors: list.
        """
        if out_features is None:
            out_features = in_features
        assert in_features.shape[1] + out_features.shape[2] == self.mlp.layers[0].in_features
        rep_features = in_features[neighbors[0]]
        if in_weights is None:
            rep_weights = 1
        else:
            rep_weights = in_weights[neighbors[0]].unsqueeze(-1)
        rs = neighbors[1]
        num_reps = rs[1:] - rs[:-1]
        # repeat the self features using num_reps
        self_features = torch.repeat_interleave(out_features, num_reps, dim=1)
        rep_features = torch.repeat_interleave(rep_features.unsqueeze(0), self_features.shape[0], dim=0)
        agg_features = torch.cat([rep_features, self_features], dim=2)
        rep_features = rep_weights * self.mlp(agg_features)

        rep_features = rep_features.permute(1, 0, 2)
        out_features = segment_csr(
            rep_features, neighbors[1], reduce=self.reduction
        )
        out_features = out_features.permute(1, 0, 2)
        return out_features

class NeighborMLPConvLayerLinear(torch.nn.Module):
    def __init__(
            self, mlp=None, in_channels=8, hidden_dim=32, out_channels=32, reduction="mean"
    ):
        super().__init__()
        # if linear_kernel=False
        #   out_features = sum k([in_features, out_features])
        # if linear_kernel=True
        #   out_features = sum k([x_in, x_out]) * in_features

        self.reduction = reduction
        if mlp is None:
            mlp = MLP(
                [2 * in_channels, hidden_dim, out_channels], torch.nn.GELU
            )
        self.mlp = mlp

    ########################################################
    # @torch.no_grad()
    ########################################################
    def forward(
            self,
            x_in: TensorType["N", "2"],
            neighbors: list,
            in_features: TensorType["B", "N", "C"],
            x_out: Optional[TensorType["M", "2"]] = None,
    ) -> TensorType["B", "M", "C_out"]:
        """
        inp_features: [N,C]
        outp_features: [M,C]
        neighbors: list.
        """
        if x_out is None:
            x_out = x_in
        assert x_in.shape[1] + x_out.shape[1] == self.mlp.layers[0].in_features
        rep_features = x_in[neighbors[0]]
        in_features = in_features[:, neighbors[0]]
        rs = neighbors[1]
        num_reps = rs[1:] - rs[:-1]
        # repeat the self features using num_reps
        self_features = torch.repeat_interleave(x_out, num_reps, dim=0)
        agg_features = torch.cat([rep_features, self_features], dim=1)
        rep_features = self.mlp(agg_features) # (N, C)
        rep_features = rep_features.unsqueeze(0)*in_features # (N, C) * (N, C) -> (N, C)
        rep_features = rep_features.permute(1,0,2)
        out_features = segment_csr(
            rep_features, neighbors[1], reduce=self.reduction
        )
        out_features = out_features.permute(1,0,2)
        return out_features


if __name__ == "__main__":
    pass
