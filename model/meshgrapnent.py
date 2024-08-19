import torch
import numpy as np
import torch.nn as nn
import torch
from torch_scatter import scatter_sum


class MLP(nn.Module):
    def __init__(self, input_size, output_size=128, layer_norm=True, n_hidden=2, hidden_size=128):
        super(MLP, self).__init__()
        if hidden_size == 0:
            f = [nn.Linear(input_size, output_size)]
        else:
            f = [nn.Linear(input_size, hidden_size), nn.ReLU()]
            h = 1
            for i in range(h, n_hidden):
                f.append(nn.Linear(hidden_size, hidden_size))
                f.append(nn.ReLU())
            f.append(nn.Linear(hidden_size, output_size))
            if layer_norm:
                f.append(nn.LayerNorm(output_size))

        self.f = nn.Sequential(*f)

    def forward(self, x):
        return self.f(x)


class GNN(nn.Module):
    def __init__(self, n_hidden=2, node_size=128, edge_size=128, output_size=None, layer_norm=False):
        super(GNN, self).__init__()
        output_size = output_size or node_size
        self.f_edge = MLP(input_size=edge_size + node_size * 2, n_hidden=n_hidden, layer_norm=layer_norm,
                          output_size=edge_size)
        self.f_node = MLP(input_size=edge_size + node_size, n_hidden=n_hidden, layer_norm=layer_norm,
                          output_size=output_size)

    def forward(self, V, E, edges):
        senders = torch.gather(V, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, V.shape[-1]))
        receivers = torch.gather(V, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, V.shape[-1]))

        edge_inpt = torch.cat([senders, receivers, E], dim=-1)
        edge_embeddings = self.f_edge(edge_inpt)

        col = edges[..., 0].unsqueeze(-1).repeat(1, 1, edge_embeddings.shape[-1])
        edge_sum = scatter_sum(edge_embeddings, col, dim=-2)

        node_inpt = torch.cat([V, edge_sum], dim=-1)
        node_embeddings = self.f_node(node_inpt)

        return node_embeddings, edge_embeddings

class meshgrapnent(nn.Module):
    def __init__(self, 
                 N=15, 
                 state_size=1,
                 ): 
        super(meshgrapnent, self).__init__()

        self.encoder = Encoder(12)
        self.processor = Processor(N)
        self.decoder = MLP(input_size=128, output_size=state_size, layer_norm=False)
        
    def forward(self, node_pos, areas, edges, info):

        N = node_pos.shape[1]
        info = info.expand(-1, N, -1)
        en_in = torch.cat((node_pos, areas, info), dim=-1)

        V, E = self.encoder(node_pos, edges, en_in)
        V, E = self.processor(V, E, edges)
        next_output = self.decoder(V)
        
        return next_output

class Encoder(nn.Module):
    def __init__(self, state_size):
        super(Encoder, self).__init__()

        self.fv = MLP(input_size=12)
        self.fe = MLP(input_size=4)

    def forward(self, mesh_pos, edges, V):
        # Get edges attr
        senders = torch.gather(mesh_pos, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, 3))
        receivers = torch.gather(mesh_pos, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, 3))

        distance = senders - receivers
        norm = torch.sqrt((distance ** 2).sum(-1, keepdims=True))
        E = torch.cat([distance, norm], dim=-1)

        V = self.fv(V)
        E = self.fe(E)

        return V, E


class Processor(nn.Module):
    def __init__(self, N=15):
        super(Processor, self).__init__()
        self.gnn = nn.ModuleList([])
        for i in range(N):
            self.gnn.append(GNN())

    def forward(self, V, E, edges):
        for i, gn in enumerate(self.gnn):
            edges = edges
            v, e = gn(V, E, edges)
            V = V + v
            E = E + e

        V = V
        E = E
        return V, E


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.delta = MLP(input_size=128, output_size=1, layer_norm=False)

    def forward(self, V):
        
        output = self.delta(V)
        
        
        return output
