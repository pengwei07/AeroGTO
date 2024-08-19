import torch
import torch.nn as nn
from torch_scatter import scatter_mean

class MLP(nn.Module): 
    def __init__(self, 
                input_size=256, 
                output_size=256, 
                layer_norm=True, 
                n_hidden=2, 
                hidden_size=256, 
                act = 'PReLU',
                ):
        super(MLP, self).__init__()
        if act == 'GELU':
            self.act = nn.GELU()
        elif act == 'SiLU':
            self.act = nn.SiLU()
        elif act == 'PReLU':
            self.act = nn.PReLU()
            
        if hidden_size == 0:
            f = [nn.Linear(input_size, output_size)]
        else:
            f = [nn.Linear(input_size, hidden_size), self.act]
            h = 1
            for i in range(h, n_hidden):
                f.append(nn.Linear(hidden_size, hidden_size))
                f.append(self.act)
            f.append(nn.Linear(hidden_size, output_size))
            if layer_norm:
                f.append(nn.LayerNorm(output_size))

        self.f = nn.Sequential(*f)

    def forward(self, x):
        return self.f(x)
    
class GNN(nn.Module):
    def __init__(self, n_hidden=1, node_size=128, edge_size=128, output_size=None, layer_norm=False):
        super(GNN, self).__init__()
        output_size = output_size or node_size
        
        self.f_edge = MLP(input_size=edge_size + node_size * 2, n_hidden=n_hidden, layer_norm=layer_norm, act = 'GELU', output_size=edge_size)
        
        self.f_node = MLP(input_size=edge_size + node_size, n_hidden=n_hidden, layer_norm=layer_norm, act = 'GELU', output_size=output_size)

    def forward(self, V, E, edges):
        
        edges = edges.long()
        senders = torch.gather(V, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, V.shape[-1]))
        receivers = torch.gather(V, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, V.shape[-1]))

        edge_inpt = torch.cat([senders, receivers, E], dim=-1)
        edge_embeddings = self.f_edge(edge_inpt)

        col = edges[..., 1].unsqueeze(-1).repeat(1, 1, edge_embeddings.shape[-1])
        edge_sum = scatter_mean(edge_embeddings, col, dim=-2, dim_size=V.shape[1])

        node_inpt = torch.cat([V, edge_sum], dim=-1)
        node_embeddings = self.f_node(node_inpt)

        return node_embeddings, edge_embeddings
    
class Encoder(nn.Module): 
    def __init__(self, 
                 state_embedding_dim = 128,
                 ):
        super(Encoder, self).__init__()
        
        self.state_embedding_dim = state_embedding_dim
        
        self.enc_s_dim = 128
        self.enc_a_dim = 128
        self.enc_i_dim = 64

        self.fv_1 = MLP(input_size = 4, output_size=state_embedding_dim, act = 'SiLU', layer_norm = False)
        
        self.enc_s = MLP(input_size = 3 + 42, output_size = self.enc_s_dim, act = 'SiLU', layer_norm = False)
        self.enc_a = MLP(input_size = 1 + 14, output_size = self.enc_a_dim, act = 'SiLU',layer_norm = False)
        self.enc_info = MLP(input_size = 8 + 48, output_size = self.enc_i_dim, act = 'SiLU', layer_norm = False)

        self.fv_2 = MLP(input_size = state_embedding_dim + self.enc_s_dim + self.enc_a_dim + self.enc_i_dim, output_size=state_embedding_dim, layer_norm = False)

    def FourierEmbedding(self, pos, pos_start, pos_length):
        # F(x) = [cos(2^i * pi * x), sin(2^i * pi * x)], i = -3, -2, -1, 0, 1, 2, 3
        original_shape = pos.shape
        new_pos = pos.reshape(-1, original_shape[-1])
        index = torch.arange(pos_start, pos_start + pos_length, device=pos.device)
        index = index.float()
        freq = 2 ** index * torch.pi
        cos_feat = torch.cos(freq.view(1, 1, -1) * new_pos.unsqueeze(-1))
        sin_feat = torch.sin(freq.view(1, 1, -1) * new_pos.unsqueeze(-1))
        embedding = torch.cat([cos_feat, sin_feat], dim=-1)
        embedding = embedding.view(*original_shape[:-1], -1)
        all_embeddings = torch.cat([embedding, pos], dim=-1)
        
        return all_embeddings

    def forward(self, node_pos, areas, info):
        
        # node_pos.dim = [b,n,3]
        # areas.dim = [b,n,1]
        # info.dim = [b,1,8]
        N = node_pos.shape[1]
        
        V_1 = self.fv_1(torch.cat([node_pos, areas], dim=-1)) # dim = [B, N, C]
        
        # fourier embedding
        F_pos = self.FourierEmbedding(node_pos, -3, 7) # 2 * 7 * 3 + 3 = 45
        enc_s = self.enc_s(F_pos)
        
        F_area = self.FourierEmbedding(areas, -3, 7) # 2 * 7 * 1 + 1 = 15
        enc_a = self.enc_a(F_area)
        
        F_info = self.FourierEmbedding(info, -1, 3) # 2 * 3 * 8 + 8 = 56
        fea_3 = self.enc_info(F_info)
        enc_i = fea_3.expand(-1, N, -1)
        
        V_in = self.fv_2(torch.cat([V_1, enc_s, enc_a, enc_i], dim=-1)) # dim = [B, N, C]

        return V_in, F_pos
    
class AttentionBlock(nn.Module): 
    def __init__(self, 
                n_token = 64,
                w_size = 128, 
                n_heads = 4,
                ):
        super(AttentionBlock, self).__init__()
        
        self.channel_dim = w_size
        self.n_token = n_token
        
        self.softmax = nn.Softmax(dim=-1)
        self.scale = self.channel_dim ** -0.5
        self.n_heads = n_heads
        
        # 1. 
        self.Q = nn.Parameter(torch.randn(self.n_token, self.channel_dim), requires_grad=True)
        
        self.to_q_1 = nn.Linear(self.channel_dim, self.channel_dim)
        self.to_k_1 = nn.Linear(self.channel_dim, self.channel_dim)
        self.to_v_1 = nn.Linear(self.channel_dim, self.channel_dim)
        
        # 2. 
        self.attention2 = nn.MultiheadAttention(embed_dim=w_size, num_heads=n_heads, batch_first=True)
        
        # 3.
        self.to_q_2 = nn.Linear(self.channel_dim, self.channel_dim)
        self.to_k_2 = nn.Linear(self.channel_dim, self.channel_dim)
        self.to_v_2 = nn.Linear(self.channel_dim, self.channel_dim)

    def forward(self, W_0):
        
        # 1. transform decoder
        B = W_0.shape[0]
        learned_Q = self.Q.unsqueeze(0).expand(B, -1, -1)
        
        Q_1 = self.to_q_1(learned_Q)
        K_1 = self.to_k_1(W_0)
        V_1 = self.to_v_1(W_0)

        attn1 = self.softmax(torch.einsum('bmc, bnc -> bmn', Q_1, K_1) * self.scale)
        W_1 = torch.matmul(attn1, V_1)  
    
        # 2. self-attention
        W_2, _ = self.attention2(W_1, W_1, W_1)
        
        # 3. transform decoder
        Q_2 = self.to_q_2(W_0)
        K_2 = self.to_k_2(W_2)
        V_2 = self.to_v_2(W_2)
                
        attn2 = self.softmax(torch.einsum('bnc, bmc -> bnm', Q_2, K_2) * self.scale)
        W_3 = torch.matmul(attn2, V_2)  

        return W_3  
    
class MixerBlock(nn.Module):
    def __init__(self, state_embedding_dim, att_embedding_dim, n_head, n_token, enc_s_dim=0, idx = 0, n_blocks = 4):
        super(MixerBlock, self).__init__()

        self.enc_s_dim =  enc_s_dim
        node_size = state_embedding_dim if enc_s_dim == 0 else state_embedding_dim + 45
        self.gnn = GNN(node_size=node_size, edge_size=state_embedding_dim, output_size=state_embedding_dim, layer_norm=True)

        self.ln1 = nn.LayerNorm(att_embedding_dim)
        self.ln2 = nn.LayerNorm(att_embedding_dim)
        self.linear = nn.Linear(att_embedding_dim, att_embedding_dim)
        self.MHA = AttentionBlock(n_token=n_token, w_size=att_embedding_dim, n_heads=n_head)

        self.idx = idx
        self.n_blocks = n_blocks

        if self.idx < self.n_blocks:
            self.alpha_1 = nn.Parameter(torch.zeros(1).float(), requires_grad=True)
            self.alpha_2 = nn.Parameter(torch.zeros(1).float(), requires_grad=True)
            self.alpha_3 = nn.Parameter(torch.zeros(1).float(), requires_grad=True)
            self.alpha_4 = nn.Parameter(torch.zeros(1).float(), requires_grad=True)
        else:
            self.alpha_1 = nn.Parameter(torch.zeros(1).float(), requires_grad=True)
            self.alpha_3 = nn.Parameter(torch.zeros(1).float(), requires_grad=True)
            self.alpha_4 = nn.Parameter(torch.zeros(1).float(), requires_grad=True)
            
        self.act = nn.SiLU()
        
    def forward(self, V, E, edges, F_space):
        
        if self.enc_s_dim > 0:
            V_in = torch.cat([V, F_space], dim=-1)
        else:
            V_in = V

        v, e = self.gnn(V_in, E, edges)
        
        alpha_1 = self.act(self.alpha_1)
        V = V + alpha_1 * v
        
        if self.idx < self.n_blocks:
            alpha_2 = self.act(self.alpha_2)
            E = E + alpha_2 * e
        else:
            E = E + e

        W_1 = self.MHA(self.ln1(V))
        
        alpha_3 = self.act(self.alpha_3)
        W_2 = V + alpha_3 * W_1
        
        alpha_4 = self.act(self.alpha_4)
        W_3 = W_2 + alpha_4 * self.linear(self.ln2(W_2))

        return W_3, E

class Processor(nn.Module):
    def __init__(self, N, state_embedding_dim, att_embedding_dim, n_head, n_token):
        super(Processor, self).__init__()

        self.fe = MLP(input_size=4, output_size=state_embedding_dim, n_hidden=1, act = 'SiLU', layer_norm=False)
        
        self.blocks = nn.ModuleList([
            MixerBlock(state_embedding_dim=state_embedding_dim, att_embedding_dim=att_embedding_dim, 
                       n_head=n_head, n_token=n_token, enc_s_dim=0 if i != (N//2) else 1, idx = int(i), n_blocks = N-1)
            for i in range(N)
        ])

    def forward(self, V, edges, node_pos, F_space):
        edges = edges.long()

        senders = torch.gather(node_pos, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, node_pos.shape[-1]))
        receivers = torch.gather(node_pos, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, node_pos.shape[-1]))
        distance = receivers - senders
        norm = torch.sqrt((distance ** 2).sum(-1, keepdims=True))
        E = torch.cat([distance, norm], dim=-1)
        E = self.fe(E)

        for block in self.blocks:
            V, E = block(V, E, edges, F_space)

        return V

class Decoder(nn.Module):
    def __init__(self,
                 state_embedding_dim=128, 
                 state_size=1):
        super(Decoder, self).__init__()
        
        self.linear_1 = nn.Linear(state_embedding_dim, state_embedding_dim)
        self.prelu_1 = nn.PReLU()

        self.linear_2 = nn.Linear(state_embedding_dim, state_embedding_dim)
        self.prelu_2 = nn.PReLU()
                
        self.output_layer = nn.Linear(state_embedding_dim, state_size)
        
    def forward(self, V_in):

        V_out = self.prelu_1(self.linear_1(V_in))
        V_out = self.prelu_2(self.linear_2(V_out))
        final_state_node = self.output_layer(V_out)
        
        return final_state_node

class Model(nn.Module):
    def __init__(self, 
                N_block = 4, 
                state_embedding_dim = 128, 
                att_embedding_dim = 256,
                n_head = 4,
                n_token = 64,
                ):
        super(Model, self).__init__()

        self.encoder = Encoder(
            state_embedding_dim = state_embedding_dim,
            )

        self.processor = Processor(
            N= N_block, 
            state_embedding_dim = state_embedding_dim,
            att_embedding_dim = att_embedding_dim,
            n_head = n_head,
            n_token = n_token
        )
        
        self.decoder = Decoder(
            state_embedding_dim = state_embedding_dim,
            )
 
    def forward(self, node_pos, areas, edges, info):
        
        # 1. Encoder
        V_1, F_space = self.encoder(node_pos, areas, info)
        
        # 2. Processor
        V_2 = self.processor(V_1, edges, node_pos, F_space)
        
        # 3. Decoder
        V_3 = self.decoder(V_2)

        return V_3
    
