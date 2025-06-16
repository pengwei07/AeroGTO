import sys
import numpy as np
sys.path.append("/workspace/AeroGTO/paddle_project")
sys.path.append("/workspace/AeroGTO/paddle_project/src/paddle_scatter")
import paddle
from paddle_utils import *
from paddle.io import DataLoader, Dataset
# from torch_scatter import scatter_mean
from paddle_scatter import scatter_mean  # type: ignore
from src.pressure_dataset import Pressure_Dataset


class MLP(paddle.nn.Layer):
    def __init__(
        self,
        input_size=256,
        output_size=256,
        layer_norm=True,
        n_hidden=2,
        hidden_size=256,
        act="PReLU",
    ):
        super(MLP, self).__init__()
        if act == "GELU":
            self.act = paddle.nn.GELU()
        elif act == "SiLU":
            self.act = paddle.nn.Silu()
        elif act == "PReLU":
            self.act = paddle.nn.PReLU()
        if hidden_size == 0:
            f = [paddle.nn.Linear(in_features=input_size, out_features=output_size)]
        else:
            f = [
                paddle.nn.Linear(in_features=input_size, out_features=hidden_size),
                self.act,
            ]
            h = 1
            for i in range(h, n_hidden):
                f.append(
                    paddle.nn.Linear(in_features=hidden_size, out_features=hidden_size)
                )
                f.append(self.act)
            f.append(
                paddle.nn.Linear(in_features=hidden_size, out_features=output_size)
            )
            if layer_norm:
                f.append(paddle.nn.LayerNorm(normalized_shape=output_size))
        self.f = paddle.nn.Sequential(*f)

    def forward(self, x):
        return self.f(x)


class GNN(paddle.nn.Layer):
    def __init__(
        self,
        n_hidden=1,
        node_size=128,
        edge_size=128,
        output_size=None,
        layer_norm=False,
    ):
        super(GNN, self).__init__()
        output_size = output_size or node_size
        self.f_edge = MLP(
            input_size=edge_size + node_size * 2,
            n_hidden=n_hidden,
            layer_norm=layer_norm,
            act="GELU",
            output_size=edge_size,
        )
        self.f_node = MLP(
            input_size=edge_size + node_size,
            n_hidden=n_hidden,
            layer_norm=layer_norm,
            act="GELU",
            output_size=output_size,
        )

    def forward(self, V, E, edges):
        edges = edges.astype(dtype="int64")
        senders = paddle.take_along_axis(
            arr=V,
            axis=-2,
            indices=edges[..., 0]
            .unsqueeze(axis=-1)
            .tile(repeat_times=[1, 1, tuple(V.shape)[-1]]),
            broadcast=False,
        )
        receivers = paddle.take_along_axis(
            arr=V,
            axis=-2,
            indices=edges[..., 1]
            .unsqueeze(axis=-1)
            .tile(repeat_times=[1, 1, tuple(V.shape)[-1]]),
            broadcast=False,
        )
        edge_inpt = paddle.concat(x=[senders, receivers, E], axis=-1)
        edge_embeddings = self.f_edge(edge_inpt)
        col = (
            edges[..., 1]
            .unsqueeze(axis=-1)
            .tile(repeat_times=[1, 1, tuple(edge_embeddings.shape)[-1]])
        )
        edge_sum = scatter_mean(
            edge_embeddings, col, dim=-2, dim_size=tuple(V.shape)[1]
        )
        node_inpt = paddle.concat(x=[V, edge_sum], axis=-1)
        node_embeddings = self.f_node(node_inpt)
        return node_embeddings, edge_embeddings


class Encoder(paddle.nn.Layer):
    def __init__(self, state_size=4, state_embedding_dim=128):
        super(Encoder, self).__init__()
        self.state_embedding_dim = state_embedding_dim
        self.enc_s_dim = 128
        self.num_frequencies = 36
        self.frequencies = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.randn(shape=[self.num_frequencies, 3])
        )
        self.enc_s = MLP(
            input_size=3 + 54, output_size=self.enc_s_dim, act="SiLU", layer_norm=False
        )
        self.fv = MLP(
            input_size=self.enc_s_dim, output_size=state_embedding_dim, layer_norm=False
        )

    def FourierEmbedding(self, pos, pos_start, pos_length):
        original_shape = tuple(pos.shape)
        new_pos = pos.reshape(-1, original_shape[-1])
        index = paddle.arange(start=pos_start, end=pos_start + pos_length)
        index = index.astype(dtype="float32")
        freq = 2**index * paddle.pi
        cos_feat = paddle.cos(x=freq.view(1, 1, -1) * new_pos.unsqueeze(axis=-1))
        sin_feat = paddle.sin(x=freq.view(1, 1, -1) * new_pos.unsqueeze(axis=-1))
        embedding = paddle.concat(x=[cos_feat, sin_feat], axis=-1)
        embedding = embedding.view(*original_shape[:-1], -1)
        all_embeddings = paddle.concat(x=[embedding, pos], axis=-1)
        return all_embeddings

    def Learnable_FourierEmbedding(self, pos):
        coords_proj = paddle.einsum(
            "bnc,fc->bnf", pos, self.frequencies * 2 * paddle.pi
        )
        fourier_emb = paddle.concat(
            x=[paddle.sin(x=coords_proj), paddle.cos(x=coords_proj)], axis=-1
        )
        out = paddle.concat(x=[pos, fourier_emb], axis=-1)
        return out

    def RotaryEmbedding(self, x):
        base = 10000
        width = self.enc_s_dim
        theta = 1.0 / (
            base
            ** (
                paddle.arange(start=0, end=width, step=2).astype(dtype="float32")
                / width
            )
        ).to(x.place)
        idx_theta = paddle.einsum("bsi,d->bsd", x, theta)
        cos_item = paddle.cos(x=idx_theta)
        sin_item = paddle.sin(x=idx_theta)
        embedding = paddle.concat(x=[cos_item, sin_item], axis=-1)
        return embedding

    def forward(self, node_pos):
        pos_enc = self.FourierEmbedding(node_pos, -4, 9)
        s_enc = self.enc_s(pos_enc)
        V_in = paddle.concat(x=[s_enc], axis=-1)
        V_in = self.fv(V_in)
        return V_in, s_enc


class AttentionBlock(paddle.nn.Layer):
    def __init__(self, n_token=64, w_size=128, n_heads=4):
        super(AttentionBlock, self).__init__()
        self.channel_dim = w_size
        self.n_token = n_token
        self.softmax = paddle.nn.Softmax(axis=-1)
        self.scale = self.channel_dim**-0.5
        self.n_heads = n_heads
        self.Q = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.randn(shape=[self.n_token, self.channel_dim]), trainable=True
        )
        self.to_q_1 = paddle.nn.Linear(
            in_features=self.channel_dim, out_features=self.channel_dim
        )
        self.to_k_1 = paddle.nn.Linear(
            in_features=self.channel_dim, out_features=self.channel_dim
        )
        self.to_v_1 = paddle.nn.Linear(
            in_features=self.channel_dim, out_features=self.channel_dim
        )
        self.attention2 = paddle.nn.MultiHeadAttention(
            embed_dim=w_size,
            num_heads=n_heads,
            weight_attr = paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()
                ),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()
                ),
            )

        self.to_q_2 = paddle.nn.Linear(
            in_features=self.channel_dim, out_features=self.channel_dim
        )
        self.to_k_2 = paddle.nn.Linear(
            in_features=self.channel_dim, out_features=self.channel_dim
        )
        self.to_v_2 = paddle.nn.Linear(
            in_features=self.channel_dim, out_features=self.channel_dim
        )

    def forward(self, W_0):
        B = tuple(W_0.shape)[0]
        learned_Q = self.Q.unsqueeze(axis=0).expand(shape=[B, -1, -1])
        Q_1 = self.to_q_1(learned_Q)
        K_1 = self.to_k_1(W_0)
        V_1 = self.to_v_1(W_0)
        attn1 = self.softmax(paddle.einsum("bmc, bnc -> bmn", Q_1, K_1) * self.scale)
        W_1 = paddle.matmul(x=attn1, y=V_1)
        W_2 = self.attention2(W_1, W_1, W_1)
        Q_2 = self.to_q_2(W_0)
        K_2 = self.to_k_2(W_2)
        V_2 = self.to_v_2(W_2)
        attn2 = self.softmax(paddle.einsum("bnc, bmc -> bnm", Q_2, K_2) * self.scale)
        W_3 = paddle.matmul(x=attn2, y=V_2)
        return W_3


class MixerBlock(paddle.nn.Layer):
    def __init__(
        self,
        state_embedding_dim,
        att_embedding_dim,
        n_head,
        n_token,
        enc_s_dim=0,
        idx=0,
        n_blocks=4,
    ):
        super(MixerBlock, self).__init__()
        self.enc_s_dim = enc_s_dim
        node_size = (
            state_embedding_dim if enc_s_dim == 0 else state_embedding_dim + enc_s_dim
        )
        self.gnn = GNN(
            node_size=node_size,
            edge_size=state_embedding_dim,
            output_size=state_embedding_dim,
            layer_norm=True,
        )
        self.ln1 = paddle.nn.LayerNorm(normalized_shape=att_embedding_dim)
        self.ln2 = paddle.nn.LayerNorm(normalized_shape=att_embedding_dim)
        self.linear = paddle.nn.Linear(
            in_features=att_embedding_dim, out_features=att_embedding_dim
        )
        self.MHA = AttentionBlock(
            n_token=n_token, w_size=att_embedding_dim, n_heads=n_head
        )
        self.idx = idx
        self.n_blocks = n_blocks
        if self.idx < self.n_blocks:
            self.alpha_1 = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.to_tensor(data=0.0), trainable=True
            )
            self.alpha_2 = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.to_tensor(data=0.0), trainable=True
            )
            self.alpha_3 = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.to_tensor(data=0.0), trainable=True
            )
            self.alpha_4 = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.to_tensor(data=0.0), trainable=True
            )
        else:
            self.alpha_1 = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.to_tensor(data=0.0), trainable=True
            )
            self.alpha_3 = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.to_tensor(data=0.0), trainable=True
            )
            self.alpha_4 = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.to_tensor(data=0.0), trainable=True
            )
        self.act = paddle.nn.Silu()

    def forward(self, V, E, edges, s_enc):
        if self.enc_s_dim > 0:
            V_in = paddle.concat(x=[V, s_enc], axis=-1)
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


class Mixer(paddle.nn.Layer):
    def __init__(self, N, state_embedding_dim, att_embedding_dim, n_head, n_token):
        super(Mixer, self).__init__()
        self.fe = MLP(
            input_size=3,
            output_size=state_embedding_dim,
            n_hidden=1,
            act="SiLU",
            layer_norm=False,
        )
        self.blocks = paddle.nn.LayerList(
            sublayers=[
                MixerBlock(
                    state_embedding_dim=state_embedding_dim,
                    att_embedding_dim=att_embedding_dim,
                    n_head=n_head,
                    n_token=n_token,
                    enc_s_dim=0 if i > 0 else 128,
                    idx=int(i),
                    n_blocks=N - 1,
                )
                for i in range(N)
            ]
        )

    def forward(self, V, edges, node_pos, s_enc):
        edges = edges.astype(dtype="int64")
        senders = paddle.take_along_axis(
            arr=node_pos,
            axis=-2,
            indices=edges[..., 0].unsqueeze(axis=-1).tile(repeat_times=[1, 1, 2]),
            broadcast=False,
        )
        receivers = paddle.take_along_axis(
            arr=node_pos,
            axis=-2,
            indices=edges[..., 1].unsqueeze(axis=-1).tile(repeat_times=[1, 1, 2]),
            broadcast=False,
        )
        distance = receivers - senders
        norm = paddle.sqrt(x=(distance**2).sum(-1, keepdim=True))
        E = paddle.concat(x=[distance, norm], axis=-1)
        E = self.fe(E)
        for block in self.blocks:
            V, E = block(V, E, edges, s_enc)
        return V


class Decoder(paddle.nn.Layer):
    def __init__(self, state_embedding_dim=128, state_size=4):
        super(Decoder, self).__init__()
        self.final_mlp_node = paddle.nn.Sequential(
            paddle.nn.Linear(
                in_features=state_embedding_dim, out_features=state_embedding_dim
            ),
            paddle.nn.PReLU(),
            paddle.nn.Linear(
                in_features=state_embedding_dim, out_features=state_embedding_dim
            ),
            paddle.nn.PReLU(),
            paddle.nn.Linear(in_features=state_embedding_dim, out_features=state_size),
        )

    def forward(self, V):
        V_in = paddle.concat(x=[V], axis=-1)
        final_state_node = self.final_mlp_node(V_in)
        return final_state_node


class time_stepping(paddle.nn.Layer):
    def __init__(
        self,
        N_block=3,
        state_size=3,
        state_embedding_dim=128,
        att_embedding_dim=256,
        n_head=4,
        n_token=64,
    ):
        super(time_stepping, self).__init__()
        self.encoder = Encoder(
            state_size=state_size, state_embedding_dim=state_embedding_dim
        )
        self.mixer = Mixer(
            N=N_block,
            state_embedding_dim=state_embedding_dim,
            att_embedding_dim=att_embedding_dim,
            n_head=n_head,
            n_token=n_token,
        )
        self.decoder = Decoder(
            state_embedding_dim=state_embedding_dim, state_size=state_size
        )

    def forward(self, node_pos, edges):
        V, s_enc = self.encoder(node_pos)
        V = self.mixer(V, edges, node_pos, s_enc)
        final_state_node = self.decoder(V)
        return final_state_node


class AeroGTO(paddle.nn.Layer):
    def __init__(
        self,
        N_block=4,
        state_size=4,
        state_embedding_dim=128,
        att_embedding_dim=256,
        n_head=4,
        n_token=64,
    ):
        super(AeroGTO, self).__init__()
        self.time_stepping = time_stepping(
            N_block=N_block,
            state_size=state_size,
            state_embedding_dim=state_embedding_dim,
            att_embedding_dim=att_embedding_dim,
            n_head=n_head,
            n_token=n_token,
        )
        self.alpha = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.to_tensor(data=0.0), trainable=True
        )
        self.act = paddle.nn.Silu()

    def forward(self, node_pos, edges):
        f_t = self.time_stepping(node_pos, edges)
        alpha = self.act(self.alpha)
        next_state = f_t * alpha
        return next_state


if __name__ == "__main__":
    model = AeroGTO(att_embedding_dim=128)
    dataset = Pressure_Dataset(data_path="../../data/pressure/")
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    optimizer = paddle.optimizer.Adam(parameters=model.parameters())
    att_embedding_dim = 128
    loss_fn = paddle.nn.MSELoss()
    for epoch in range(10):
        for step, (input, name) in enumerate(data_loader):
            label = input["gt"]
            node_pos = input["node_pos"]
            edges = input["edges"]
            pred = model(node_pos, edges)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                print("epoch: {}, step: {}, loss is: {}".format(epoch, step, loss))
            optimizer.clear_grad()

