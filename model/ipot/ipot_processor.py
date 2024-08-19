from .layers import *
from .position_encoding import *


class IPOTProcessor(nn.Module):
    def __init__(
            self,
            *,
            self_per_cross_attn=6,
            self_heads_channel=None,  # default: latent_channel // self_heads_num
            latent_channel=2 ** 6,
            self_heads_num=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            weight_tie_layers=False,
            use_query_residual=True):
        super(IPOTProcessor, self).__init__()

        if self_heads_channel is None:
            self_heads_channel = int(latent_channel // self_heads_num)

        self.use_query_residual = use_query_residual

        self.processor_self_attn = PreNorm(
            latent_channel,
            Attention(
                latent_channel,
                heads_num=self_heads_num,
                heads_channel=self_heads_channel,
                dropout=attn_dropout
            )
        )
        self.processor_ff = PreNorm(
            latent_channel,
            FeedForward(
                latent_channel,
                mult=ff_mult,
                dropout=ff_dropout
            )
        )

        self.layers = nn.ModuleList([])

        for i in range(self_per_cross_attn):
            self.layers.append(nn.ModuleList([
                self.processor_self_attn,
                self.processor_ff
            ]))

    def forward(self, z, mask=None, return_embedding=False):
        b, *axis = z.shape

        # Processing layers
        for self_attn, self_ff in self.layers:
            if self.use_query_residual:
                z = self_attn(z, context=z) + z
            else:
                z = self_attn(z, context=z)
            z = self_ff(z) + z
        return z

