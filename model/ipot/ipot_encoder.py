from .layers import *
from .position_encoding import *


class IPOTEncoder(nn.Module):
    def __init__(
            self,
            *,
            input_channel=None,
            cross_heads_channel=None,  # default: latent_channel // cross_heads_num
            num_latents=2 ** 10,
            latent_channel=2 ** 6,
            latent_init_scale=0.02,
            cross_heads_num=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            weight_tie_layers=False,
            use_query_residual=True):
        super(IPOTEncoder, self).__init__()

        self.input_channel = input_channel

        if cross_heads_channel is None:
            cross_heads_channel = int(latent_channel // cross_heads_num)

        self.use_query_residual = use_query_residual

        self.latents = nn.Parameter(torch.randn(num_latents, latent_channel) * latent_init_scale)

        self.encoder_cross_attn = PreNorm(
            latent_channel,
            Attention(
                latent_channel,
                input_channel,
                heads_num=cross_heads_num,
                heads_channel=cross_heads_channel,
                dropout=attn_dropout
            ),
            context_channel=input_channel
        )
        self.encoder_ff = PreNorm(
            latent_channel,
            FeedForward(
                latent_channel,
                mult=ff_mult,
                dropout=ff_dropout
            )
        )

    def forward(self, inputs, mask=None, return_embedding=False):
        b, *axis = inputs.shape

        # concat to channels of data and flatten axis
        inputs = rearrange(inputs, 'b ... d -> b (...) d')
        z = repeat(self.latents, 'n d -> b n d', b=b)

        if self.use_query_residual:
            z = self.encoder_cross_attn(z, context=inputs) + z
        else:
            z = self.encoder_cross_attn(z, context=inputs)

        return z
