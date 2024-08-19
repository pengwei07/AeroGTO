from .layers import *
from .position_encoding import *


class IPOTDecoder(nn.Module):
    def __init__(
            self,
            output_channel,
            query_channel,
            latent_channel=2 ** 8,
            output_scale=0.1,
            cross_heads_num=8,
            cross_heads_channel=None,  # default: latent_channel // cross_heads_num
            ff_mult=4,
            position_encoding_type: str = "pos2fourier",
            use_qeury_residual=False,
            concat_preprocessed_input=False,
            project_pos_channel: int = -1,
            position_encoding_only=False,
            **position_encoding_kwargs):
        super(IPOTDecoder, self).__init__()
        if cross_heads_channel is None:
            cross_heads_channel = int(latent_channel // cross_heads_num)
        self.query_channel = query_channel
        self.output_channel = output_channel
        self.output_scale = output_scale
        self.position_encoding_type = position_encoding_type
        self.use_query_residual = use_qeury_residual
        self.concat_preprocessed_input = concat_preprocessed_input
        self.position_encoding_kwargs = position_encoding_kwargs

        # If position_embedding_type is 'None', the decoder will not construct
        # any position embeddings. In that casse, you should construct your own decoder_query.

        # Position embeddings
        self.project_pos_dim = project_pos_channel
        if position_encoding_type != "none":
            self.position_embeddings, self.position_projection = build_position_encoding(
                position_encoding_type=position_encoding_type,
                out_channel=query_channel,
                project_pos_channel=project_pos_channel,
                **position_encoding_kwargs,
            )

        self.decoder_cross_attn = PreNorm(
            query_channel,
            Attention(
                query_channel,
                latent_channel,
                output_channel,
                heads_num=cross_heads_num,
                heads_channel=cross_heads_channel
            ),
            context_channel=latent_channel
        )

        self.decoder_ff = PreNorm(
            output_channel,
            FeedForward(
                output_channel,
                mult=ff_mult,
            )
        )

    @property
    def num_channels(self) -> int:
        # position embedding
        if self.project_pos_dim > 0:
            pos_channel = self.project_pos_dim
        else:
            pos_channel = self.position_embeddings.output_size()
        return pos_channel

    def _build_decoder_query(self, pos_query, network_input_is_1d: bool = False):
        """
        Construct the final input, including position encoding.

        This method expects the inputs to always have channels as last dimension.
        """
        index_dims = pos_query.shape[:-1]
        indices = np.prod(index_dims)

        # Flatten input features to a 1D index dimension if necessary.
        if len(pos_query.shape) > 2 and network_input_is_1d:
            pos_query = torch.reshape(pos_query, [indices, -1])

        if self.position_encoding_type == "trainable":
            pos_enc_query = self.position_embeddings()
        elif self.position_encoding_type == "fourier":
            pos_enc_query = self.position_embeddings(index_dims)
        elif self.position_encoding_type == "pos2fourier":
            pos_enc_query = self.position_embeddings(pos=pos_query)
        else:
            pos_enc_query = pos_query

        # Optionally project them to a target dimension.
        pos_enc_query = self.position_projection(pos_enc_query)

        return pos_enc_query, pos_query

    def forward(self, dec_query, z):
        b, *axis = z.shape

        if not exists(dec_query):
            return z

        # Build decoder query with positional embeddings
        if self.position_encoding_type != "none":
            dec_query, _ = self._build_decoder_query(dec_query)

        # Make sure query contains batch dimension
        if dec_query.ndim == 2:
            dec_query = repeat(dec_query, 'n d -> b n d', b=b)

        if dec_query.ndim > 3:
            dec_query = dec_query.reshape(dec_query.shape[0], -1, dec_query.shape[-1])

        # Make sure that queries and latents are on the same device.
        if dec_query.device != z.device:
            dec_query = dec_query.to(z.device)

        # Cross attend from decoder queries to latents.
        if self.use_query_residual:
            out = self.decoder_cross_attn(dec_query, context=z) + dec_query
        else:
            out = self.decoder_cross_attn(dec_query, context=z)

        # Optional decoder feedforward
        if exists(self.decoder_ff):
            out = self.decoder_ff(out) + out

        return out * self.output_scale