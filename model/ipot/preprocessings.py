from .position_encoding import *
from einops import repeat


class AbstractPreprocessor(nn.Module):
    @property
    def num_channels(self) -> int:
        """ Returns size of preprocessor output. """
        raise NotImplementedError()


class IPOTBasicPreprocessor(AbstractPreprocessor):
    """
    Preprocessing inputs for Encoder.
    """

    def __init__(
            self,
            config=None,
            prep_type="pixels",
            spatial_downsample: int = 1,
            temporal_downsample: int = 1,
            position_encoding_type: str = "pos2fourier",
            in_channel: int = 1,
            pos_channel: int = 1,
            out_channel: int = None,
            concat_or_add_pos: str = "concat",
            project_pos_channel: int = -1,
            **position_encoding_kwargs,
    ):
        super(IPOTBasicPreprocessor, self).__init__()
        self.config = config

        if concat_or_add_pos not in ["concat", "add"]:
            raise ValueError(f"Invalid value {concat_or_add_pos} for concat_or_add_pos.")

        self.in_channel = in_channel
        self.pos_channel = pos_channel
        self.out_channel = out_channel
        self.prep_type = prep_type
        self.spatial_downsample = spatial_downsample
        self.temporal_downsample = temporal_downsample
        self.position_encoding_type = position_encoding_type
        self.concat_or_add_pos = concat_or_add_pos

        # Position embeddings
        self.project_pos_channel = project_pos_channel
        if position_encoding_type != "none":
            self.position_embeddings, self.position_projection = build_position_encoding(
                position_encoding_type=position_encoding_type,
                out_channel=out_channel,
                project_pos_channel=project_pos_channel,
                **position_encoding_kwargs,
            )

    @property
    def num_channels(self) -> int:
        # position embedding
        if self.project_pos_channel > 0:
            pos_channel = self.project_pos_channel
        else:
            pos_channel = self.position_embeddings.output_size()

        if self.concat_or_add_pos == "add":
            return pos_channel

        # inputs
        if self.prep_type == "pixels":
            input_channel = self.in_channel
        else:
            raise NotImplementedError("Not supported yet.")

        return input_channel + pos_channel

    def _build_network_inputs(self, inputs: torch.Tensor, network_input_is_1d: bool = False):
        """
        Construct the final input, including position encoding.

        This method expects the inputs to always have channels as last dimension.
        """
        self.batch_size = inputs.shape[0]
        index_dims = inputs.shape[1:-1]
        indices = np.prod(index_dims)

        # Flatten input features to a 1D index dimension if necessary.
        if len(inputs.shape) > 3 and network_input_is_1d:
            inputs = torch.reshape(inputs, [self.batch_size, indices, -1])

        inputs_function = inputs[:, :, :self.in_channel]
        inputs_pos = inputs[:, :, self.in_channel:]

        # Construct the position encoding.
        if self.position_encoding_type == "trainable":
            inputs_pos_enc = self.position_embeddings(self.batch_size)
        elif self.position_encoding_type == "fourier":
            inputs_pos_enc = self.position_embeddings(index_dims)
        elif self.position_encoding_type == "pos2fourier":
            inputs_pos_enc = self.position_embeddings(pos=inputs_pos)
        else:
            raise NotImplementedError

        # Optionally project them to a target dimension.
        inputs_pos_enc = self.position_projection(inputs_pos_enc)

        return inputs_function, inputs_pos_enc

    def forward(self, inputs: torch.Tensor, network_input_is_1d: bool = True):
        # Split function values and positions, and build positional encoding
        if self.position_encoding_type != "none":
            inputs_function, inputs_pos_enc = self._build_network_inputs(inputs, network_input_is_1d)

        # Make sure inputs_pos_enc contains batch dimensions.
        if inputs_pos_enc.ndim == 2:
            inputs_pos_enc = repeat(inputs_pos_enc, 'n d -> b n d', b=self.batch_size)

        # Make sure that inputs_pos_enc and inputs_function are on the same device.
        if inputs_pos_enc.device != inputs_function.device:
            inputs_pos_enc = inputs_pos_enc.to(inputs_function.device)

        # Concat or add?
        if self.concat_or_add_pos == "concat":
            inputs_for_encoder = torch.cat([inputs_function, inputs_pos_enc], dim=-1)
        elif self.concat_or_add_pos == "add":
            inputs_for_encoder = inputs_function + inputs_pos_enc
        else:
            raise NotImplementedError

        return inputs_for_encoder
