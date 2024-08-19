import torch
import torch.nn as nn
import abc
import numpy as np


def build_position_encoding(
        position_encoding_type,
        out_channel=None,
        project_pos_channel=-1,
        trainable_position_encoding_kwargs=None,
        fourier_position_encoding_kwargs=None,
        pos2fourier_position_encoding_kwargs=None,
):
    """
    Builds the position encodings.

    Args:
        - output_channel: refers to the number of channels of the position encodings.
        - project_pos_channel: if specified, will project the position encodings to this dimensions.
    """

    if position_encoding_type == "trainable":
        if not trainable_position_encoding_kwargs:
            raise ValueError("Make sure to pass trainable_position_encoding_kwargs")
        output_pos_enc = TrainablePositionEncoding(**trainable_position_encoding_kwargs)

    elif position_encoding_type == "fourier":
        # We don't use the index_dims arguent, as this is only known during the forward pass
        if not fourier_position_encoding_kwargs:
            raise ValueError("Make sure to pass fourier_position_encoding_kwargs")
        output_pos_enc = FourierPositionEncoding(**fourier_position_encoding_kwargs)

    elif position_encoding_type == "pos2fourier":
        if not pos2fourier_position_encoding_kwargs:
            raise ValueError("Make sure to pass pos2fourier_position_encoding_kwargs")
        output_pos_enc = FourierPositionEncoding(**pos2fourier_position_encoding_kwargs)

    else:
        raise ValueError(f"Unknown position encoding type: {position_encoding_type}")

    # Optionally, project the position encoding to a target dimension:
    position_projection = nn.Linear(out_channel, project_pos_channel) if project_pos_channel > 0 else nn.Identity()

    return output_pos_enc, position_projection


class AbstractPositionEncoding(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract positional encoding.
    """

    @property
    @abc.abstractclassmethod
    def num_dimensions(self) -> int:
        raise NotImplementedError

    @abc.abstractclassmethod
    def output_size(self, *args, **kwargs) -> int:
        raise NotImplementedError

    @abc.abstractclassmethod
    def forward(self, pos):
        raise NotImplementedError


class TrainablePositionEncoding(AbstractPositionEncoding):
    """ Trainable position encoding. """

    def __init__(self, index_dims, num_channel=128):
        super(TrainablePositionEncoding, self).__init__()
        self._num_channel = num_channel
        self._index_dims = index_dims
        index_dim = np.prod(index_dims)
        self.position_embeddings = nn.Parameter(torch.randn(index_dim, num_channel))

    @property
    def num_dimensions(self) -> int:
        if isinstance(self._index_dims, int):
            return 1
        return len(self._index_dims)

    def output_size(self, *args, **kwargs) -> int:
        return self._num_channel

    def forward(self, pos=None):
        position_embeddings = self.position_embeddings
        return position_embeddings


class FourierPositionEncoding(AbstractPositionEncoding):
    """ Fourier (Sinusoidal) position encoding. """

    def __init__(self, num_bands, max_resolution, concat_pos=True, sine_only=False):
        super(FourierPositionEncoding, self).__init__()
        self.num_bands = num_bands
        self.max_resolution = max_resolution
        self.concat_pos = concat_pos
        self.sine_only = sine_only

    @property
    def num_dimensions(self) -> int:
        return len(self.max_resolution)

    def output_size(self):
        """ Returns size of positional encodings last dimension. """
        encoding_size = sum(self.num_bands)
        if not self.sine_only:
            encoding_size *= 2
        if self.concat_pos:
            encoding_size += self.num_dimensions

        return encoding_size

    def forward(self, pos=None):
        fourier_pos_enc = generate_fourier_features(
            pos,
            num_bands=self.num_bands,
            max_resolution=self.max_resolution,
            concat_pos=self.concat_pos,
            sine_only=self.sine_only)
        return fourier_pos_enc


def generate_fourier_features(pos, num_bands, max_resolution=(2 ** 10), concat_pos=True, sine_only=False):
    """
    Generate a Fourier feature position encoding with linear spacing.

    Args:
        pos: The Tensor containing the position of n points in d dimensional space.
        num_bands: The number of frequency bands (K) to use.
        max_resolution: The maximum resolution (i.e., the number of pixels per dim). A tuple representing resoltuion for each dimension.
        concat_pos: Whether to concatenate the input position encoding to the Fourier features.
        sine_only: Whether to use a single phase (sin) or two (sin/cos) for each frequency band.
    """
    if sum(num_bands) == 0:
        return pos

    if len(pos.shape) > 2:
        batch_size = pos.shape[0]
    else:
        batch_size = None

    min_freq = 1.0

    # Nyquist frequency at the target resolution:
    freq_bands = torch.stack(
        [torch.linspace(
            start=min_freq, end=res / 2, steps=num_band) for res, num_band in zip(max_resolution, num_bands)],
        dim=0).to(pos.device)

    # Get frequency bands for each spatial dimension.
    # Output is size [n, d * num_bands]
    if batch_size is not None:
        per_pos_features = pos[:, :, :, None] * freq_bands[None, :, :]  # This is for elasticity
        per_pos_features = torch.reshape(per_pos_features, [batch_size, -1, np.prod(per_pos_features.shape[2:])])
    else:
        per_pos_features = pos[:, :, None] * freq_bands[None, :, :]
        per_pos_features = torch.reshape(per_pos_features, [-1, np.prod(per_pos_features.shape[1:])])
    if sine_only:
        # Output is size [n, d * num_bands]
        per_pos_features = torch.sin(np.pi * per_pos_features)
    else:
        # Output is size [n, 2 * d * num_bands]
        per_pos_features = torch.cat(
            [torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)], dim=-1)

    # Concatenate the raw input positions.
    if concat_pos:
        # Adds d bands to the encoding.
        per_pos_features = torch.cat([pos, per_pos_features], dim=-1)
    return per_pos_features
