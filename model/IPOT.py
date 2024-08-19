import torch
import torch.nn as nn
from .ipot.ipot_encoder import IPOTEncoder
from .ipot.ipot_decoder import IPOTDecoder
from .ipot.ipot_processor import IPOTProcessor
from .ipot.preprocessings import IPOTBasicPreprocessor

class EncoderProcessorDecoder(nn.Module):
    def __init__(
            self,
            input_channel = 9,
            pos_channel = 3,  # 位置维度
            num_bands = [128],
            max_resolution = [128],
            num_latents = 512,
            latent_channel = 256,
            self_per_cross_attn = 4,
            cross_heads_num = 4,
            self_heads_num = 4,
            cross_heads_channel = 256,
            self_heads_channel = None,
            ff_mult = 2,
            latent_init_scale = 0.02,
            output_scale = 0.1,
            output_channel = 1,
            position_encoding_type = "pos2fourier"
    ):
        super(EncoderProcessorDecoder, self).__init__()
        
        # Preprocessor - positional encoding / flatten
        ipot_input_preprocessor = IPOTBasicPreprocessor(
            position_encoding_type=position_encoding_type,
            in_channel=input_channel,
            pos_channel=pos_channel,
            pos2fourier_position_encoding_kwargs=dict(
                num_bands=num_bands,
                max_resolution=max_resolution,
            )
        )
        # Encoder
        ipot_encoder = IPOTEncoder(
            input_channel=input_channel + (6 * sum(num_bands)) + pos_channel,  # pos2fourier
            num_latents=num_latents,
            latent_channel=latent_channel,
            cross_heads_num=cross_heads_num,
            cross_heads_channel=cross_heads_channel,
            latent_init_scale=latent_init_scale
        )
        # Processor
        ipot_processor = IPOTProcessor(
            self_per_cross_attn=self_per_cross_attn,
            self_heads_channel=self_heads_channel,
            latent_channel=latent_channel,
            self_heads_num=self_heads_num,
            ff_mult=ff_mult,
        )
        # Decoder
        ipot_decoder = IPOTDecoder(
            output_channel=output_channel,
            query_channel=(6 * sum(num_bands)) + pos_channel,  # pos2fourier
            latent_channel=latent_channel,
            cross_heads_num=cross_heads_num,
            cross_heads_channel=cross_heads_channel,
            ff_mult=ff_mult,
            output_scale=output_scale,
            position_encoding_type=position_encoding_type,
            pos2fourier_position_encoding_kwargs=dict(
                num_bands=num_bands,
                max_resolution=max_resolution, )
        )
        
        self.encoder = ipot_encoder
        self.processor = ipot_processor
        self.decoder = ipot_decoder
        self.input_preprocessor = ipot_input_preprocessor
        
    def forward(self, node_pos, areas, info):
        N = node_pos.shape[1]
        info = info.expand(-1, N, -1)
        inputs = torch.cat((areas, info, node_pos), dim=-1)
        
        # Operating input_preprocessor.
        if self.input_preprocessor:
            inputs_for_encoder = self.input_preprocessor(inputs, network_input_is_1d=False)
        else:
            inputs_for_encoder = inputs
        # Operating encoder.
        z = self.encoder(inputs_for_encoder)

        # Operator processor.
        z = self.processor(z)

        # Operating decoder.
        f_t = self.decoder(node_pos,z)

        return f_t
        
