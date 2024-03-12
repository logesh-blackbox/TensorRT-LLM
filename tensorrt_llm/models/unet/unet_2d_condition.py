# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple

from ...functional import silu
from ...layers import Conv2d, GroupNorm
from ...module import Module, ModuleList
from .embeddings import TimestepEmbedding, Timesteps
from .unet_2d_blocks import (UNetMidBlock2DCrossAttn, get_down_block,
                             get_up_block)


class UNet2DConditionModel(Module):
    """
    A U-Net model with conditioning on timesteps.

    Args:
        sample_size (int, optional): The size of the input sample. Defaults to None.
        in_channels (int, optional): The number of input channels. Defaults to 4.
        out_channels (int, optional): The number of output channels. Defaults to 4.
        center_input_sample (bool, optional): Whether to center the input sample. Defaults to False.
        flip_sin_to_cos (bool, optional): Whether to flip sin to cos in the time embedding. Defaults to True.
        freq_shift (int, optional): The frequency shift for the time embedding. Defaults to 0.
        down_block_types (Tuple[str], optional): The types of down-sampling blocks. Defaults to ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D").
        up_block_types (Tuple[str], optional): The types of up-sampling blocks. Defaults to ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D").
        block_out_channels (Tuple[int], optional): The number of output channels for each block. Defaults to (320, 640, 1280, 1280).
        layers_per_block (int, optional): The number of layers per block. Defaults to 2.
        downsample_padding (int, optional): The padding for down-sampling. Defaults to 1.
        mid_block_scale_factor (float, optional): The scale factor for the mid block. Defaults to 1.0.
        act_fn (str, optional): The activation function. Defaults to "silu".
        norm_num_groups (int, optional): The number of groups for normalization. Defaults to 32.
        norm_eps (float, optional): The epsilon value for normalization. Defaults to 1e-5.
        cross_attention_dim (int, optional): The dimension for cross-attention. Defaults to 1280.
        attention_head_dim (int, optional): The dimension for attention heads. Defaults to 8.
    """

    def __init__(
        self,
        sample_size=None,
        in_channels=4,
        out_channels=4,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                          "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
                        "CrossAttnUpBlock2D"),
        block_out_channels=(320, 640, 1280, 1280),
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1.0,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-5,
        cross_attention_dim=1280,
        attention_head_dim=8,
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # Input convolution
        self.conv_in = Conv2d(in_channels,
                              block_out_channels[0],
                              kernel_size=(3, 3),
                              padding=(1, 1))

        # Time embedding
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos,
                                   freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim,
                                                time_embed_dim)

        down_blocks = []
        up_blocks = []

        # Down-sampling blocks
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
               
