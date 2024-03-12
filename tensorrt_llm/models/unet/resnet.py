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
from functools import partial

from ...functional import avg_pool2d, interpolate, silu, view
from ...layers import (AvgPool2d, Conv2d, ConvTranspose2d, GroupNorm, Linear,
                       Mish)
from ...module import Module


class Upsample2D(Module):
    """
    Upsample 2D feature maps using either nearest-neighbor interpolation or a
    transposed convolution.

    Args:
        channels (int): The number of channels in the input feature maps.
        use_conv (bool): Whether to use a transposed convolution for upsampling.
            If True, the `out_channels` argument must be provided.
        use_conv_transpose (bool): Deprecated. Use `use_conv` instead.
        out_channels (int, optional): The number of channels in the output
            feature maps. Required if `use_conv` is True.
    """

    def __init__(self,
                 channels: int,
                 use_conv=False,
                 use_conv_transpose=False,
                 out_channels=None) -> None:
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels

        self.use_conv_transpose = use_conv_transpose
        self.use_conv = use_conv
        if self.use_conv_transpose:
            self.conv = ConvTranspose2d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = Conv2d(self.channels,
                               self.out_channels, (3, 3),
                               padding=(1, 1))
        else:
            self.conv = None

    def forward(self, hidden_states, output_size=None):
        """
        Applies the upsampling operation to the input feature maps.

        Args:
            hidden_states (torch.Tensor): The input feature maps with shape
                (batch_size, channels, height, width).
            output_size (Tuple[int, int], optional): The desired output size of
                the upsampled feature maps. If not provided, nearest-neighbor
                interpolation with a scale factor of 2 is used.

        Returns:
            torch.Tensor: The upsampled feature maps with shape
            (batch_size, out_channels, new_height, new_width).
        """
        assert not hidden_states.is_dynamic()
        batch, channels, _, _ = hidden_states.size()
        assert channels == self.channels

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        if output_size is None:
            hidden_states = interpolate(hidden_states,
                                        scale_factor=2.0,
                                        mode="nearest")
        else:
            hidden_states = interpolate(hidden_states,
                                        size=output_size,
                                        mode="nearest")

        if self.use_conv:
            hidden_states = self.conv(hidden_states)

        return hidden_states


class Downsample2D(Module):
    """
    Downsample 2D feature maps using either average pooling or a convolution.

    Args:
        channels (int): The number of channels in the input feature maps.
        use_conv (bool): Whether to use a convolution for downsampling. If True,
            the `out_channels` argument must be provided.
        out_channels (int, optional): The number of channels in the output
            feature maps. Required if `use_conv` is True.
        padding (int, optional): The amount of padding to apply to the input
            feature maps before downsampling. Default: 1.
    """

    def __init__(self,
                 channels,
                 use_conv=False,
                 out_channels=None,
                 padding=1) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = (2, 2)

        if use_conv:
            self.conv = Conv2d(self.channels,
                               self.out_channels, (3, 3),
                               stride=stride,
                               padding=(padding, padding))
        else:
            assert self.channels == self.out_channels
            self.conv = AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, hidden_states):
        """
        Applies the downsampling operation to the input feature maps.

        Args:
            hidden_states (torch.Tensor): The input feature maps with shape
                (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The downsampled feature maps with shape
            (batch_size, out_channels, new_height, new_width).
        """
        assert not hidden_states.is_dynamic()
        batch, channels, _, _ = hidden_states.size()
        assert channels == self.channels

        #TODO add the missing pad function
        hidden_states = self.conv(hidden_states)

        return hidden_states


class ResnetBlock2D(Module):
    """
    A residual block for 2D feature maps with pre
