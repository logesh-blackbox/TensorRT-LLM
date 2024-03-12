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

from typing import Tuple

import torch
import torch.nn as nn


class Conv2d(nn.Module):
    r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    Args:
        in_channels (int): Number of input image channels.
        out_channels (int): Number of output image channels.
        kernel_size (Tuple[int, int]): Size of the convolving kernel.
        stride (Tuple[int, int], optional): Stride of the convolution. Default: (1, 1)
        padding (Tuple[int, int], optional): Zero-padding added to both sides of the input. Default: (0, 0)
        dilation (Tuple[int, int], optional): Spacing between kernel elements. Default: (1, 1)
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
        padding_mode (str, optional): Type of padding. Default: 'zeros'

    Returns:
        torch.Tensor: The convolved tensor.

    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int, int],
            stride: Tuple[int, int] = (1, 1),
            padding: Tuple[int, int] = (0, 0),
            dilation: Tuple[int, int] = (1, 1),
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            dtype=None) -> None:
        super().__init__()
        self._validate_groups(in_channels, out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.weight = torch.nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *kernel_size,
                        dtype=dtype))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    def _validate_groups(self, in_channels, out_channels, groups):
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

    def forward(self, input):
        return torch.nn.functional.conv2d(
            input, self.weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups)


class ConvTranspose2d(nn.Module):
    r"""Applies a 2D transposed convolution over an input signal composed of several input
    planes.

    Args:
        in_channels (int): Number of input image channels.
        out_channels (int): Number of output image channels.
        kernel_size (Tuple[int, int]): Size of the convolving kernel.
        stride (Tuple[int, int], optional): Stride of the convolution. Default: (1, 1)
        padding (Tuple[int, int], optional): Zero-padding added to both sides of the input. Default: (0, 0)
        output_padding (Tuple[int, int], optional): Additional size added to one side of the output shape. Default: (0, 0)
        dilation (Tuple[int, int], optional): Spacing between kernel elements. Default: (1, 1)
        groups (int, optional): Number of
