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
'''
Utilities for SmoothQuant models
'''

import functools
from collections import defaultdict

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D


def apply_smoothing(scales,
                    gemm_weights,
                    layernorm_weights=None,
                    layernorm_bias=None,
                    dtype=torch.float32,
                    layernorm_1p=False):
    '''
    Applies smoothing to the weights of the model based on the given scales.

    Args:
    scales (torch.Tensor): The scales to apply to the weights.
    gemm_weights (list of torch.Tensor): The weights of the GEMM layers.
    layernorm_weights (torch.Tensor, optional): The weights of the LayerNorm layers.
                                                Defaults to None.
    layernorm_bias (torch.Tensor, optional): The bias of the LayerNorm layers.
                                              Defaults to None.
    dtype (torch.dtype, optional): The data type to use for the smoothing.
                                    Defaults to torch.float32.
    layernorm_1p (bool, optional): Whether to use 1p LayerNorm. Defaults to False.
    '''
    if not isinstance(gemm_weights, list):
        gemm_weights = [gemm_weights]

    if layernorm_weights is not None:
        assert layernorm_weights.numel() == scales.numel()
        layernorm_weights.div_(scales).to(dtype)
    if layernorm_bias is not None:
        assert layernorm_bias.numel() == scales.numel()
        layernorm_bias.div_(scales).to(dtype)
    if layernorm_1p:
        layernorm_weights += (1 / scales) - 1

    for gemm in gemm_weights:
        gemm.mul_(scales.view(1, -1)).to(dtype)


def smooth_gemm(gemm_weights,
                act_scales,
                layernorm_weights=None,
                layernorm_bias=None,
                alpha=0.5,
                weight_scales=None):
    '''
    Smooths the weights of the GEMM layers based on the activation scales.

    Args:
    gemm_weights (list of torch.Tensor): The weights of the GEMM layers.
    act_scales (torch.Tensor): The activation scales.
    layernorm_weights (torch.Tensor, optional): The weights of the LayerNorm layers.
                                                Defaults to None.
    layernorm_bias (torch.Tensor, optional): The bias of the LayerNorm layers.
                                              Defaults to None.
    alpha (float, optional): The alpha value for the smoothing. Defaults to 0.5.
    weight_scales (torch.Tensor, optional): The weight scales. Defaults to None.

    Returns:
    torch.Tensor: The smoothed scales.
    '''
    if not isinstance(gemm_weights, list):
        gemm_weights = [gemm_weights]
    orig_dtype = gemm_weights[0].dtype

    for gemm in gemm_weights:
        # gemm_weights are expected to be transposed
        assert gemm.shape[1] == act_scales.numel()

    if weight_scales is None:
        weight_scales = torch.cat(
            [gemm.abs().max(dim=0, keepdim=True)[0] for gemm in gemm_weights],
            dim=0)
        weight_scales = weight_scales.max(dim=0)[0]
    weight_scales.to(float).clamp(min=1e-5)
    scales = (act_scales.to(gemm_weights[0].device).to(float).pow(alpha) /
              weight_scales.pow(1 - alpha)).clamp(min=1e-5)

    apply_smoothing(scales, gemm_weights, layernorm_weights, layernorm_bias,
                    orig_dtype)

    return scales


def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    '''
    Smooths the weights of the fully connected layers and LayerNorm layers based
    on the activation scales.

    Args:
    ln (nn.Module): The LayerNorm layer.
