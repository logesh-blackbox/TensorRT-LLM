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

import copy
import functools
from collections import defaultdict

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D


@torch.no_grad()
def apply_smoothing(scales,
                    gemm_weights,
                    layernorm_weights=None,
                    layernorm_bias=None,
                    dtype=torch.float32,
                    layernorm_1p=False):
    """
    Applies smoothing to the given scales, gemm_weights, layernorm_weights, and
    layernorm_bias using the specified dtype and layernorm_1p flag.

    Args:
    scales (torch.Tensor): The scales tensor to be applied to gemm_weights.
    gemm_weights (list or torch.Tensor): A list of gemm weights or a single
        gemm weight tensor.
    layernorm_weights (torch.Tensor, optional): The layernorm weights tensor.
        Defaults to None.
    layernorm_bias (torch.Tensor, optional): The layernorm bias tensor.
        Defaults to None.
    dtype (torch.dtype, optional): The data type to be used for smoothing
        operations. Defaults to torch.float32.
    layernorm_1p (bool, optional): A flag indicating whether to apply 1p
        normalization. Defaults to False.

    Returns:
    None
    """
    if not isinstance(gemm_weights, list):
        gemm_weights = [gemm_weights]

    # Scale the gemm_weights and convert them to the specified dtype
    for gemm in gemm_weights:
        gemm.mul_(scales.view(1, -1)).to(dtype)

    # Scale the layernorm_weights and layernorm_bias if provided
    if layernorm_weights is not None:
        assert layernorm_weights.numel() == scales.numel()
        layernorm_weights.div_(scales).to(dtype)
    if layernorm_bias is not None:
        assert layernorm_bias.numel() == scales.numel()
        layernorm_bias.div_(scales).to(dtype)

    # Apply 1p normalization if specified
    if layernorm_1p:
        layernorm_weights += (1 / scales) - 1


@torch.no_grad()
def smooth_gemm(gemm_weights,
                act_scales,
                layernorm_weights=None,
                layernorm_bias=None,
                alpha=0.5,
                weight_scales=None):
    """
    Smooths the given gemm_weights using the specified act_scales, alpha, and
    weight_scales. Also scales the layernorm_weights and layernorm_bias if
    provided.

    Args:
    gemm_weights (list or torch.Tensor): A list of gemm weights or a single
        gemm weight tensor.
    act_scales (torch.Tensor): The activation scales tensor.
    layernorm_weights (torch.Tensor, optional): The layernorm weights tensor.
        Defaults to None.
    layernorm_bias (torch.Tensor, optional): The layernorm bias tensor.
        Defaults to None.
    alpha (float, optional): The alpha value used for smoothing. Defaults to 0.5.
    weight_scales (torch.Tensor, optional): The weight scales tensor.
        Defaults to None.

    Returns:
    torch.Tensor: The smoothed scales tensor.
    """
    if not isinstance(gemm_weights, list):
        gemm_weights = [gemm_weights]
    orig_dtype = gemm_weights[0].dtype

    # Calculate the scales tensor using the act_scales, alpha, and weight_scales
    scales = (act_scales.to(gemm_weights[0].device).to(float).pow(alpha) /
              weight_scales.pow(1 - alpha)).clamp(min=1e-5)

    # Apply smoothing to the gemm_weights, layernorm_weights, and layernorm_bias
    apply_smoothing(scales, gemm_weights, layernorm_weights, layernorm_bias,
                    orig_dtype)

    return scales


@torch.no_grad()
def smooth_gemm_fc1_gate(fc1_weights,
                         gate_weights,
                         act_scales,
                         layernorm_weights=None,
                         layernorm_bias=None,
                         alpha=0.5,
                         weight_scales=None):
    """
    Smooths the given fc1_weights and gate_weights using the specified act_scales,
    alpha, and weight_scales. Also scales the layernorm_weights and layernorm_bias
    if provided.

    Args:
    fc1_weights (list or torch.Tensor): A list of fc1 weights or a single fc1
        weight tensor.
    gate_weights (list or torch.Tensor): A list of gate weights or a single gate
        weight tensor.
    act_scales (torch.Tensor): The activation scales tensor.
