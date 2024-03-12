'''
Utilities for SmoothQuant models
'''

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
    Applies smoothing to the given weights using the provided scales.
    :param scales: The scales to apply to the weights.
    :param gemm_weights: The weights to apply smoothing to.
    :param layernorm_weights: The layernorm weights to apply smoothing to.
    :param layernorm_bias: The layernorm bias to apply smoothing to.
    :param dtype: The data type to use for the smoothed weights.
    :param layernorm_1p: Whether to use 1p layernorm smoothing.
    """
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


@torch.no_grad()
def smooth_gemm(gemm_weights,
                act_scales,
                layernorm_weights=None,
                layernorm_bias=None,
                alpha=0.5,
                weight_scales=None):
    """
    Smooths the given gemm weights using the provided act_scales and alpha value.
    :param gemm_weights: The weights to smooth.
    :param act_scales: The activation scales to use for smoothing.
    :param layernorm_weights: The layernorm weights to smooth.
    :param layernorm_bias: The layernorm bias to smooth.
    :param alpha: The alpha value to use for smoothing.
    :param weight_scales: The weight scales to use for smoothing.
    :return: The smoothed scales.
    """
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


@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    """
    Smooths the given layernorm and fc weights using the provided act_scales and alpha value.
    :param ln: The layernorm to smooth.
    :param fcs: The fcs to smooth.
    :param act_scales: The activation scales to use for smoothing.
    :param alpha: The alpha value to use for smoothing.
    :return: The smoothed scales.
    """
    if not isinstance(fcs, list):
        fcs = [fcs]
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (act_scales.pow(alpha) /
              weight_scales.pow(1 - alpha)).clamp(min=1e-5).to(device).to(dtype)

    if ln is not None:
        ln.weight.div_(scales)
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
    return scales


@torch.no_grad()
def capture_activation_range(model,
                             tokenizer,
                             dataset,
                             num_samples=512,
                             seq_len=512):
    """
    Captures the activation range of the given model using the provided tokenizer, dataset, and number of samples.
    :param model: The model to capture the activation range of.
    :param tokenizer: The tokenizer to use for processing the dataset.
    :
