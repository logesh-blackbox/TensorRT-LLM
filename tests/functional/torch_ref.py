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

import math

import torch
from einops import rearrange


def geglu(x):
    """
    Applies Gaussian Error Linear Units (GELU) activation function element-wise.

    Args:
        x (torch.Tensor): Input tensor of any shape.

    Returns:
        torch.Tensor: Output tensor with the same shape as the input tensor.
    """
    a, b = x.chunk(2, dim=-1)
    return a * torch.nn.functional.gelu(b)


def swiglu(x):
    """
    Applies Sigmoid Weighted Linear Units (SiLU) activation function element-wise.

    Args:
        x (torch.Tensor): Input tensor of any shape.

    Returns:
        torch.Tensor: Output tensor with the same shape as the input tensor.
    """
    x, gate = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * x


def generate_qkv(x, Wqkv, nheads, kvpacked=False, qkvpacked=False):
    """
    Generates query, key, and value vectors for multi-head attention.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seqlen, nheads * d).
        Wqkv (nn.Linear): Linear layer to generate query, key, and value vectors.
        nheads (int): Number of attention heads.
        kvpacked (bool): Whether to pack key and value vectors together. Default is False.
        qkvpacked (bool): Whether to pack query, key, and value vectors together. Default is False.

    Returns:
        tuple: A tuple containing the following tensors:
            - qkv_unpad (torch.Tensor): Unpadded query, key, and value vectors.
            - cu_seqlens_q (torch.Tensor): CUDA sequence lengths for query vectors.
            - max_seqlen_q (int): Maximum sequence length for query vectors.
            - qkv (torch.Tensor): Padded query, key, and value vectors.
            - output_pad_fn (callable): Function to pad output tensor.
            - dqkv_pad_fn (callable): Function to pad differentiated query, key, and value vectors.
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen, dim = x.shape
    q, k, v = Wqkv(x).chunk(3, dim=-1)

    q_unpad = rearrange(q, 'b s (h d) -> (b s) h d', h=nheads)
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen,
                                step=seqlen,
                                dtype=torch.int32,
                                device=q_unpad.device)
    max_seqlen_q = seqlen
    output_pad_fn = lambda output_unpad: rearrange(
        output_unpad, '(b s) h d -> b s h d', b=batch_size)

    k_unpad = rearrange(k, 'b s (h d) -> (b s) h d', h=nheads)
    v_unpad = rearrange(v, 'b s (h d) -> (b s) h d', h=nheads)
    cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen,
                                step=seqlen,
                                dtype=torch.int32,
                                device=q_unpad.device)
    max_seqlen_k = seqlen

    if qkvpacked:
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = rearrange(torch.stack([q, k, v], dim=2),
                        'b s t (h d) -> b s t h d',
                        h=nheads)
        dqkv_pad_fn = lambda dqkv_unpad: rearrange(
            dqkv_unpad, '(b s) t h d -> b s t h d', b=batch_size)
        return (qkv_unpad, cu_seqlens_q, max_seqlen_q, qkv, output_pad_fn,
                dqkv_pad_fn)
    elif kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        q = rearrange(q, 'b s (h d) -> b s h d', h=nheads)
        kv = rearrange(torch.stack([k, v], dim=2),
                       'b s t (h d) -> b s t h d',
                       h=nheads)
        dq_pad_fn = output_pad_fn
        dkv_pad_fn = lambda dkv_unpad: rearrange(
            dkv_unpad, '(b s) t h d -> b s t h d', b=batch_size)
        return (q_unpad, kv_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                max_seqlen_k, q, kv, output_pad_fn, dq_pad_fn
