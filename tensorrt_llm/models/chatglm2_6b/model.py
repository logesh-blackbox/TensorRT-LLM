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
import numpy as np
import tensorrt as trt
import torch

from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt
from ...functional import (PositionEmbeddingType, Tensor, concat, constant,
                           expand, expand_dims, gather_last_token_logits,
                           gpt_attention, index_select, select, shape, slice,
                           split)
from ...layers import (MLP, AttentionMaskType, AttentionParams, Attention,
                       ColumnLinear, Embedding, KeyValueCacheParams,
                       RmsNorm, RowLinear)
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...parameter import Parameter
from ...quantization import QuantMode
from ..generation_mixin import GenerationMixin


def apply_rotary_pos_emb_trt(x: Tensor, rope_cache: Tensor) -> Tensor:
    # x-> [seq, batch, num_heads, 2]
    x = x.permute((1, 0, 2, 3))
    # sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    sq = shape(x, 0)
    b = shape(x, 1)
    nh = shape(x, 2)
    shape(x, 3)
    # rope_cache shape: seq,batch,heads,2 rot_dim = 2* numheads
    #rope_cache: seq,batch,num_states/4,2
    rot_dim = shape(rope_cache, 2) * constant(np.array(2, dtype=np.int32))
    starts = concat([0, 0, 0, 0])
    sizes = concat([sq, b, nh, rot_dim])
    # first half
    x_rot = slice(x, starts, sizes)
    starts = concat([0, 0, 0, rot_dim])
    # second half
    x_pass = slice(x, starts, sizes)
    # truncate to support variable sizes
    rope_cache = slice(rope_cache, (0, 0, 0, 0), (concat(
        [sq,
         shape(rope_cache, 1),
         shape(rope_cache, 2),
         shape(rope_cache, 3)])))
    xshaped = x_rot.view(concat([sq, b, nh, rot_dim / 2, 2]))
    rope_cache = rope_cache.view(concat([sq, b, 1, shape(xshaped, 3), 2]))
    # first half
    xshape0 = select(xshaped, 4, 0)
    # second half
    xshape1 = select(xshaped, 4, 1)
    # first half
    rope_cache0 = select(rope_cache, 4, 0)
    # second half
    rope_cache1 = select(rope_cache, 4, 1)
    out0 = xshape0 * rope_cache0 - xshape1 * rope_cache1
    out1 = xshape1 * rope_cache0 + xshape0 * rope_cache1
    out0 = expand_dims(out0, 4)
    out1 = expand_dims(out1, 4)
    x_out2_v1 = concat([out0, out1], 4)
    x_out2 = x_out2_v1.view(
        concat([sq, b, nh, shape(x_out2_v1, 3) * shape(x_out2_v1, 4)]))
    output = concat([x_out2, x_pass], dim=3)
    # to batch,seq,num_group,head_states
    output = output.permute((1, 0, 2, 3))
    return output


class RotaryEmbedding(Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, seq_len: int):
        theta = 1.0 / (10000**(torch.arange(0, self.dim, 2) / self.dim))
        seq_idx = torch.arange(seq_len)
        idx_theta = torch.outer(seq_idx, theta).float()
        cache = torch.stack(
            [torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        cache = cache.half()
        # create rope embeddings and make it constant
        cache = constant(cache.numpy())
        return cache


class ChatGLM2Attention(Module):

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        layer_number,
        kv_channels=128,
        multi_query_group_num=2,
        apply_query_key_layer_scaling=False,
        attention_mask_type=AttentionMaskType.causal,
        qkv_bias=True,
        linear_bias=False,
        dtype='float16',
        use_int8_kv_cache=False,
        tp_group=None,
        tp_size=1,
    ):
        super().__
