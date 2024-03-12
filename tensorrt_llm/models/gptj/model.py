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
from collections import OrderedDict

import tensorrt as trt

from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt
from ...functional import (PositionEmbeddingType, Tensor, assertion,
                           gather_last_token_logits, shape)
from ...layers import (MLP, Attention, AttentionMaskType, AttentionParams,
                       ColumnLinear, Embedding, KeyValueCacheParams, LayerNorm)
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...quantization import QuantMode


class GPTJDecoderLayer(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings,
                 rotary_dim,
                 dtype=None,
                 hidden_act='relu',
                 tp_group=None,
                 tp_size=1,
                 quant_mode=QuantMode(0)):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rotary_dim = rotary_dim
        self.dtype = dtype
        self.hidden_act = hidden_act
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.quant_mode = quant_mode
        self.input_layernorm = LayerNorm(normalized_shape=hidden_size,
                                         dtype=dtype)

        self.attention = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            rotary_embedding_percentage=rotary_dim /
            (hidden_size // num_attention_heads),
            position_embedding_type=PositionEmbeddingType.rope_gptj,
            max_position_embeddings=max_position_embeddings,
            dtype=dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
            use_int8_kv_cache=quant_mode.has_int8_kv_cache(),
            quant_mode=quant_mode)

        self.mlp = MLP(hidden_size=hidden_size,
                       ffn_hidden_size=hidden_size * 4,
                       hidden_act=hidden_act,
                       dtype=dtype,
                       tp_group=tp_group,
                       tp_size=tp_size,
                       quant_mode=quant_mode)

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None):
        if not default_net(
        ).plugin_config.layernorm_plugin and trt.__version__[:3] == '8.6':
            raise AssertionError(
                "You need to enable the LayerNorm plugin for GPT-J with TensorRT 8.6. Please set plugin_config.layernorm_plugin"
            )
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attention_output = self.attention(hidden_states,
                                          attention_mask=attention_mask,
                                          use_cache=use_cache,
                                          kv_cache_params=kv_cache_params,
                                          attention_params=attention_params)

        if use_cache:
            attention_output, presents = attention_output
        attention_output = attention_output

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attention_output + feed_forward_hidden_states + residual
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class GPTJModel(Module):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 rotary_dim,
                 dtype=None,
                 mapping=Mapping(),
                 quant_mode=QuantMode(0)):
        super().__init__()
        self.embedding = Embedding(vocab_size, hidden_size, dtype=dtype)

        self.layers = ModuleList([
            GPTJDecoderLayer(hidden_size=hidden_size,
                             num_attention_heads=num_heads,
                             max_position_embeddings=max_position_embeddings,
                             rotary_dim=rotary_dim,
                             dtype=dtype,
                             hidden_act=hidden_act,
                             tp_group=mapping.tp_group,
                             tp_size=mapping.tp_size,
                             quant_mode=quant_mode) for _ in range(num_layers)
        ])

        self.ln_f = LayerNorm(normalized_shape=hidden_size, dtype=dtype)

    def forward(self,
                input_ids: Tensor,
               
