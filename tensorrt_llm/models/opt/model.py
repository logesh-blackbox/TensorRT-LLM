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
from collections import OrderedDict

import tensorrt as trt

from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt
from ...functional import Tensor, gather_last_token_logits
from ...layers import (MLP, Attention, AttentionMaskType, AttentionParams,
                       ColumnLinear, KeyValueCacheParams, LayerNorm,
                       PositionEmbeddingType)
from ...mapping import Mapping
from ...module import Module, ModuleList
from ..generation_mixin import GenerationMixin
from ..gpt.model import GPTEmbedding


class OPTDecoderLayer(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings,
                 dtype=None,
                 hidden_act='relu',
                 pre_norm=False,
                 tp_group=None,
                 tp_size=1):
        super().__init__()
        self.input_layernorm = LayerNorm(normalized_shape=hidden_size,
                                         dtype=dtype)

        self.attention = Attention(
            hidden_size,
            num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            attention_mask_type=AttentionMaskType.causal,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size)

        self.mlp = MLP(hidden_size=hidden_size,
                       ffn_hidden_size=hidden_size * 4,
                       hidden_act=hidden_act,
                       dtype=dtype,
                       tp_group=tp_group,
                       tp_size=tp_size)
        self.post_layernorm = LayerNorm(normalized_shape=hidden_size,
                                        dtype=dtype)

        self.pre_norm = pre_norm

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None):
        residual = hidden_states

        attention_input = hidden_states
        if self.pre_norm:
            attention_input = self.input_layernorm(hidden_states)

        # At this point the hidden_states object must be a Tensor.
        assert isinstance(attention_input, Tensor)

        attention_output = self.attention(attention_input,
                                          attention_mask=attention_mask,
                                          use_cache=use_cache,
                                          kv_cache_params=kv_cache_params,
                                          attention_params=attention_params)
        if use_cache:
            attention_output, presents = attention_output

        hidden_states = residual + attention_output
        if not self.pre_norm:
            hidden_states = self.input_layernorm(hidden_states)

        residual = hidden_states
        if self.pre_norm:
            hidden_states = self.post_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        if not self.pre_norm:
            hidden_states = self.post_layernorm(hidden_states)

        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class OPTModel(Module):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 dtype=None,
                 mapping=Mapping(),
                 pre_norm=False,
                 do_layer_norm_before=True,
                 use_prompt_tuning=False,
                 use_parallel_embedding=False,
                 embedding_sharding_dim=0):
        super().__init__()
        self.do_layer_norm_before = do_layer_norm_before

        self.embedding = GPTEmbedding(
            vocab_size,
            hidden_size,
            max_position_embeddings,
            position_embedding_type=PositionEmbeddingType.learned_absolute,
            dtype=dtype,
            use_prompt_tuning=use_prompt_tuning,
            tensor_parallel=mapping.tp_size if use_parallel_embedding else 1,
            tensor_parallel_group=mapping.tp_group
            if use_parallel_embedding else None,
            sharding_dim=embedding_sharding_dim,
            tp_rank=mapping.tp_rank)

        self.layers = ModuleList([
            OPTDecoderLayer(hidden_size=hidden_size,
                            num_attention_heads=num_heads,
                            max_position_embeddings=max_position_embeddings,
                            dtype=dtype,
                            hidden_act=hidden_act,
                            pre_norm=pre_norm,
                            tp_group=mapping.tp_group,
                            tp_size=mapping.tp_size) for _ in range(num_layers)
        ])

        if self.do_layer_norm_before:
            self.ln_f = LayerNorm(normalized_shape
