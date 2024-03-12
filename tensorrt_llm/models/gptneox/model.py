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
import tensorrt as trt

from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt
from ...functional import (PositionEmbeddingType, Tensor,
                           gather_last_token_logits, gpt_attention)
from ...layers import (MLP, AttentionMaskType, AttentionParams, ColumnLinear,
                       Embedding, KeyValueCacheParams, LayerNorm, RowLinear)
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...parameter import Parameter
from ...quantization import QuantMode
from ..generation_mixin import GenerationMixin


class GPTNeoXAttention(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 rotary_dim,
                 max_position_embeddings,
                 dtype=None,
                 multi_block_mode=False,
                 position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
                 quant_mode=QuantMode(0),
                 tp_group=None,
                 tp_size=1):
        super().__init__()
        self.attention_head_size = hidden_size // num_attention_heads
        self.num_attention_heads = num_attention_heads // tp_size
        self.max_position_embeddings = max_position_embeddings
        self.rotary_dim = rotary_dim
        self.position_embedding_type = position_embedding_type
        self.multi_block_mode = multi_block_mode
        self.multi_query_mode = False
        self.quant_mode = quant_mode

        if self.quant_mode.has_int8_kv_cache():
            self.kv_quantization_scale = Parameter(shape=(1, ), dtype='float32')
            self.kv_dequantization_scale = Parameter(shape=(1, ),
                                                     dtype='float32')
        else:
            self.register_parameter('kv_quantization_scale', None)
            self.register_parameter('kv_dequantization_scale', None)

        self.qkv = ColumnLinear(in_features=hidden_size,
                                out_features=hidden_size * 3,
                                bias=True,
                                tp_group=tp_group,
                                tp_size=tp_size,
                                gather_output=False,
                                dtype=dtype)
        self.dense = RowLinear(in_features=hidden_size,
                               out_features=hidden_size,
                               bias=True,
                               dtype=dtype,
                               tp_group=tp_group,
                               tp_size=tp_size)

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None):
        if not default_net().plugin_config.gpt_attention_plugin:
            raise ValueError(
                'GPT-NeoX RoPE is only supported with GPTAttention plugin')
        qkv = self.qkv(hidden_states)

        assert attention_params.is_valid(
            default_net().plugin_config.gpt_attention_plugin,
            default_net().plugin_config.remove_input_padding)
        assert kv_cache_params.is_valid(
            default_net().plugin_config.gpt_attention_plugin)

        context, past_key_value = gpt_attention(
            tensor=qkv,
            past_key_value=kv_cache_params.get_first_past_key_value(),
            sequence_length=attention_params.sequence_length,
            host_past_key_value_lengths=kv_cache_params.
            host_past_key_value_lengths,
            context_lengths=attention_params.context_lengths,
            cache_indirection=kv_cache_params.cache_indirection,
            host_request_types=attention_params.host_request_types,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_attention_heads,
            hidden_size_per_head=self.attention_head_size,
            q_scaling=1.0,
            rotary_embedding_dim=self.rotary_dim,
            position_embedding_type=self.position_embedding_type,
            multi_block_mode=self.multi_block_mode,
            kv_orig_quant_scale=self.kv_quantization_scale,
            kv_quant_orig_scale=self.kv_dequantization_scale,
            kv_cache_quant_mode=self.quant_mode,
            max_context_length=attention_params.max_context_length,
            host_context_lengths=attention_params.host_context_lengths)

        context = self.dense(context)

        if use_cache:
            return (context, past_key_value)

        return context


class GPTNeoXDecoderLayer(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings,

