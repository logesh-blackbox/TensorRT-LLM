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
from ...functional import gather_last_token_logits, recv, send
from ...layers import (Attention, AttentionMaskType, AttentionParams,
                       ColumnLinear, Embedding, GatedMLP, KeyValueCacheParams,
                       PositionEmbeddingType, RmsNorm)
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...quantization import QuantMode
from ..generation_mixin import GenerationMixin


class LLaMADecoderLayer(Module):

    def __init__(self,
                 layer_id,
                 hidden_size,
                 num_attention_heads,
                 num_kv_heads=None,
                 max_position_embeddings=2048,
                 dtype=None,
                 attention_mask_type=AttentionMaskType.causal,
                 hidden_act='silu',
                 position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
                 rotary_base=10000.0,
                 rotary_scaling=None,
                 mlp_hidden_size=None,
                 tp_group=None,
                 tp_size=1,
                 quant_mode=QuantMode(0),
                 rms_norm_eps=1e-06):
        super().__init__()
        self._layer_id = layer_id  # useful for debugging
        # used for quantizing model
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.max_position_embeddings = max_position_embeddings
        self.dtype = dtype
        self.hidden_act = hidden_act
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.mlp_hidden_size = mlp_hidden_size
        self.attention_mask_type = attention_mask_type
        self.position_embedding_type = position_embedding_type
        self.input_layernorm = RmsNorm(normalized_shape=hidden_size,
                                       eps=rms_norm_eps,
                                       dtype=dtype)

        self.attention = Attention(
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            max_position_embeddings,
            dtype=dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=False,
            position_embedding_type=position_embedding_type,
            rotary_embedding_base=rotary_base,
            rotary_embedding_scaling=rotary_scaling,
            tp_group=tp_group,
            tp_size=tp_size,
            use_int8_kv_cache=quant_mode.has_int8_kv_cache(),
            quant_mode=quant_mode,
            instance_id=2 * layer_id,
        )
        if not mlp_hidden_size:
            self.mlp_hidden_size = hidden_size * 4
        self.mlp = GatedMLP(hidden_size=hidden_size,
                            ffn_hidden_size=self.mlp_hidden_size,
                            hidden_act=hidden_act,
                            dtype=dtype,
                            bias=False,
                            tp_group=tp_group,
                            tp_size=tp_size,
                            quant_mode=quant_mode,
                            instance_id=2 * layer_id + 1)
        self.post_layernorm = RmsNorm(normalized_shape=hidden_size,
                                      eps=rms_norm_eps,
                                      dtype=dtype)

    def forward(self,
                hidden_states,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None,
                all_reduce_workspace=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self._layer_id == 0:
            self.register_network_output(f"norm0", hidden_states)

        attention_output = self.attention(hidden_states,
                                          attention_mask=attention_mask,
                                          use_cache=use_cache,
                                          kv_cache_params=kv_cache_params,
                                          attention_params=attention_params,
                                          workspace=all_reduce_workspace)

        if use_cache:
            attention_output, presents = attention_output
        if self._layer_id == 0:
            self.register_network_output(f"attn", attention_output)

        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        if self._layer_id == 0:
            self.register_network_output(f"norm1", hidden_states)

        hidden_states = self.mlp(hidden_states, all_reduce_workspace)
        if self._layer_id ==
