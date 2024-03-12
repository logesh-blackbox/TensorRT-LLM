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

import numpy as np

from ..._common import default_net
from ...functional import (bert_attention, concat, constant, expand, matmul,
                           shape, slice, softmax, split)
from ...layers import MLP, ColumnLinear, Embedding, LayerNorm, Linear, RowLinear
from ...mapping import Mapping
from ...module import Module, ModuleList


class BertEmbedding(Module):

    def __init__(self,
                 vocab_size,
                 hidden_size,
                 max_position_embeddings,
                 type_vocab_size,
                 dtype=None):
        super().__init__()
        self.vocab_embedding = Embedding(vocab_size, hidden_size, dtype=dtype)
        self.position_embedding = Embedding(max_position_embeddings,
                                            hidden_size,
                                            dtype=dtype)
        self.token_embedding = Embedding(type_vocab_size,
                                         hidden_size,
                                         dtype=dtype)
        self.max_position_embeddings = max_position_embeddings

        self.embedding_ln = LayerNorm(normalized_shape=hidden_size, dtype=dtype)

    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        position_ids_buffer = constant(
            np.expand_dims(
                np.arange(self.max_position_embeddings).astype(np.int32), 0))

        token_type_ids_buffer = constant(
            np.expand_dims(
                np.zeros(self.max_position_embeddings).astype(np.int32), 0))

        seq_len_2d = concat([1, shape(input_ids, 1)])

        if position_ids is None:
            position_ids = slice(position_ids_buffer,
                                 starts=[0, 0],
                                 sizes=seq_len_2d)
            position_ids = expand(position_ids, shape(input_ids))

        if token_type_ids is None:
            token_type_ids = slice(token_type_ids_buffer,
                                   starts=[0, 0],
                                   sizes=seq_len_2d)
            token_type_ids = expand(token_type_ids, shape(input_ids))

        x = self.vocab_embedding(input_ids)
        x = x + self.position_embedding(position_ids)
        x = x + self.token_embedding(token_type_ids)
        x = self.embedding_ln(x)
        return x


class BertAttention(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings,
                 dtype=None,
                 tp_group=None,
                 tp_size=1):
        super().__init__()

        self.attention_head_size = hidden_size // num_attention_heads
        self.num_attention_heads = num_attention_heads // tp_size
        self.hidden_size = hidden_size // tp_size
        self.max_position_embeddings = max_position_embeddings
        self.norm_factor = math.sqrt(self.attention_head_size)

        self.qkv = ColumnLinear(hidden_size,
                                hidden_size * 3,
                                dtype=dtype,
                                tp_group=tp_group,
                                tp_size=tp_size,
                                gather_output=False)
        self.dense = RowLinear(hidden_size,
                               hidden_size,
                               dtype=dtype,
                               tp_group=tp_group,
                               tp_size=tp_size)

    def forward(self, hidden_states, attention_mask=None, input_lengths=None):
        qkv = self.qkv(hidden_states)

        # attention
        if default_net().plugin_config.bert_attention_plugin:
            assert input_lengths is not None
            context = bert_attention(qkv, input_lengths,
                                     self.num_attention_heads,
                                     self.attention_head_size, 1.0)
        else:
            query, key, value = split(qkv, self.hidden_size, dim=2)
            query = query.permute([0, 2, 1, 3])
            key = key.permute([0, 2, 1, 3])
            value = value.permute([0, 2, 1, 3])

            attention_scores = matmul(query, key.transpose(-1, -2))
            attention_scores = attention_scores / self.norm_factor

            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            attention_probs = softmax(attention_scores, axis=-1)

            context = matmul(attention_probs, value)
            context = context.permute([0, 2, 1, 3])

        output = self.dense(
