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

from ..functional import embedding, unsqueeze, where
from ..module import Module
from ..parameter import Parameter


class Embedding(Module):
    """
    The embedding layer takes input indices (x) and the embedding lookup table (weight) as input.
    And output the corresponding embeddings according to input indices.
    The size of weight is [num_embeddings, embedding_dim]

    Four parameters (tp_size, tp_group, sharding_dim, tp_rank) are involved in tensor parallelism.
    Only when "tp_size > 1 and tp_group is not None", tensor parallelism is enabled.
        When "sharding_dim == 0", the weight is shared in the vocabulary dimension.
            tp_rank must be set when sharding_dim == 0.
        When "sharding_dim == 1",  the weight is shard in the hidden dimension.
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 dtype=None,
                 tp_size=1,
                 tp_group=None,
                 sharding_dim=0,
                 tp_rank=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_size = tp_size
        self.tp_group = tp_group
        self.sharding_dim = sharding_dim
        self.tp_rank = tp_rank

        if sharding_dim == 1:
            self.weight = Parameter(shape=(self.num_embeddings,
                                           self.embedding_dim // self.tp_size),
                                    dtype=dtype)
        elif sharding_dim == 0:
            self.weight = Parameter(shape=(math.ceil(
                self.num_embeddings / self.tp_size), self.embedding_dim),
                                    dtype=dtype)

    def forward(self, x):
        return embedding(x,
                         self.weight.value,
                         tp_size=self.tp_size,
                         tp_group=self.tp_group,
                         sharding_dim=self.sharding_dim,
                         tp_rank=self.tp_rank)


class PromptTuningEmbedding(Module):
    """
        Pass all tokens though both normal and prompt embedding tables.
    Then, combine results based on whether the token was "normal" or "prompt/virtual".
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 vocab_size=None,
                 dtype=None,
                 tp_size=1,
                 tp_group=None,
                 sharding_dim=0,
                 tp_rank=0):
        super().__init__()
        if vocab_size is None:
            vocab_size = num_embeddings
        self.vocab_size = vocab_size
        self.embedding = Embedding(num_embeddings, embedding_dim, dtype, tp_size,
                                   tp_group, sharding_dim, tp_rank)
        self.prompt_embedding = Embedding(vocab_size, embedding_dim, dtype, tp_size,
                                          tp_group, sharding_dim, tp_rank)

    def forward(self, tokens, prompt_embedding_table):
        prompt_tokens_mask = tokens > (
