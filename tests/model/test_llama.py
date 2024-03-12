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
import os
import random
import sys
import tempfile
import unittest
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch
from parameterized import parameterized
from transformers import LlamaConfig, LlamaForCausalLM

import tensorrt_llm
from tensorrt_llm import Builder
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.layers import PositionEmbeddingType
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from examples.llama.weight import load_from_hf_llama

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import getSMVersion


class TestLLaMA(unittest.TestCase):
    EOS_TOKEN = 2
    PAD_TOKEN = 2

    def _gen_tensorrt_llm_network(self, network, hf_llama,
                                  llama_config: LlamaConfig, batch_size,
                                  beam_width, input_len, output_len, dtype,
                                  rank, tensor_parallel):
        list(range(tensor_parallel))

        with net_guard(network):
            kv_dtype = str_dtype_to_trt(dtype)

            # Initialize model
            tensorrt_llm_llama = tensorrt_llm.models.LLaMAForCausalLM(
                num_layers=llama_config.num_hidden_layers,
                num_heads=llama_config.num_attention_heads,
                num_kv_heads=llama_config.num_key_value_heads,
                hidden_size=llama_config.hidden_size,
                vocab_size=llama_config.vocab_size,
                hidden_act=llama_config.hidden_act,
                max_position_embeddings=llama_config.max_position_embeddings,
                dtype=kv_dtype,
                mlp_hidden_size=llama_config.intermediate_size,
                position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
                mapping=tensorrt_llm.Mapping(world_size=tensor_parallel,
                                             tp_size=tensor_parallel),
            )
            load_from_hf_llama(tensorrt_llm_llama,
                               hf_llama,
                               dtype=dtype,
                               mapping=tensorrt_llm.Mapping(
                                   world_size=tensor_parallel,
                                   rank=rank,
                                   tp_size=tensor_parallel))
            # Prepare
            network.set_named_parameters(tensorrt_llm_llama.named_parameters())
            inputs = tensorrt_llm_llama.prepare_inputs(batch_size, input_len,
                                                       output_len, True,
                                                       beam_width)
            # Forward
            tensorrt_llm_llama(*inputs)

        return network

    def _gen_tensorrt_llm_engine(self,
                                 dtype,
                                 rank,
                                 world_size,
                                 llama_config,
                                 hf_llama,
                                 model_name,
                                 use_plugin,
                                 batch_size,
                                 beam_width,
                                 input_len,
                                 output_len,
                                 use_refit,
                                 fast_building=False,
                                 context_fmha_flag=ContextFMHAType.disabled,
                                 enable_remove_input_padding=False):

        builder = Builder()

        with tempfile.TemporaryDirectory() as tmpdirname:
            network = builder.create_network()
            if use_plugin:
                network.plugin_config.set_gpt_attention_plugin(dtype)
            if fast_building:
                network.plugin_config.set_gemm_plugin(dtype)
            if enable_remove_input_padding:
                network.plugin_config.enable_remove_input_padding()
            network.plugin_config.set_context_fmha(context_fmha_flag)

            self._gen_tensorrt_llm_network(network, hf_llama, llama_config,
                                           batch_size, beam_width, input_len,
                                           output_len, dtype, rank, world_size)

            builder_config = builder.create_builder_config(
                name=model_name,
                precision=dtype,
                timing_cache='model.cache',
                tensor_parallel=world_size,  # TP only
                use_refit=use_refit,
            )
            engine_buffer = builder.build_engine(network, builder_config)
            return engine_buffer

    def _gen_tensorrt_llm_runtime(self,
                                  log_level,
                                  dtype,
                                  world_size,
                                  rank,
                                  llama_config,
                                  hf_llama,
                                  model_name,
