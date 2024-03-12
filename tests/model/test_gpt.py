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
import os
import random
import sys
import tempfile
import unittest
from itertools import product

import numpy as np
import pytest
import tensorrt as trt
import torch
from parameterized import parameterized
from transformers import GPT2Config, GPT2LMHeadModel

import tensorrt_llm
from tensorrt_llm import Builder
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from tensorrt_llm.runtime.generation import _prepare_attention_mask
from tensorrt_llm.runtime.kv_cache_manager import (GenerationSequence,
                                                   KVCacheManager)

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from examples.gpt.weight import load_from_hf_gpt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import getSMVersion


class TestGPT(unittest.TestCase):

    def _gen_hf_gpt(self, hidden_act, n_layer, max_length, dtype):
        """
        Generate a Hugging Face GPT model and its configuration.

        :param hidden_act: Activation function to use in the GPT model.
        :param n_layer: Number of layers in the GPT model.
        :param max_length: Maximum sequence length for the GPT model.
        :param dtype: Data type to use for the GPT model.
        :return: A tuple containing the GPT model configuration and the GPT model.
        """
        gpt_config = GPT2Config(
            activation_function=hidden_act,
            n_layer=n_layer,
            max_length=max_length,
            torch_dtype=dtype,
        )
        hf_gpt = GPT2LMHeadModel(gpt_config).cuda().eval()
        return gpt_config, hf_gpt

    def _gen_tensorrt_llm_network(self, network, builder, hf_gpt, gpt_config,
                                  batch_size, input_len, output_len, fp16,
                                  gpt_attention_plugin, tensor_parallel,
                                  apply_query_key_layer_scaling,
                                  gather_all_token_logits):
        """
        Generate a TensorRT LLM network for the GPT model.

        :param network: The TensorRT network to generate the GPT model for.
        :param builder: The TensorRT builder to use for building the network.
        :param hf_gpt: The Hugging Face GPT model to generate the TensorRT LLM network for.
        :param gpt_config: The configuration for the GPT model.
        :param batch_size: The batch size to use for the GPT model.
        :param input_len: The length of the input sequence for the GPT model.
        :param output_len: The length of the output sequence for the GPT model.
        :param fp16: Whether to use FP16 precision for the GPT model.
        :param gpt_attention_plugin: Whether to use the GPT attention plugin for the GPT model.
        :param tensor_parallel: The number of GPUs to use for tensor parallelism.
        :param apply_query_key_layer_scaling: Whether to apply query-key layer scaling for the GPT model.
        :param gather_all_token_logits: Whether to gather all token logits for the GPT model.
        :return: The TensorRT network with the generated GPT model.
        """
        num_layers = gpt_config.n_layer
        num_heads = gpt_config.n_head
        hidden_size = gpt_config.n_embd
        vocab_size = gpt_config.vocab_size
        hidden_act = gpt_config.activation_function
        n_positions = gpt_config.n_positions
        tensor_parallel_group = list(range(tensor_parallel))

        with net_guard(network):
            kv_dtype = trt.float16 if fp16 else trt.float32
            # Initialize model
            tensorrt_llm_gpt = tensorrt_llm.models.GPTLMHeadModel(
                num_layers=num_layers,
                num_heads=num_heads,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                hidden_act=hidden_act,
                max_position_embeddings=n_positions,
                dtype=kv_dtype,
                mapping=tensorrt_llm.Mapping(world_size=tensor_parallel,
                                             tp_size=tensor_parallel),
                apply_query_key_layer_scaling=apply_query_key_layer_scaling)
            inputs = tensorrt_llm_
