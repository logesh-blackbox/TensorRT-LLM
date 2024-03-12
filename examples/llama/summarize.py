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

import argparse
import copy
import json
import os
import torch
from datasets import load_dataset, load_metric
from transformers import AutoModelForCausalLM, LlamaTokenizer

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization import QuantMode

from build import get_engine_name  # isort:skip


def TRTLLaMA(args, config):
    dtype = config['builder_config']['precision']
    tp_size = config['builder_config']['tensor_parallel']
    pp_size = config['builder_config']['pipeline_parallel']
    world_size = tp_size * pp_size

    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'

    num_heads = config['builder_config']['num_heads'] // tp_size
    hidden_size = config['builder_config']['hidden_size'] // tp_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']
    use_gpt_attention_plugin = bool(
        config['plugin_config']['gpt_attention_plugin'])
    remove_input_padding = config['plugin_config']['remove_input_padding']
    num_kv_heads = config['builder_config'].get('num_kv_heads', num_heads)
    paged_kv_cache = config['plugin_config']['paged_kv_cache']
    tokens_per_block = config['plugin_config']['tokens_per_block']
    use_custom_all_reduce = config['plugin_config'].get('use_custom_all_reduce',
                                                        False)

    quant_mode = QuantMode(config['builder_config']['quant_mode'])
    if config['builder_config'].get('multi_query_mode', False):
        tensorrt_llm.logger.warning(
            "`multi_query_mode` config is deprecated. Please rebuild the engine."
        )
        num_kv_heads = 1
    num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size

    model_config = tensorrt_llm.runtime.ModelConfig(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        paged_kv_cache=paged_kv_cache,
        tokens_per_block=tokens_per_block,
        gpt_attention_plugin=use_gpt_attention_plugin,
        remove_input_padding=remove_input_padding,
        use_custom_all_reduce=use_custom_all_reduce,
        dtype=dtype,
        quant_mode=quant_mode)

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=tp_size,
                                           pp_size=pp_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    engine_name = get_engine_name('llama', dtype, tp_size, pp_size,
                                  runtime_rank)
    serialize_path = os.path.join(args.engine_dir, engine_name)

    tensorrt_llm.logger.set_level(args.log_level)

    profiler.start('load tensorrt_llm engine')
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                     engine_buffer,
                                                     runtime_mapping)
    profiler.stop('load tensorrt_llm engine')
    tensorrt_llm.logger.info(
        f'Load engine takes: {profiler.elapsed_time_in_sec("load tensorrt_llm engine")} sec'
    )
    return decoder


def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    test_hf = args.test_hf and runtime_rank == 0  # only run hf on rank 0
    test_trt_llm = args.test_trt_llm
    hf_model_location = args.hf_model_location
    profiler.start('load tokenizer')
    tokenizer = LlamaTokenizer.from_pretrained(hf_model_location,
                                               legacy=False,
                                               padding_side='left')
    profiler.stop('load tokenizer')
    tensorrt_llm.logger.info(
        f'Load tokenizer takes: {profiler.elapsed_time_in_sec("load tokenizer")} sec'
    )
    tokenizer.pad_token = tokenizer.eos_token

    dataset_cnn = load_dataset("ccdv
