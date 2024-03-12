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
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm.logger import logger

from build import find_engines  # isort:skip


def TRTGPT(args, config):
    dtype = config['builder_config']['precision']
    world_size = config['builder_config']['tensor_parallel']
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'

    world_size = config['builder_config']['tensor_parallel']
    num_heads = config['builder_config']['num_heads'] // world_size
    hidden_size = config['builder_config']['hidden_size'] // world_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']
    use_gpt_attention_plugin = bool(
        config['plugin_config']['gpt_attention_plugin'])
    remove_input_padding = config['plugin_config']['remove_input_padding']
    multi_query_mode = config['builder_config']['multi_query_mode']
    num_kv_heads = 1 if multi_query_mode else num_heads
    paged_kv_cache = config['plugin_config']['paged_kv_cache']
    tokens_per_block = config['plugin_config']['tokens_per_block']
    use_custom_all_reduce = config['plugin_config']['use_custom_all_reduce']

    model_config = tensorrt_llm.runtime.ModelConfig(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        gpt_attention_plugin=use_gpt_attention_plugin,
        remove_input_padding=remove_input_padding,
        tokens_per_block=tokens_per_block,
        paged_kv_cache=paged_kv_cache,
        dtype=dtype,
        use_custom_all_reduce=use_custom_all_reduce,
    )

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=world_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    serialize_path = find_engines(args.engine_dir,
                                  dtype=dtype,
                                  tp_size=world_size,
                                  rank=runtime_rank)[0]

    tensorrt_llm.logger.set_level(args.log_level)

    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                     engine_buffer,
                                                     runtime_mapping)

    return decoder


def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    test_hf = args.test_hf and runtime_rank == 0  # only run hf on rank 0
    test_trt_llm = args.test_trt_llm
    hf_model_location = args.hf_model_location

    if args.vocab_file is not None:
        tokenizer = T5Tokenizer(vocab_file=args.vocab_file, padding_side='left')
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,
                                                  padding_side='left')

    tokenizer.pad_token = tokenizer.eos_token

    if args.eval_type == 'code_completion':
        dataset_name = "openai_humaneval"
        dataset_revision = None
        dataset_input_key = 'prompt'
        dataset_output_key = 'canonical_solution'
    elif args.eval_type == 'summarize':
        dataset_name = "ccdv/cnn_dailymail"
        dataset_revision = "3.0.0"
        dataset_input_key = 'article'
        dataset_output_key = 'highlights'
    dataset = load_dataset(dataset_name,
                           dataset_revision,
                           cache_dir=args.dataset_path)

    config_path = str(args.engine_dir / 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    max_batch_size = args.batch_size

    # runtime parameters
    # repetition_penalty = 1
    top_k = args.top_k
    output_len = args.output_len
    test_token_num = 92
