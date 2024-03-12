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
import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, T5Tokenizer

import tensorrt_llm
from tensorrt_llm.runtime import ModelConfig, SamplingConfig

from build import get_engine_name  # isort:skip


def read_config(config_path: Path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
    remove_input_padding = config['plugin_config']['remove_input_padding']
    tp_size = config['builder_config']['tensor_parallel']
    pp_size = config['builder_config'].get('pipeline_parallel', 1)
    world_size = tp_size * pp_size
    assert tp_size * pp_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({tp_size} * {pp_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
    assert (config['builder_config']['num_heads'] %
            tp_size) == 0, f"The number of heads must be a multiple of tp_size"
    num_heads = config['builder_config']['num_heads'] // tp_size
    num_kv_heads = (config['builder_config']['num_kv_heads'] + tp_size -
                    1) // tp_size
    hidden_size = config['builder_config']['hidden_size'] // tp_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']
    paged_kv_cache = config['plugin_config']['paged_kv_cache']
    tokens_per_block = config['plugin_config']['tokens_per_block']
    use_prompt_tuning = config['builder_config']['use_prompt_tuning']
    dtype = config['builder_config']['precision']
    gather_all_token_logits = config['builder_config'][
        'gather_all_token_logits']
    use_custom_all_reduce = config['plugin_config']['use_custom_all_reduce']

    model_config = ModelConfig(num_heads=num_heads,
                               num_kv_heads=num_kv_heads,
                               hidden_size=hidden_size,
                               vocab_size=vocab_size,
                               num_layers=num_layers,
                               gpt_attention_plugin=use_gpt_attention_plugin,
                               remove_input_padding=remove_input_padding,
                               paged_kv_cache=paged_kv_cache,
                               tokens_per_block=tokens_per_block,
                               use_prompt_tuning=use_prompt_tuning,
                               dtype=dtype,
                               gather_all_token_logits=gather_all_token_logits,
                               use_custom_all_reduce=use_custom_all_reduce)

    dtype = config['builder_config']['precision']
    max_input_len = config['builder_config']['max_input_len']

    return model_config, world_size, dtype, max_input_len


def parse_input(input_text: str, input_file: str, tokenizer, pad_id: int,
                remove_input_padding: bool):
    input_tokens = []
    if input_file is None:
        input_tokens.append(
            tokenizer.encode(input_text, add_special_tokens=False))
    else:
        if input_file.endswith('.csv'):
            with open(input_file, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for line in csv_reader:
                    input_tokens.append(np.array(line, dtype='int32'))
        elif input_file.endswith('.npy'):
            inputs = np.load(input_file)
            for row in inputs:
                row = row[row != pad_id]
                input_tokens.append(row)
        else:
            print('Input file format not supported.')
            raise SystemExit

    input_ids = None
    input_lengths = torch.tensor([len(x) for x in input_tokens],
                                 dtype=torch.int32,
                                 device='cuda')
    if remove_input_padding:
        input_ids = np.concatenate(input_tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.int32,
                                 device='cuda').unsqueeze(0)
    else:
        input_ids = torch.nested.to_padded_tensor(
            torch.nested.nested_tensor(input_tokens, dtype=torch.int32),
            pad_id).cuda()

    return input_ids, input_lengths


def ptuning_setup(prompt_table, dtype, hidden_size, tasks, input_ids,
                  input_lengths, remove_input_padding):
    if prompt_table is not None:
        prompt_table = torch.from_numpy(np.load(prompt_table
