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
from utils import token_encoder

import tensorrt_llm
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import ModelConfig, SamplingConfig

from build import get_engine_name  # isort:skip

# GPT3 Related variables
# Reference : https://github.com/NVIDIA/FasterTransformer/blob/main/sample/pytorch/gpt_sample.py
MERGES_FILE = "merges.txt"
VOCAB_FILE = "vocab.json"

PAD_ID = 50256
START_ID = 50256
END_ID = 50256


def read_config(config_path: Path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
    remove_input_padding = config['plugin_config']['remove_input_padding']
    world_size = config['builder_config']['tensor_parallel']
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
    num_heads = config['builder_config']['num_heads'] // world_size
    hidden_size = config['builder_config']['hidden_size'] // world_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']
    quant_mode = QuantMode(config['builder_config']['quant_mode'])
    paged_kv_cache = config['plugin_config']['paged_kv_cache']
    tokens_per_block = config['plugin_config']['tokens_per_block']
    dtype = config['builder_config']['precision']

    model_config = ModelConfig(num_heads=num_heads,
                               num_kv_heads=num_heads,
                               hidden_size=hidden_size,
                               vocab_size=vocab_size,
                               num_layers=num_layers,
                               gpt_attention_plugin=use_gpt_attention_plugin,
                               remove_input_padding=remove_input_padding,
                               paged_kv_cache=paged_kv_cache,
                               tokens_per_block=tokens_per_block,
                               quant_mode=quant_mode,
                               dtype=dtype)

    max_input_len = config['builder_config']['max_input_len']

    return model_config, world_size, dtype, max_input_len


def parse_input(input_text: str, input_file: str, tokenizer, pad_id: int,
                remove_input_padding: bool):
    input_tokens = []
    if input_file is None:
        input_tokens.append(tokenizer.encode(input_text))
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


def print_output(output_ids, cum_log_probs, input_lengths, sequence_lengths,
                 tokenizer, output_csv, output_npy):

    num_beams = output_ids.size(1)
    if output_csv is None and output_npy is None:
        for b in range(input_lengths.size(0)):
            inputs = output_ids[b][0][:input_lengths[b]].tolist()
            input_text = tokenizer.decode(inputs)
            print(f'Input {b}: \"{input_text}\"')
            for beam in range(num_beams):
                output_begin = input_lengths[b]
                output_end = sequence_lengths[b][beam]
                outputs = output_ids[b][beam][output_begin:output
