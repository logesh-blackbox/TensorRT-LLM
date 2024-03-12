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
from transformers import PreTrainedTokenizerFast

import tensorrt_llm
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import ModelConfig, SamplingConfig

from build import get_engine_name  # isort:skip

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, required=True)
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument('--engine_dir', type=str, default='falcon_outputs')
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        default="tiiuae/falcon-rw-1b",
                        help="Tokenizer path or name.")
    parser.add_argument('--input_text',
                        type=str,
                        default='Born in north-east France, Soyer trained as a')
    parser.add_argument('--input_tokens',
                        dest='input_file',
                        type=str,
                        help=
        'CSV or Numpy file containing tokenized input. Alternative to text input.',
                        default=None)
    parser.add_argument('--output_csv',
                        type=str,
                        help='CSV file where the tokenized output is stored.',
                        default=None)
    parser.add_argument('--output_npy',
                        type=str,
                        help='Numpy file where the tokenized output is stored.',
                        default=None)
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()

# Read the model configuration from a JSON file
def read_config(config_path: Path):
    with config_path.open('r') as f:
        config = json.load(f)

    builder_config = config['builder_config']
    dtype = builder_config['precision']
    tp_size = builder_config['tensor_parallel']
    pp_size = builder_config['pipeline_parallel']
    world_size = tp_size * pp_size
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size '\
        f'({tensorrt_llm.mpi_world_size()})'

    num_heads = builder_config['num_heads'] // tp_size
    num_kv_heads = builder_config.get('num_kv_heads', num_heads)
    num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size
    hidden_size = builder_config['hidden_size'] // tp_size

    vocab_size = builder_config['vocab_size']
    num_layers = builder_config['num_layers']
    quant_mode = QuantMode(builder_config['quant_mode'])

    plugin_config = config['plugin_config']
    use_gpt_attention_plugin = plugin_config['gpt_attention_plugin']
    paged_kv_cache = plugin_config['paged_kv_cache']
    tokens_per_block = plugin_config['tokens_per_block']
    remove_input_padding = plugin_config['remove_input_padding']
    use_custom_all_reduce = plugin_config.get('use_custom_all_reduce', False)

    model_config = ModelConfig(num_heads=num_heads,
                               num_kv_heads=num_kv_heads,
                               hidden_size=hidden_size,
                               vocab_size=vocab_size,
                               num_layers=num_layers,
                               gpt_attention_plugin=use_gpt_attention_plugin,
                               paged_kv_cache=paged_kv_cache,
                               tokens_per_block=tokens_per_block,
                               remove_input_padding=remove_input_padding,
                               quant_mode=quant_mode,
                               dtype=dtype,
                               use_custom_all_reduce=use_custom_all_reduce)

    return model_config, tp_size, pp_size, world_size, dtype

# Prepare input data for the model
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
        elif
