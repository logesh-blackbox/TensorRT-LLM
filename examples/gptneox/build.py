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
import json
import os
import time

import torch
import torch.multiprocessing as mp
from safetensors import safe_open
from transformers import AutoModelForCausalLM, GPTNeoXConfig
from weight import load_from_hf_gpt_neox

import tensorrt_llm
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import (weight_only_groupwise_quantize,
                                 weight_only_quantize)
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode

MODEL_NAME = "gptneox"
hf_gpt = None


class StateDict():

    def __init__(self, quant_ckpt_dir):
        self.model_state_dict = safe_open(quant_ckpt_dir,
                                          framework="pt",
                                          device=0)

    def get(self, k):
        return self.model_state_dict.get_tensor(k).cpu()


class GPTQModel():

    def __init__(self, model_dir, quant_ckpt_dir):
        with open(model_dir + '/config.json', 'r') as f:
            model_config = json.load(f)
            self.config = GPTNeoXConfig()
            self.config.vocab_size = model_config['vocab_size']
            self.config.hidden_size = model_config['hidden_size']
            self.config.num_hidden_layers = model_config['num_hidden_layers']
            self.config.num_attention_heads = model_config[
                'num_attention_heads']
            self.config.intermediate_size = model_config['intermediate_size']
            self.config.hidden_act = model_config['hidden_act']
            self.config.rotary_pct = model_config['rotary_pct']
            self.config.rotary_emb_base = model_config['rotary_emb_base']
            self.config.max_position_embeddings = model_config[
                'max_position_embeddings']
            self.config.initializer_range = model_config['initializer_range']
            self.config.layer_norm_eps = model_config['layer_norm_eps']
            self.config.use_cache = model_config['use_cache']
            self.config.bos_token_id = model_config['bos_token_id']
            self.config.eos_token_id = model_config['eos_token_id']
            self.config.tie_word_embeddings = model_config[
                'tie_word_embeddings']
        self.model_state_dict = StateDict(quant_ckpt_dir)

    def state_dict(self):
        return self.model_state_dict


def get_engine_name(model, dtype, tp_size, rank):
    return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)


def serialize_engine(engine, path):
    logger.info(f'Serializing engine to {path}...')
    tik = time.time()
    with open(path, 'wb') as f:
        f.write(bytearray(engine))
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help='world size, only support tensor parallelism now')
    parser.add_argument(
        '--model_dir',
        type=str,
        default=None,
        help='The path to HF GPT-NeoX model / checkpoints to read weights from')
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'float32'])
    parser.add_argument(
        '--timing_cache',
        type=str,
        default='model.cache',
        help=
        'The path of to read timing cache from, will be ignored if the file does not exist'
    )
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--vocab_size', type=int, default=50432)
    parser.add_argument('--n_layer', type=int, default=44)
    parser.add_argument('--n_positions', type=int, default=2048)
    parser.add_argument('--n_embd', type=int, default=6144)
    parser.add_argument('--n_head', type=int, default=64)
    parser.add_argument('--hidden_act', type=str, default='gelu')
    parser.add_argument(
