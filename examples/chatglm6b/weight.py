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
import configparser
import time
from pathlib import Path

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm.functional import is_gated_activation
from tensorrt_llm.models import ChatGLM6BHeadModel
from tensorrt_llm.quantization import QuantMode


def gen_suffix(rank, use_smooth_quant, quant_per_channel):
    suffix = f"{rank}.bin"
    if use_smooth_quant:
        sq_prefix = "int8."
        if quant_per_channel:
            sq_prefix += "col."
        suffix = sq_prefix + suffix
    return suffix


def extract_layer_idx(name):
    ss = name.split('.')
    for s in ss:
        if s.isdigit():
            return s
    return None


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return np.ascontiguousarray(np.split(v, tp_size)[idx])
    elif len(v.shape) == 2:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])
    return None


def parse_ft_config(ini_file):
    chatglm6b_config = configparser.ConfigParser()
    chatglm6b_config.read(ini_file)

    n_embd = chatglm6b_config.getint('chatglm6b', 'hidden_size')
    n_head = chatglm6b_config.getint('chatglm6b', 'num_attention_heads')
    n_layer = chatglm6b_config.getint('chatglm6b', 'num_layers')
    n_positions = chatglm6b_config.getint('chatglm6b', 'max_sequence_length')
    vocab_size = chatglm6b_config.getint('chatglm6b', 'vocab_size')
    do_layer_norm_before = chatglm6b_config.getboolean('chatglm6b',
                                                       'do_layer_norm_before',
                                                       fallback=True)
    rotary_pct = chatglm6b_config.getfloat('chatglm6b',
                                           'rotary_pct',
                                           fallback=0.0)
    hidden_act = 'gelu'  #chatglm6b_config.get('chatglm6b', 'activation_function')
    bias = chatglm6b_config.getboolean('chatglm6b', 'bias', fallback=True)
    inter_size = chatglm6b_config.getint('chatglm6b',
                                         'intermediate_size',
                                         fallback=None)

    if inter_size is None:
        inter_size = 4 * n_embd

    multi_query_mode = chatglm6b_config.getboolean('chatglm6b',
                                                   'multi_query_mode',
                                                   fallback=False)
    return n_embd, n_head, n_layer, n_positions, vocab_size, do_layer_norm_before, hidden_act, rotary_pct, bias, inter_size, multi_query_mode


def load_from_ft(chatglm6bModel: ChatGLM6BHeadModel,
                 dir_path,
                 rank=0,
                 tensor_parallel=1,
                 fp16=False):
    tensorrt_llm.logger.info('Loading weights from FT...')
    tik = time.time()

    quant_mode = getattr(chatglm6bModel, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    n_embd, n_head, n_layer, n_positions, vocab_size, do_layer_norm_before, hidden_act, rotary_pct, bias, inter_size, multi_query_mode = parse_ft_config(
        Path(dir_path) / 'config.ini')
    np_dtype = np.float16 if fp16 else np.float32

    def fromfile(dir_path, name, shape=None, dtype=None):
        dtype = np_dtype if dtype is None else dtype
        p = dir_path + '/' + name
        if Path(p).exists():
            t = np.fromfile(p, dtype=dtype)
            if shape is not None:
                t = t.reshape(shape)
            return t
        return None

    def set_smoothquant_scale_factors(module,
                                      pre_scale_weight,
                                      dir_path,
                                      basename,
                                      shape,
                                      per_tok_dyn,
                                      per_channel,
                                      is_qkv=False,
                                      rank=None):
        suffix = "bin"

