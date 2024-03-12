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
'''
Convert huggingface Meta OPT model. Use https://huggingface.co/facebook/opt-125m as demo.
'''

import argparse
import configparser
import multiprocessing
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM  # transformers-4.20.0.dev0

from tensorrt_llm.logger import logger


def save_val(val, dir, key, tp_num=None):
    path = str(dir / ("model." + key))
    if tp_num is not None:
        path += "." + str(tp_num)
    path += ".bin"

    val.tofile(path)


def save_split(split_vals, dir, key, i, factor):
    for j, val in enumerate(split_vals):
        save_val(val, dir, key, i * factor + j)


def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    else:
        assert False, f"Invalid weight data type {data_type}"


def quantize(mat, act_range):
    # qkv proj weight quantization
    if mat.ndim == 3 and mat.shape[1] == 3:
        # get max_q, max_k, max_v
        mat_max = np.abs(mat).clip(1e-8, None).max(axis=(0, 2))[None, :, None]
    else:
        mat_max = np.abs(mat).clip(1e-8, None).max()

    act_scale_in = 127. / np.array(act_range["input"])
    weight_scales = 127. / mat_max
    act_scale_post = 127. / np.array(act_range["output"])

    mat_quant = (mat * weight_scales).round().astype(np.int8)
    return mat_quant, weight_scales, act_scale_in, act_scale_post


def split_and_convert_process(i, saved_dir, factor, key, args, val, old_name,
                              dtype):
    logger.debug(f"split_and_convert_process {key}")
    old_name.rpartition(".")[0]

    if "input_layernorm.weight" in key or "input_layernorm.bias" in key or \
        "attention.dense.bias" in key or "post_attention_layernorm.weight" in key or \
        "post_attention_layernorm.bias" in key or "mlp.dense_4h_to_h.bias" in key or \
        "final_layernorm.weight" in key or "final_layernorm.bias" in key:

        # shared weights, only need to convert the weights of rank 0
        if i == 0:
            save_val(val, saved_dir, key)

    elif "attention.dense.weight" in key or "mlp.dense_4h_to_h.weight" in key:
        save_split(np.split(val, factor, axis=0), saved_dir, key, i, factor)

    elif "mlp.dense_h_to_4h.weight" in key or "mlp.dense_h_to_4h.bias" in key:
        save_split(np.split(val, factor, axis=-1), saved_dir, key, i, factor)

    elif "attention.query_key_value.bias" in key:
        local_dim = val.shape[-1] // 3

        val = val.reshape(3, local_dim)
        save_split(np.split(val, factor, axis=-1), saved_dir, key, i, factor)

    elif "attention.query_key_value.weight" in key:
        hidden_dim = val.shape[0]
        local_dim = val.shape[-1] // 3

        val = val.reshape(hidden_dim, 3, local_dim)
        save_split(np.split(val, factor, axis=-1), saved_dir, key, i, factor)

    else:
        logger.error("[ERROR] Key '{}' not handled".format(key))


@torch.no_grad()
def split_and_convert(args):
    saved_dir = Path(args.saved_dir) / f"{args.infer_gpu_num}-gpu"

    if not saved_dir.exists():
        saved_dir.mkdir(parents=True)

    t_gpu_num = args.trained_gpu_num
    i_gpu_num = args.infer_gpu_num
    assert (i_gpu_num % t_gpu_num == 0)

    factor = (int)(i_gpu_num / t_gpu_num)

    # load position_embedding from rank 0
    model = AutoModelForCausalLM.from_pretrained(args.in_file,
                                                 device_map="auto")
    hf_config = vars(model.config)

    num
