# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2022, MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert MPT model checkpoint to FT format.

It's a modified version of
https://github.com/NVIDIA/FasterTransformer/blob/main/examples/pytorch/gpt/utils/huggingface_gpt_convert.py
"""

import argparse
import configparser
import os
from typing import Any, Dict, List

import numpy as np
import torch
import transformers

from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy

# numpy doesn't know bfloat16, define abstract binary type instead
np_bfloat16 = np.dtype('V2', metadata={"dtype": "bfloat16"})


def write_zero_bias(weight_name: str, weight_file_path: str,
                    bias_shape: List[int], data_type: torch.dtype) -> None:
    """Write zeros for bias.

    MPT model might not have bias while FT expects bias.

    Args:
        weight_name (str): Name of the weight tensor.
        weight_file_path (str): Output path for storing the weight (NOT zero bias).
        bias_shape (List[int]): Shape of the bias array.
    """
    if 'weight' not in weight_file_path:
        raise RuntimeError(
            f'Cannot write zero bias for {weight_name}. Input is not a weight tensor'
        )
    print(f'zero bias for weight: {weight_name}')
    bias_file_path = weight_file_path.replace('.weight', '.bias')
    bias = torch_to_numpy(torch.zeros(bias_shape, dtype=data_type))
    bias.tofile(bias_file_path)


def convert_weight_to_ft_each(out_dir: str, tensor_parallelism: int,
                              tensor_name: str, config: Dict[str, Any],
                              data: np.ndarray, data_type: torch.dtype):
    """Convert an MPT checkpoint to a FasterTransformer compatible format.

    Args:
        out_dir (str): Path of the directory to save the weight in FT format. The directory must already exist.
        tensor_parallelism (int): The number of gpus you are planning to use for inference.
        tensor_name (str): Name of the weight tensor. Used in naming the output file.
        config (Dict[str, Any]): Configuration for the model. This is used in getting model specific parameters.
        data (np.ndarray): Tensor data in np.ndarray format.

    Returns:
        None: Writes to a file in `out_dir`. File name is based on the `tensor_name`
    """
    if tensor_name.find('input_layernorm.weight') != -1 or tensor_name.find('input_layernorm.bias') != -1 or \
        tensor_name.find('attention.dense.bias') != -1 or tensor_name.find('post_attention_layernorm.weight') != -1 or \
        tensor_name.find('post_attention_layernorm.bias') != -1 or tensor_name.find('mlp.dense_4h_to_h.bias') != -1 or \
        tensor_name.find('final_layernorm.weight') != -1 or tensor_name.find('final_layernorm.bias') != -1:

        save_path = os.path.join(out_dir, f'model.{tensor_name}.bin')
        data.tofile(save_path)
        if 'weight' in tensor_name and config['no_bias']:
            write_zero_bias(tensor_name, save_path, data.shape[-1], data_type)

    elif tensor_name.find('attention.dense.weight') != -1:
        assert data.shape == (
            config['d_model'],
            config['d_model']), f'unexpected dim for {tensor_name}'
        # nn.Linear weights are transposed
        data = data.T
        split_vals = np.split(data, tensor_parallelism, axis=0)
        for j in range(tensor_parallelism):
            save_path = os.path.join(out_dir, f'model.{tensor_name}.{j}.bin')
            split_vals[j].tofile(save_path)
        if config['no_bias']:
            fake_weight_path = os.path.join(out_dir, f'model.{tensor_name}.bin')
            write_zero_bias(tensor_name, fake_weight_path, data.shape[-1],
                            data_type)

    elif tensor_name.find('mlp.dense_4h_to_h.weight') != -1:
        assert data.shape == (
            config['d_model'], config['expansion_ratio'] *
            config['d_model']), f'unexpected dim for {tensor_name}'
        # nn.Linear weights are transposed
        data = data.T
        split_vals = np.split(data, tensor_parallelism, axis=0)
        for j in range(tensor_parallelism):
            save_path = os.path.join(out_dir, f'model.{tensor_name}.{j}.bin')
            split_vals[j].tofile(save_path
