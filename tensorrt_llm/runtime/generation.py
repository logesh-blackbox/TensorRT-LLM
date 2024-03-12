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

import csv
import math
from dataclasses import dataclass, field
from functools import wraps
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import tensorrt as trt
import torch
from cuda import cudart

from .._ipc_utils import IpcMemory, set_peer_access
from .._utils import pad_vocab_size, str_dtype_to_torch, trt_dtype_to_torch
from ..logger import logger
from ..mapping import Mapping
from ..quantization import QuantMode
from .kv_cache_manager import GenerationSequence, KVCacheManager
from .session import _scoped_stream


def to_word_list_format(word_dict: List[List[str]], tokenizer=None):
    '''
    format of word_dict
        len(word_dict) should be same to batch_size
        word_dict[i] means the words for batch i
        len(word_dict[i]) must be 1, which means it only contains 1 string
        This string can contains several sentences and split by ",".
        For example, if word_dict[2] = " I am happy, I am sad", then this function will return
        the ids for two short sentences " I am happy" and " I am sad".
    '''
    assert tokenizer != None, "need to set tokenizer"

    flat_ids = []
    offsets = []
    for word_dict_item in word_dict:
        item_flat_ids = []
        item_offsets = []

        if isinstance(word_dict_item[0], bytes):
            word_dict_item = [word_dict_item[0].decode()]

        words = list(csv.reader(word_dict_item))[0]
        for word in words:
            ids = tokenizer.encode(word)

            if len(ids) == 0:
                continue

            item_flat_ids += ids
            item_offsets.append(len(ids))

        flat_ids.append(np.array(item_flat_ids))
        offsets.append(np.cumsum(np.array(item_offsets)))

    pad_to = max(1, max(len(ids) for ids in flat_ids))

    for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
        flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
        offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

    return np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))


def _prepare_input_ids(tensors: Sequence[torch.Tensor]):
    tensors = [torch.flatten(t) for t in tensors]
    data = torch.unsqueeze(torch.concat(tensors), 0)
    row_lengths = [t.size(0) for t in tensors]
    row_lengths = torch.tensor(row_lengths,
                               dtype=torch.int32,
                               device=data.device)
    return (data, row_lengths)


def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1:]
    return None


def _update_cuda_graph_instance(instance, graph):
    err = cudart.cudaGraphExecUpdate(instance, graph)
    if err != cudart.cudaError_t.cudaSuccess:
        # When updating cuda graph failed, destroy and instantiate one.
        CUASSERT(cudart.cudaGraphExecDestroy(instance))
        instance = CUASSERT(cudart.cudaGraphInstantiate(graph, 0))[0]
    return instance


def _prepare_attention_mask(input_ids: torch.Tensor,
                            pad_id: Optional[int] = None):
    is_pad_id_in_inputs = (pad_id is not None) and (pad_id in input_ids)
    if input_ids is not None and is_pad_id_in_inputs:
        return input_ids.ne(pad_id).int()
    else:
        return torch.ones(input_ids.shape,
                          dtype=torch.int32,
                          device=input_ids.device)


def _tile_beam_width(tensor: torch.Tensor, num_beams: int):
    new_shape = np.array(tensor.shape)
    new_shape[0] = new_shape[0] * num_beams

    tile_size = np.ones(new_shape.shape, dtype=np.int32)
    tile_size = np.insert(tile_size, 1, num_beams)

    new_tensor = torch.unsqueeze(tensor, 
