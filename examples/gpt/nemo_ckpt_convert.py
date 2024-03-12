#! /usr/bin/env python3
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
import configparser
import datetime
import logging
import multiprocessing
import shutil
import sys
import tempfile
import typing
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer, T5Tokenizer
from utils.convert import (cpu_map_location, gpu_map_location,
                           split_and_save_weight)
from utils.nemo import (UnpackedNemoCheckpointDir, extract_layers_with_prefix,
                        nemo_to_gpt_config, unpack_nemo_ckpt)

from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy

LOGGER = logging.getLogger(__name__)


def rename_key(old_key: str, pp_rank: int, num_layers: int, pp_size: int):
    new_key = old_key

    if "layers." in old_key:
        split_key = old_key.split(".")
        split_key[1] = str(int(split_key[1]) + pp_rank * num_layers // pp_size)
        new_key = ".".join(split_key)

        if "self_attention" in new_key:
            new_key = new_key.replace("self_attention", "attention")
    return new_key


@torch.no_grad()
def convert_checkpoint(unpacked_checkpoints_dir: UnpackedNemoCheckpointDir,
                       args):
    nemo_model_config = unpacked_checkpoints_dir.model_config

    checkpoints_paths = unpacked_checkpoints_dir.get_checkpoints_paths(
        nemo_model_config.get("tensor_model_parallel_size", 1),
        nemo_model_config.get("pipeline_model_parallel_size", 1),
    )

    # if checkpoints files could be found - start preparing output dir
    out_dir = create_out_dir(args)

    map_location_fn = gpu_map_location if args.load_checkpoints_on_gpu else cpu_map_location
    storage_type = str_dtype_to_torch(args.storage_type)

    # load position_embedding from rank 0
    model_00 = torch.load(checkpoints_paths[0][0], map_location=map_location_fn)
    model_00 = model_00.get("state_dict", model_00)

    has_position_embedding = "model.language_model.embedding.position_embeddings.weight" in model_00
    has_lm_head = "model.language_model.output_layer.weight" in model_00

    num_layers = nemo_model_config["num_layers"]
    training_tp_size = nemo_model_config.get("tensor_model_parallel_size", 1)
    training_pp_size = nemo_model_config.get("pipeline_model_parallel_size", 1)
    inference_tp_size = args.tensor_parallelism

    export_config = {
        "apply_layernorm_1p":
        nemo_model_config.get('normalization', '') == "layernorm1p",
        "tp_size":
        training_tp_size,
        "split_gated_activation":
        "swiglu" in nemo_model_config.get('activation', "gelu"),
        "num_attention_heads":
        nemo_model_config["num_attention_heads"],
        "use_attention_nemo_shape":
        True,
        "transpose_weights":
        True,
    }

    # merge_factor: how many TP training nodes are merged into an inference TP node
    # split_factor: in how many parts a TP training node is split
    gcd = np.gcd(training_tp_size, inference_tp_size)
    merge_factor = training_tp_size // gcd
    split_factor = inference_tp_size // gcd

    model_level_weights = defaultdict(list)

    def handle_model_level_weights(model, tp_idx: int, pp_idx: int):
        if tp_idx == 0 and pp_idx == 0:
            if has_position_embedding:
                val = model[
                    "model.language_model.embedding.position_embeddings.weight"]
                # not weight, do not need to transpose
                val = torch_to_numpy(val.to(storage_type).cpu())
                val.tofile(out_dir / "model.wpe.bin")
                model_level_weights["model.wpe.bin"].append(val)
        if pp_idx == 0:
            val = model.get(
                "state_dict",
                model)["model.language_model.embedding.word_embeddings.weight"]
            val = torch_to_numpy(val.to(storage_type).cpu())
            model_level_weights["model.wte.bin"].append(val)
        if has_lm_head and pp_idx == training_pp_size - 1:
