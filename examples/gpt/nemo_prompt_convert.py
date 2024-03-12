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
import datetime
import logging
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from utils.convert import cpu_map_location
from utils.nemo import unpack_nemo_ckpt

from tensorrt_llm._utils import torch_to_numpy

# Set up the logging format and basic configuration
log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
logging.basicConfig(format=log_format)
LOGGER = logging.getLogger(__name__)


def prompt_convert(args, prompt_config, prompt_weights):
    # Extract the prompt templates from the configuration
    prompt_templates = prompt_config["task_templates"]

    # Initialize variables to store the actual task ID, task embeddings, and
    # lengths
    actual_task_id = 0
    vtokens_embeddings = []
    vtokens_len = []

    # Iterate through the prompt templates
    for task_name_id, prompt_task in enumerate(prompt_templates):
        # Get the task name and weights
        prompt_task_name = prompt_task["taskname"]
        LOGGER.info(f"Task {actual_task_id}: {prompt_task['taskname']}")

        # Get the prompt task weights
        prompt_task_weights = prompt_weights["prompt_table"].get(
            f"prompt_table.{prompt_task_name}.prompt_embeddings.weight")

        # Check if the weights exist and append them to the list
        if prompt_task_weights is not None:
            vtokens_embeddings.append(prompt_task_weights)
            vtokens_len.append(prompt_task_weights.shape[0])
            actual_task_id += 1

    # Find the maximum length of the task embeddings
    max_vtoken_len = max(vtokens_len)

    # Get the embedding dimension
    embedding_dim = vtokens_embeddings[0].shape[1]

    # Pad the task embeddings to the maximum length
    for i, vtoken_emb_table in enumerate(vtokens_embeddings):
        padded_table = torch.zeros((max_vtoken_len, embedding_dim))
        padded_table[:vtoken_emb_table.shape[0], :] = vtoken_emb_table
        vtokens_embeddings[i] = padded_table

    # Stack the padded task embeddings
    vtokens_embeddings = torch.stack(vtokens_embeddings)

    # Save the stacked task embeddings as a .npy file
    np.save(args.out_file, torch_to_numpy(vtokens_embeddings))


def main(args):
    # Record the start time
    start_time = datetime.datetime.now()

    # Create a temporary directory for unpacking the NeMo prompt archive
    with tempfile.TemporaryDirectory() as prompt_out_dir:
        prompt_out_dir = Path(prompt_out_dir)

        # Unpack the NeMo prompt archive
        unpack_nemo_ckpt(args.in_file, prompt_out_dir)
        LOGGER.info("Spent %s (h:m:s) to unpack NeMo prompt archive",
                    datetime.datetime.now() - start_time)

        # Load the model configuration and weights
        model_weights_ckpt = "model_weights.ckpt"
        with open(prompt_out_dir / "model_config.yaml") as f:
            prompt_config = yaml.full_load(f)
        LOGGER.debug(prompt_config)

        start_time = datetime.datetime.now()

        # Load the prompt weights from the checkpoint file
        weight_path = prompt_out_dir / model_weights_ckpt
        if not weight_path.exists():
            weight_path = prompt_out_dir / "mp_rank_00" / model_weights_ckpt

        prompt_weights = torch.load(
            weight_path,
            map_location=cpu_map_location,
        )

        # Convert the prompt model
        prompt_convert(args, prompt_config, prompt_weights)

        LOGGER.info("Spent %s (h:m:s) to convert the prompt model",
                    datetime.datetime.now() - start_time)


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument(
        '--out-file',
        '-o',
        type=Path,
        help='path to output embedding table file in the .npy format',
        required=True)
    parser.add_argument('--in-file',
                        '-i',
                        type=Path,
                        help='path to input prompt-tuning checkpoint file',
                        required=True)
    parser.add_argument("--storage-type",
                        "-t",
                        type=str,
                        default="fp32",
                        choices=["fp32", "fp16"])
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Provide verbose messages
