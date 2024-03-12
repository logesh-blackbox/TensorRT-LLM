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
import os
from collections import OrderedDict

import tensorrt as trt
import torch
from transformers import BertConfig, BertForQuestionAnswering, BertModel

import tensorrt_llm
from tensorrt_llm.builder import Builder
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType

from weight import load_from_hf_bert, load_from_hf_qa_bert  # isort:skip


def get_engine_name(model, dtype, tp_size, rank):
    return f"{model}_{dtype}_tp{tp_size}_rank{rank}.engine"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--timing_cache", type=str, default="model.cache")
    parser.add_argument("--log_level", type=str, default="info")
    parser.add_argument("--vocab_size", type=int, default=51200)
    parser.add_argument("--n_labels", type=int, default=2)
    parser.add_argument("--n_layer", type=int, default=24)
    parser.add_argument("--n_positions", type=int, default=1024)
    parser.add_argument("--n_embd", type=int, default=1024)
    parser.add_argument("--n_head", type=int, default=16)
    parser.add_argument("--hidden_act", type=str, default="gelu")
    parser.add_argument("--max_batch_size", type=int, default=256)
    parser.add_argument("--max_input_len", type=int, default=512)
    parser.add_argument("--gpus_per_node", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="bert_outputs")
    parser.add_argument(
        "--use_bert_attention_plugin",
        nargs="?",
        const="float16",
        type=str,
        default=False,
        choices=["float16", "float32"],
    )
    parser.add_argument(
        "--use_gemm_plugin",
        nargs="?",
        const="float16",
        type=str,
        default=False,
        choices=["float16", "float32"],
    )
    parser.add_argument(
        "--use_layernorm_plugin",
        nargs="?",
        const="float16",
        type=str,
        default=False,
        choices=["float16", "float32"],
    )
    parser.add_argument(
        "--enable_qk_half_accum", default=False, action="store_true"
    )
    parser.add_argument(
        "--enable_context_fmha", default=False, action="store_true"
    )
    parser.add_argument(
        "--enable_context_fmha_fp32_acc", default=False, action="store_true"
    )
    parser.add_argument(
        "--model",
        default=tensorrt_llm.models.BertModel.__name__,
        choices=[
            tensorrt_llm.models.BertModel.__name__,
            tensorrt_llm.models.BertForQuestionAnswering.__name__
        ],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    tensorrt_llm.logger.set_level(args.log_level)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    bs_range = [1, (args.max_batch_size + 1) // 2, args.max_batch_size]
    inlen_range = [1, (args.max_input_len + 1) // 2, args.max_input_len]
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    trt_dtype = trt.float16 if args.dtype == "float16" else trt.float32

    builder = Builder()
    builder_config = builder.create_builder_config(
        name=args.model,
        precision=args.dtype,
        timing_cache=args.timing_cache,
        tensor_parallel=args.world_size,  # TP only
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
    )
    # Initialize model

    bert_config = BertConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.n_embd,
        num_hidden_layers=args.n_layer
