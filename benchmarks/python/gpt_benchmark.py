# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
from math import ceil

import torch
from base_benchmark import BaseBenchmark
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.layers import PositionEmbeddingType
from tensorrt_llm.models import (fp8_quantize, smooth_quantize,
                                 weight_only_quantize)
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode

class GPTBenchmark(BaseBenchmark):
    """Benchmark for GPT models."""

    def __init__(self,
                 engine_dir: str,
                 model_name: str,
                 mode: str,
                 batch_sizes: tuple[int],
                 in_out_lens: tuple[tuple[int, int]],
                 dtype: str,
                 refit: bool,
                 num_beams: int,
                 top_k: int,
                 top_p: float,
                 output_dir: str,
                 n_positions: int = None,
                 max_input_len: int = None,
                 max_output_len: int = None,
                 max_batch_size: int = None,
                 enable_custom_all_reduce: bool = None,
                 **kwargs):
        """
        Initialize the GPTBenchmark object.

        Args:
            engine_dir: The directory where the engine file is stored.
            model_name: The name of the GPT model.
            mode: The mode to use for the benchmark. Can be either 'plugin' or 'ootb'.
            batch_sizes: A tuple of batch sizes to use for the benchmark.
            in_out_lens: A tuple of input and output lengths to use for the benchmark.
            dtype: The data type to use for the benchmark.
            refit: Whether to refit the engine.
            num_beams: The number of beams to use for the benchmark.
            top_k: The value of top_k to use for the benchmark.
            top_p: The value of top_p to use for the benchmark.
            output_dir: The directory where the benchmark results will be stored.
            n_positions: The number of positions.
            max_input_len: The maximum input length.
            max_output_len: The maximum output length.
            max_batch_size: The maximum batch size.
            enable_custom_all_reduce: Whether to enable custom all-reduce.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(engine_dir, model_name, dtype, output_dir)
        self.batch_sizes = batch_sizes
        self.in_out_lens = in_out_lens
        self.refit = refit
        self.num_beams = num_beams
        self.build_time = 0
        self.mode = mode
        self.fuse_bias = True

        self.cuda_graph_mode = kwargs.get('enable_cuda_graph', False)
        self.enable_custom_all_reduce = enable_custom_all_reduce

        if engine_dir is not None:
            # Get build configs from engine directory is done in base class
            # Deserialize engine from engine directory
            self.serialize_path = os.path.join(engine_dir, self.engine_name)
            with open(self.serialize_path, 'rb') as f:
                engine_buffer = f.read()
        else:
            # Build engine
            self.world_size = tensorrt_llm.mpi_world_size()
            self.apply_query_key_layer_scaling = False
            self.use_smooth_quant = False
            self.enable_fp8 = kwargs.get('enable_fp8', False)
            self.fp8_kv_cache = kwargs.get('fp8_kv_cache', False)

            self.use_weight_only = False
            self.weight_only_precision = 'int8'
            self.per_token = False
            self.per_channel = False

            is_plugin_mode = mode == 'plugin'
            plg_dtype = dtype if is_plugin_mode else False
            self.use_gpt_attention_plugin = plg_dtype
            self.use_gemm_plugin = plg_dtype
            # Starting TRT9.1 O
