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
import math
from collections import OrderedDict

import tensorrt as trt

from ..functional import Tensor
from ..mapping import Mapping


class GenerationMixin:

    def get_transformer_layers(self, mapping, num_layers):
        layers_per_pipeline_stage = num_layers // mapping.pp_size
        layers_range = list(
            range(mapping.pp_rank * layers_per_pipeline_stage,
                  (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))
        return layers_range

    def prepare_basic_inputs(self,
                             max_batch_size,
                             max_beam_width,
                             max_input_len,
                             max_new_tokens,
                             num_kv_heads,
                             head_size,
                             num_layers,
                             kv_dtype,
                             remove_input_padding=False,
                             use_gpt_attention_plugin=False,
                             use_gemm_plugin=False,
                             use_custom_all_reduce=False,
                             paged_kv_cache=False,
                             tokens_per_block=64,
                             gather_all_token_logits=False,
                             dtype=None,
                             num_heads=None,
                             mapping=Mapping(),
                             max_num_tokens=None):

        max_len = max_input_len + max_new_tokens

        bb_range_cxt = [1, (max_batch_size + 1) // 2, max_batch_size]
        bb_range_gen = [
            1, (max_batch_size * max_beam_width + 1) // 2,
            max_batch_size * max_beam_width
        ]
        _bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        _beam_width_range = [1, (max_beam_width + 1) // 2, max_beam_width]
        inlen_range_cxt = [1, (max_input_len + 1) // 2, max_input_len]
        inlen_range_gen = [1, 1, 1]
        _mask_len_ctx = [1, (max_input_len + 1) // 2, max_input_len]
        _mask_len_gen = [2, (max_len + 1) // 2 + 1, max_len + 1]
        _kv_cache_range_ctx = [0, 0, 0]
        _kv_cache_range_gen = [1, (max_len + 1) // 2, max_len]
        _max_len_range = [0, (max_len + 1) // 2, max_len]

        if max_num_tokens is None:
            num_tokens_range_ctx = [
                1, (max_input_len * max_batch_size + 1) // 2,
                max_input_len * max_batch_size
            ]
            num_tokens_range_gen = [
                1, max_batch_size * max_beam_width,
                max_beam_width * max_batch_size
            ]
        else:
            num_tokens_range_ctx = [[
                1, (max_num_tokens + 1) // 2, max_num_tokens
            ]]
            num_tokens_range_gen = [[
                1, (max_num_tokens + 1) // 2, max_num_tokens
            ]]

        enable_two_optimization_profiles = False
        if use_gpt_attention_plugin == False or use_gemm_plugin == False:
            use_in_flight_batching = use_gpt_attention_plugin and remove_input_padding and paged_kv_cache
            enable_two_optimization_profiles = not use_in_flight_batching
        if enable_two_optimization_profiles:
            bb_range = [bb_range_cxt, bb_range_gen]
            bs_range = [_bs_range, _bs_range]
            beam_width_range = [_beam_width_range, _beam_width_range]
            inlen_range = [inlen_range_cxt, inlen_range_gen]
            mask_len_range = [_mask_len_ctx, _mask_len_gen]
            if use_gpt_attention_plugin:
                kv_cache_range = [_kv_cache_range_gen, _kv_cache_range_gen]
            else:
                kv_cache_range = [_kv_cache_range_ctx, _kv_cache_range_gen]
            max_len_range = [_max_len_range, _max_len_range]
            num_tokens_range = [num_tokens_range_ctx, num_tokens_range_gen]
        else:
            bb_range = [bb_range_gen]
            bs_range = [_bs_range]
            beam_width_range = [_beam_width_range]
            inlen_range = [[1, 1, max_input_len]]
            mask_len_range = [[1,
