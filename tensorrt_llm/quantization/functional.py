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
from typing import Optional, Tuple, Union

import numpy as np
import tensorrt as trt

from .._common import default_net, default_trtnet
from .._utils import str_dtype_to_np, str_dtype_to_trt
from ..functional import Tensor, _create_tensor, cast, clip, constant, round
from ..plugin import TRT_LLM_PLUGIN_NAMESPACE


def smooth_quant_gemm(input: Tensor, weights: Tensor, scales_a: Tensor,
                      scales_b: Tensor, per_token_scaling: bool,
                      per_channel_scaling: bool) -> Tensor:
    if not default_net().plugin_config.smooth_quant_gemm_plugin:
        raise TypeError("Smooth Quant GEMM is only supported with plugin")
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'SmoothQuantGemm', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None

        per_channel_scaling = 1 if per_channel_scaling else 0
        per_channel_scaling = trt.PluginField(
            "has_per_channel_scaling",
            np.array(per_channel_scaling, dtype=np.int32),
            trt.PluginFieldType.INT32)

        per_token_scaling = 1 if per_token_scaling else 0
        per_token_scaling = trt.PluginField(
            "has_per_token_scaling", np.array(per_token_scaling,
                                              dtype=np.int32),
            trt.PluginFieldType.INT32)

        p_dtype = default_net().plugin_config.smooth_quant_gemm_plugin
        pf_type = trt.PluginField(
            "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
            trt.PluginFieldType.INT32)

        pfc = trt.PluginFieldCollection(
            [per_channel_scaling, per_token_scaling, pf_type])
        gemm_plug = plg_creator.create_plugin("sq_gemm", pfc)
        plug_inputs = [
            input.trt_tensor, weights.trt_tensor, scales_a.trt_tensor,
            scales_b.trt_tensor
        ]
        layer = default_trtnet().add_plugin_v2(plug_inputs, gemm_plug)
        layer.get_input(0).set_dynamic_range(-127, 127)
        return _create_tensor(layer.get_output(0), layer)


def weight_only_quant_matmul(input: Tensor, weights: Tensor, scales: Tensor,
                             weightTypeId: int) -> Tensor:
    if not default_net().plugin_config.weight_only_quant_matmul_plugin:
        raise TypeError(
            "Weight Only Qunat MatMul is only supported with plugin")
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'WeightOnlyQuantMatmul', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None

        weight_type_id = trt.PluginField("weight_type_id",
                                         np.array(weightTypeId, dtype=np.int32),
                                         trt.PluginFieldType.INT32)

        p_dtype = default_net().plugin_config.weight_only_quant_matmul_plugin
        pf_type = trt.PluginField(
            "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
            trt.PluginFieldType.INT32)

        pfc = trt.PluginFieldCollection([pf_type, weight_type_id])
        matmul_plug = plg_creator.create_plugin("woq_matmul", pfc)
        plug_inputs = [input.trt_tensor, weights.trt_tensor, scales.trt_tensor]
        layer = default_trtnet().add_plugin_v2(plug_inputs, matmul_plug)
        return _create_tensor(layer.get_output(0), layer)


def weight_only_groupwise_quant_matmul(input: Tensor, pre_quant_scale: Tensor,
                                       weights: Tensor, scales: Tensor,
                                       zeros: Tensor, biases: Tensor,
                                       quant_algo: int,
                                       group_size: int) -> Tensor:
    if not default_net(
    ).plugin_config.weight_only_groupwise_quant_matmul_plugin:
        raise TypeError(
            "Weight Only Groupwise Quant MatMul is only supported with plugin")
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'WeightOnlyGroupwiseQuantMatmul', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None

        quant_algo_ = trt.PluginField("quant
