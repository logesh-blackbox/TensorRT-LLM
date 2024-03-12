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
import time

import numpy as np

from ...logger import logger


def update_timestep_weight(src, dst):
    """
    Updates the parameters of the timestep module in the destination model using
    the parameters of the timestep module in the source model.
    """
    dst.linear_1.update_parameters(src.linear_1)
    dst.linear_2.update_parameters(src.linear_2)


def update_crossattn_downblock_2d_weight(src, dst):
    """
    Updates the parameters of the cross-attention downblock 2D module in the
    destination model using the parameters of the cross-attention downblock 2D
    module in the source model.
    """
    for index, value in enumerate(src.resnets):
        update_resnet_block_weight(value, dst.resnets[index])

    for index, value in enumerate(src.attentions):
        update_transformer_2d_model_weight(dst.attentions[index], value)

    for index, value in enumerate(src.downsamplers):
        dst.downsamplers[index].conv.update_parameters(value.conv)


def update_transformer_2d_model_weight(gm, m):
    """
    Updates the parameters of the transformer 2D model in the destination module
    using the parameters of the transformer 2D model in the source module.
    """
    gm.norm.update_parameters(m.norm)
    gm.proj_in.update_parameters(m.proj_in)
    for i in range(len(gm.transformer_blocks)):
        gm.transformer_blocks[i].attn1.to_qkv.weight.value = np.concatenate(
            (m.transformer_blocks[i].attn1.to_q.weight.detach().cpu().numpy(),
             m.transformer_blocks[i].attn1.to_k.weight.detach().cpu().numpy(),
             m.transformer_blocks[i].attn1.to_v.weight.detach().cpu().numpy()))
        gm.transformer_blocks[i].attn1.to_out.update_parameters(
            m.transformer_blocks[i].attn1.to_out[0])

        gm.transformer_blocks[i].attn2.to_q.update_parameters(
            m.transformer_blocks[i].attn2.to_q)
        gm.transformer_blocks[i].attn2.to_kv.weight.value = np.concatenate(
            (m.transformer_blocks[i].attn2.to_k.weight.detach().cpu().numpy(),
             m.transformer_blocks[i].attn2.to_v.weight.detach().cpu().numpy()))
        gm.transformer_blocks[i].attn2.to_out.update_parameters(
            m.transformer_blocks[i].attn2.to_out[0])

        gm.transformer_blocks[i].ff.proj_in.update_parameters(
            m.transformer_blocks[i].ff.net[0].proj)
        gm.transformer_blocks[i].ff.proj_out.update_parameters(
            m.transformer_blocks[i].ff.net[2])

        gm.transformer_blocks[i].norm1.update_parameters(
            m.transformer_blocks[i].norm1)
        gm.transformer_blocks[i].norm2.update_parameters(
            m.transformer_blocks[i].norm2)
        gm.transformer_blocks[i].norm3.update_parameters(
            m.transformer_blocks[i].norm3)

    gm.proj_out.update_parameters(m.proj_out)


def update_upblock_2d_weight(src, dst):
    """
    Updates the parameters of the upblock 2D module in the destination model using
    the parameters of the upblock 2D module in the source model.
    """
