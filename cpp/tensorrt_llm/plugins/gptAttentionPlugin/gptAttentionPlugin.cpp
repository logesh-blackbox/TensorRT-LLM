/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gptAttentionPlugin.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/plugins/common/checkMacrosPlugin.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include "tensorrt_llm/plugins/gptAttentionCommon/gptAttentionCommon.h"
#include "tensorrt_llm/plugins/gptAttentionCommon/gptAttentionCommonImpl.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using tensorrt_llm::plugins::GPTAttentionPluginCreator;
using tensorrt_llm::plugins::GPTAttentionPlugin;

// GPTAttentionPlugin class definition
// This class implements a plugin for the NVIDIA TensorRT framework that provides a GPT-style attention mechanism.
// It is a custom implementation of the MultiHeadAttentionPlugin class provided by TensorRT.
class GPTAttentionPlugin : public MultiHeadAttentionPlugin
{
public:
    // Constructor for GPTAttentionPlugin
    // This constructor initializes the GPTAttentionPlugin object with the specified parameters.
    GPTAttentionPlugin(int num_heads, int num_kv_heads, int head_size, int unidirectional,
        float q_scaling, PositionEmbeddingType position_embedding_type, int rotary_embedding_dim,
        float rotary_embedding_base, RotaryScalingType rotary_embedding_scale_type, float rotary_embedding_scale,
        int rotary_embedding_max_positions, int tp_size, int tp_rank, ContextFMHAType context_fmha_type,
        bool multi_block_mode, int kv_cache_quant_mode, bool remove_input_padding, AttentionMaskType mask_type,
        bool paged_kv_cache, int tokens_per_block, DataType type, int32_t max_context_length, bool qkv_bias_enabled,
        bool cross_attention, int max_distance);

    // Constructor for GPTAttentionPlugin (from serialized data)
    // This constructor initializes the GPTAttentionPlugin object from serialized data.
    GPTAttentionPlugin(const void* data, size_t length);

    // Destructor for GPTAttentionPlugin
    // This destructor frees any resources allocated by the GPTAttentionPlugin object.
    ~GPTAttentionPlugin();

    // Clone method for GPTAttentionPlugin
    // This method creates a deep copy of the GPTAttentionPlugin object.
    GPTAttentionPlugin* clone() const noexcept override;

    // GetOutputDimensions method for GPTAttentionPlugin
    // This method returns the output dimensions for the specified output index.
    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    // SupportsFormatCombination method for GPTAttentionPlugin
    // This method checks whether the specified format combination is supported.
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;

    // ConfigurePlugin method for GPTAttentionPlugin
    // This method configures the plugin with the specified input and output tensor descriptions.
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;

    // GetWorkspaceSize method for GPTAttentionPlugin
    // This method returns the size of the workspace required by the plugin.
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;

    // Enqueue method for GPTAttentionPlugin
    // This method executes the attention mechanism on the specified input tensors and produces the output tensors.
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;

    // GetOutputDataType method for GPTAttentionPlugin
    // This method returns the data type of the specified output.
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

private:
    int num_heads_;
    int num_kv_heads_;
    int head_size_;
    int unidirectional_;
    float q_scaling_;
    PositionEmbeddingType position_embedding_type_;
    int rotary_embedding_dim_;
    float rotary_embedding_base_;
   
