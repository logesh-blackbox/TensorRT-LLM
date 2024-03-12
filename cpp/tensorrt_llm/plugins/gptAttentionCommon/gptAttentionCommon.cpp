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
#include "gptAttentionCommon.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/plugins/common/checkMacrosPlugin.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include <NvInferRuntimePlugin.h>
#include <algorithm>
#include <cstdint>
#include <type_traits>

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;
using tensorrt_llm::plugins::GPTAttentionPluginCreatorCommon;
using tensorrt_llm::plugins::GPTAttentionPluginCommon;

template <typename KVCacheBuffer>
struct KVCacheBufferDataType
{
};

template <>
struct KVCacheBufferDataType<KVLinearBuffer>
{
    using Type = int8_t;
};

template <>
struct KVCacheBufferDataType<KVBlockArray>
{
    using Type = int8_t;
};

template <typename T>
struct SATypeConverter
{
    using Type = T;
};

template <>
struct SATypeConverter<half>
{
    using Type = uint16_t;
};

template <typename T, typename KVCacheBuffer>
struct FusedQKVMaskedAttentionDispatchParams
{
    const T* qkv_buf;
    const T* qkv_bias;
    const T* relative_attention_bias;
    const int* cache_indir;
    T* context_buf;
    const bool* finished;
    const int* sequence_lengths;
    int max_batch_size;
    int inference_batch_size;
    int beam_width;
    int head_num;
    int kv_head_num;
    int size_per_head;
    int rotary_embedding_dim;
    float rotary_embedding_base;
    RotaryScalingType rotary_embedding_scale_type;
    float rotary_embedding_scale;
    int rotary_embedding_max_positions;
    PositionEmbeddingType position_embedding_type;
    int max_seq_len;
    const int* input_lengths;
    int step;
    float q_scaling;
    int relative_attention_bias_stride;
    const T* linear_bias_slopes;
    const int* ia3_tasks;
    const T* ia3_key_weights;
    const T* ia3_value_weights;
    const float* qkv_scale_out;
    const float* attention_out_scale;
    tc::QuantMode quant_option;
    bool multi_block_mode;
    int max_seq_len_tile;
    T* partial_out;
    float* partial_sum;
    float* partial_max;
    int* block_counter;
    const float* kv_scale_orig_quant;
    const float* kv_scale_quant_orig;
    tc::QuantMode kv_cache_quant_mode;
    int multi_processor_count;
    KVCacheBuffer kv_block_array;
    bool cross_attention = false;
    const int* memory_length_per_sample = nullptr;
    int max_distance = 0;
};

template <typename T_MMHA, typename T, typename KVCacheBuffer, bool CROSS_ATTENTION>
void fusedQKV_masked_attention_dispatch(Multihead_attention_params<T_MMHA, CROSS_ATTENTION>& params,
    const FusedQKVMaskedAttentionDispatchParams<T, KVCacheBuffer>& input_params, cudaStream_t stream)
{
    using DataType = typename SATypeConverter<T>::Type;

    // Prepare the parameters.
    memset(&params, 0, sizeof(params));

    int hidden_units = input_params.head_num * input_params.size_per_head;
    int hidden_units_kv = input_params.kv_head_num * input_params.size_per_head;
    if (input_params.qkv_bias != nullptr)
    {
        params.q_bias = reinterpret_cast<const DataType*>(input_params.qkv_bias);
        params.k_bias = reinterpret_cast<const DataType*>(input_params.qkv_bias) + hidden_units;
        params.v_bias = reinterpret_cast<const DataType*>(input_params.qkv_bias) + hidden_units + hidden_units_kv;
    }
    else
    {
        params.q_bias = nullptr;
        params.k_bias = nullptr;
        params.v_bias = nullptr;
    }

    // Set the output buffer.
    params.out = reinterpret_cast<DataType*>(input_params.context_buf);

    // Set the input buffers.
    params.q = reinterpret_cast<const DataType*>(input_params.qkv_buf
