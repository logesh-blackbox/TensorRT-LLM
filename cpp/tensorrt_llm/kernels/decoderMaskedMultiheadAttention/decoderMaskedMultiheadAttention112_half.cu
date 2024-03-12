/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "decoderMaskedMultiheadAttentionLaunch.h"

namespace tensorrt_llm
{
namespace kernels
{

namespace
{
// Changed kSizePerHead to kHeadSize
constexpr int kHeadSize = 112;
} // namespace

namespace mmha
{

// Added template parameter for the data type
template <typename T>
void maskedMultiheadAttentionLaunch(const T* input_query, const T* input_key, const T* input_value,
                                   const T* input_mask, const T* input_bias, const int batch_size,
                                   const int sequence_length, const int head_count, const int head_size,
                                   const int key_size, const int value_size, const int output_size,
                                   const int max_batch_size, const int max_sequence_length,
                                   const int max_head_count, const int max_head_size, const int max_key_size,
                                   const int max_value_size, const int max_output_size, T* output_query,
                                   T* output_key, T* output_value, T* output_mask, T* output_bias,
                                   cudaStream_t stream)
{
    // ...
}

// Inst
