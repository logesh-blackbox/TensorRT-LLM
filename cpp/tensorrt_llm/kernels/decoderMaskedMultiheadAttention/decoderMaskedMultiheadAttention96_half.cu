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
auto constexpr kSizePerHead = 96;
} // namespace

namespace mmha
{

template <typename T>
void launchDecoderMaskedMultiheadAttention(const T* input_query, const T* input_key, const T* input_value,
                                           const T* input_mask, const T* input_bias, const T* input_scale,
                                           const T* input_shift, const int64_t* input_query_shape,
                                           const int64_t* input_key_shape, const int64_t* input_value_shape,
                                           const int64_t* input_mask_shape, const int64_t* input_bias_shape,
                                           const int64_t* input_scale_shape, const int64_t* input_shift_shape,
                                           const int64_t* batch_size, const int64_t* num_heads,
                                           const int64_t* head_size, const int64_t* seq_len, const int64_t* k,
                                           const int64_t* v, const int64_t* q, const
