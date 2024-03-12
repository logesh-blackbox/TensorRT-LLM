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

#include <assert.h>
#include <float.h>

#include "tensorrt_llm/kernels/samplingPenaltyKernels.h"

// This file contains CUDA kernels for applying various sampling penalties to logits in a language model.
// The kernels are implemented as templated functions to support both single-precision floating-point (float)
// and half-precision floating-point (half) data types.

namespace tensorrt_llm
{
namespace kernels
{

// applyTemperaturePenalty: Applies a temperature penalty to logits based on a given inverse temperature value.
// The penalty is applied element-wise to the logits, scaling them by the inverse temperature.
// If the bias parameter is not null, the bias is added to the logits before applying the penalty.
//
// Parameters:
//   logits: Pointer to the logits array.
//   bias: Pointer to the bias array. If null, no bias is added.
//   temperature_inverse: The inverse of the temperature value.
//   m: The number of samples in the logits array.
//   vocab_size: The size of the vocabulary.
//   vocab_size_padd: The size of the padded vocabulary.
//
// Returns:
//   None.
template <typename T>
__global__ void applyTemperaturePenalty(T* logits, const T* bias, const float temperature_inverse, const int m,
    const int vocab_size, const int vocab_size_padd)
{
    // Check if the data type is half-precision floating-point.
    const bool IS_FP16 = std::is_same<T, half>::value;

    // Define the maximum value for half-precision floating-point.
    const T MAX_T_VAL = (IS_FP16) ? 65504.F : FLT_MAX;

    // Loop over the logits array, applying the penalty and bias (if provided).
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < m * vocab_size_padd;
         index += blockDim.x * gridDim.x)
    {
        // Calculate the bias value.
        T bias_val = bias == nullptr ? (T) (0.0f) : bias[index % vocab_size_padd];

        // Check if the index is within the vocabulary size.
        if (index % vocab_size_padd < vocab_size)
        {
            // Apply the penalty and bias (if provided) to the logits.
            logits[index] = (logits[index] + bias_val) * (T) temperature_inverse;
        }
        else
        {
            // Set the logits to a large negative value for padding.
            logits[index] = -MAX_T_VAL;
        }
    }
}

// applyTemperaturePenalty (half2 specialization): Applies a temperature penalty to logits based on a given
// inverse temperature value. This specialization is for half2 data type.
// The penalty is applied element-wise to the logits, scaling them by the inverse temperature.
// If the bias parameter is not null, the bias is added to the logits before applying the penalty.
//
// Parameters:
//   logits: Pointer to the logits array.
//   bias: Pointer to the bias array. If null, no bias is added.
//   temperature_inverse: The inverse of the temperature value.
//   batch_size: The number of batches.
//   vocab_size: The size of the vocabulary.
//   vocab_size_padded: The size of the padded vocabulary.
//
// Returns:
//   None.
template <>
__global__ void applyTemperaturePenalty<half2>(half2* logits, const half2* bias, const float temperature_inverse,
    const int batch_size, const int vocab_size, const int vocab_size_padded)
{
    // Check if the vocabulary size is even.
    assert(vocab_size % 2 == 0);
    assert(vocab_size_padded % 2 == 0);

    // Define the mask value for padding.
    const half2 mask_val = __float2half2_rn(-65504.0f);

    // Define the inverse temperature value as a half2 data type.
    const half2 temp_inv = __float2half2_rn(temperature_inverse);

    // Calculate the half vocabulary size and padded half vocabulary size.
    const int half_vocab_size = vocab_size / 2;
    const int half_vocab_size_padded = vocab_size_padded / 2;

    // Loop over the logits array, applying the penalty and bias (if provided).
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * half_vocab_size_padded;
         index += blockDim.x * gridDim.x)
    {
        // Calculate the index within the half vocabulary size.
        int vocab_idx = index % half_vocab_size_padded;

        // Load the logit value.
        half2 logit = vocab_idx < half_vocab_size ? __ldg(&logits[index]) : mask_val;

        //
