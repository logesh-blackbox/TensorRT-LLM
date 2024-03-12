/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include <algorithm>
#include <cstdio>
#include <cuda_fp16.h>
#include <limits>
#include <type_traits>

namespace tensorrt_llm
{
namespace common
{
#ifdef ENABLE_FP8

constexpr int CTA_SIZE = 256;

template <bool QUANTIZE>
__inline__ __device__ float scale(float a, float b)
{
    return QUANTIZE ? a / b : a * b;
}

// This template function scales a matrix by a given factor.
// The scaling factor can be per-channel, per-token, or per-tensor.
template <QuantizeMode QUANTIZE_MODE, bool QUANTIZE, typename T_OUT, typename T_S, typename T_IN>
__global__ void scaleMatrix(T_OUT* output, T_S const* input_scale, T_IN const* input, int64_t numel, int64_t lda)
{
    for (int64_t i = threadIdx.x + blockIdx.x * blockDim.x; i < numel; i += blockDim.x * gridDim.x)
    {
        if (QUANTIZE_MODE == QuantizeMode::PER_CHANNEL)
        {
            output[i] = T_OUT(scale<QUANTIZE>(static_cast<float>(input[i]), static_cast<float>(input_scale[i % lda])));
        }
        else if (QUANTIZE_MODE == QuantizeMode::PER_TOKEN)
        {
            output[i] = T_OUT(scale<QUANTIZE>(static_cast<float>(input[i]), static_cast<float>(input_scale[i / lda])));
        }
        else if (QUANTIZE_MODE == QuantizeMode::PER_TENSOR)
        {
            output[i] = T_OUT(scale<QUANTIZE>(static_cast<float>(input[i]), static_cast<float>(input_scale[0])));
        }
    }
}

// This template function invokes the scaleMatrix kernel for a given matrix.
template <typename T_OUT, typename T_S, typename T_IN>
void invokeQuantizeMatrix(T_OUT* output, T_S const* input_scale, T_IN const* input, int64_t numel, int64_t lda,
    QuantizeMode quantize_mode, cudaStream_t stream)
{
    dim3 grid(1024);
    dim3 block(CTA_SIZE);
    if (quantize_mode == QuantizeMode::PER_CHANNEL)
    {
        scaleMatrix<QuantizeMode::PER_CHANNEL, true>
            <<<grid, block, 0, stream>>>(output, input_scale, input, numel, lda);
    }
    else if (quantize_mode == QuantizeMode::PER_TOKEN)
    {
        scaleMatrix<QuantizeMode::PER_TOKEN, true><<<grid, block, 0, stream>>>(output, input_scale, input, numel, lda);
    }
    else if (quantize_mode == QuantizeMode::PER_TENSOR)
    {
        scaleMatrix<QuantizeMode::PER_TENSOR, true><<<grid, block, 0, stream>>>(output, input_scale, input, numel, lda);
    }
    sync_check_cuda_error();
}

// This template function invokes the scaleMatrix kernel for a given matrix.
template <typename T_OUT, typename T_S, typename T_IN>
void invokeDequantizeMatrix(T_OUT* output, T_S const* input_scale, T_IN const* input, int64_t numel, int64_t lda,
    QuantizeMode quantize_mode, cudaStream_t stream)
{
    dim3 grid(1024);
    dim3 block(CTA_SIZE);
    if (quantize_mode == QuantizeMode::PER_CHANNEL)
    {
        scaleMatrix<QuantizeMode::PER_CHANNEL, false>
            <<<grid, block, 0, stream>>>(output, input_scale, input, numel, lda);
    }
    else if (quantize_mode == QuantizeMode::PER_TOKEN)
    {
        scaleMatrix<QuantizeMode::PER_TOKEN, false><<<grid, block, 0, stream>>>(output, input_scale, input, numel, lda);
    }
    else if (quantize_mode == QuantizeMode::PER_TENSOR)
    {
        scaleMatrix<QuantizeMode::PER_TENSOR, false>
            <<<grid, block, 0, stream>>>(output, input_scale, input, numel, lda);
    }
    sync_check_cuda_error();
}

// This template function performs a fake quantization of a matrix.
template <typename T_FAKE, typename T_OUT, typename T_IN>
__global__ void fakeQuantize(T_OUT* dst, const T_IN* src, const int64_t numel)
{
    for (int64_
