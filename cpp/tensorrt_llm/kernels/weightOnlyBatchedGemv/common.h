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

#pragma once
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>

namespace tensorrt_llm
{
namespace kernels
{
// Struct representing parameters for weight-only operations
struct WeightOnlyParams
{
    const uint8_t* qweight;     // Pointer to quantized weights
    const half* scales;         // Scales for quantization
    const half* zeros;          // Zeros for quantization
    const half* in;             // Input tensor
    const half* bias;           // Bias tensor
    half* out;                  // Output tensor
    const int m;                // Height of the weight tensor
    const int n;                // Width of the weight tensor
    const int k;                // Depth of the weight tensor
    const int group_size;       // Group size for group-wise operations

    WeightOnlyParams(const uint8_t* _qweight, const half* _scales, const half* _zeros, const half* _in,
        const half* _bias, half* _out, const int _m, const int _n, const int _k, const int _group_size)
        : qweight(_qweight)
        , scales(_scales)
        , zeros(_zeros)
        , in(_in)
        , bias(_bias)
        , out(_out)
        , m(_m)
        , n(_n)
        , k(_k)
        , group_size(_group_size)
    {
    }
};

// Enum for quantization types
enum class WeightOnlyQuantType
{
    Int4b,    // 4-bit integer quantization
    Int8b     // 8-bit integer quantization
};

// Enum for weight-only operation types
enum class WeightOnlyType
{
    PerChannel,  // Per-channel quantization
    GroupWise    // Group-wise quantization
};

// Struct representing per-channel weight-only operations
struct WeightOnlyPerChannel
{
    // Function to perform per-channel weight-only operation
    __device__ void operator()(const WeightOnlyParams& params) const
    {
        const int m = params.m;
        const int n = params.n;
        const int k = params.k;
        const int group_size = params.group_size;
        const uint8_t* qweight = params.qweight;
        const half* scales = params.scales;
        const half* zeros = params.zeros;
        const half* in = params.in;
        const half* bias = params.bias;
        half* out = params.out;

        const int group_m = m / group_size;
        const int group_k = k / group_size;

        for (int g = 0; g < group_size; ++g)
        {
            for (int i = 0; i < group_m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    half sum = 0;
                    for (int l = 0; l < group_k; ++l)
                    {
                        const int idx = g * group_k * n + l * n + j;
                        const int qweight_idx = g * group_k * k + l * k + i * group_size + g;
                        const int in_idx = i * group
