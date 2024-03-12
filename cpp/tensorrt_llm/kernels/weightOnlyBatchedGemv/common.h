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
struct WeightOnlyPerChannel;

// Struct representing group-wise weight-only operations
template <int GS>  // GS: group size
struct WeightOnlyGroupWise;

// Enum for activation types
enum class WeightOnlyActivationType
{
    Gelu
