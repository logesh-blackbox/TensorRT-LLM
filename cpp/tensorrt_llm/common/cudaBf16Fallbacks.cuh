/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace tensorrt_llm
{
namespace common
{

#ifdef ENABLE_BF16

// Converts a pair of bfloat16 values to a pair of float values.
inline __device__ float2 bf1622float2(const __nv_bfloat162 val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float2 f_val;
    f_val.x = __low2float(val);
    f_val.y = __high2float(val);
    return f_val;
#else
    return __bfloat1622float2(val);
#endif
}

// Converts a pair of bfloat16 values to a single int16_t value.
inline __device__ int16_t bf1622int16(__nv_bfloat162 val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float2 f_val;
    f_val.x = max(min(__low2float(val), 127.f), -128.f);
    f_val.y = max(min(__high2float(val), 127.f), -128.f);

    union
    {
        int8_t int8[2];
        int16_t int16;
    };

    int8[0] = static_cast<int8_t>(static_cast<short>(f_val.x));
    int8[1] = static_cast<int8_t>(static_cast<short>(f_val.y));
    return int16;
#else
    val = __hmin2(val, make_bfloat162(127., 127.));
    val = __hmax2(val, make_bfloat162(-128., -128.));

    union
    {
        int8_t int8[2];
        int16_t int16;
    };

    int8[0] = static_cast<int8_t>(static_cast<short>(val.x));
    int8[1] = static_cast<int8_t>(static_cast<short>(val.y));
    return int16;
#endif
}

// Converts a pair of float values to a pair of bfloat16 values.
inline __device__ __nv_bfloat162 float22bf162(const float2 val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return __floats2bfloat162_rn(val.x, val.y);
#else
    return __float22bfloat162_rn(val);
#endif
}

// Converts a single bfloat16 value to a pair of bfloat16 values.
inline __device__ __nv_bfloat162 bf16bf162(const __nv_bfloat16 val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    __nv_bfloat162 val2;
    val2.x = val;
    val2.y = val;
    return val2;
#else
    return __bfloat162bfloat162(val);
#endif
}

// Performs horizontal addition on a pair of bfloat16 values.
inline __device__ __nv_bfloat162 bf16hadd2(const __nv_bfloat162 x, const __nv_bfloat162 y)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float fxl, fxh, fyl, fyh;
    fxl = __low2float(x);
    fxh = __high2float(x);
    fyl = __low2float(y);
    fyh = __high2float(y);
    return __floats2bfloat162_rn(fxl + fyl, fxh + fyh);
#else
    return __hadd2(x, y);
#endif
}

// Performs horizontal addition on a pair of bfloat16 values.
inline __device__ __nv_bfloat16 bf16hadd(const __nv_bfloat16 x, const __nv_bfloat16 y)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return __float2bfloat16(__bfloat162float(x) + __bfloat162float(y));
#else
    return __hadd(x, y);
#endif
}

// Performs horizontal subtraction on a pair of bfloat16 values.
inline __device__ __nv_bfloat162 bf16hsub2(const __nv_bfloat162 x, const __nv_bfloat162 y)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float fxl, fxh, fyl, fyh;
    fxl = __low2float(x);

