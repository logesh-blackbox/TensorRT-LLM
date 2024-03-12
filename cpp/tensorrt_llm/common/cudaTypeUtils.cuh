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

#include "tensorrt_llm/common/cudaBf16Fallbacks.cuh"
#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#if ENABLE_BF16
#include <cuda_bf16.h>
#endif

namespace tensorrt_llm
{
namespace common
{

template <typename T>
inline __device__ T ldg(const T* val)
{
    return __ldg(val);
}

#if ENABLE_BF16
template <>
inline __device__ __nv_bfloat162 ldg(const __nv_bfloat162* val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return val[0];
#else
    return __ldg(val);
#endif
}

template <>
inline __device__ __nv_bfloat16 ldg(const __nv_bfloat16* val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return val[0];
#else
    return __ldg(val);
#endif
}
#endif // ENABLE_BF16

// Get type2 from type or vice versa (applied to half and bfloat16)
template <typename T>
struct TypeConverter
{
    using Type = half2;
}; // keep for generality

template <>
struct TypeConverter<half2>
{
    using Type = half;
};

template <>
struct TypeConverter<half>
{
    using Type = half2;
};

#if ENABLE_BF16
template <>
struct TypeConverter<__nv_bfloat162>
{
    using Type = __nv_bfloat16;
};

template <>
struct TypeConverter<__nv_bfloat16>
{
    using Type = __nv_bfloat162;
};
#endif // ENABLE_BF16

// Defined math operations (bfloat16 fallback to fp32 when it is not supported)
template <typename T>
inline __device__ T hadd2(T a, T b)
{
    return __hadd2(a, b);
}

#if ENABLE_BF16
template <>
inline __device__ __nv_bfloat162 hadd2(__nv_bfloat162 a, __nv_bfloat162 b)
{
    return bf16hadd2(a, b);
}
#endif // ENABLE_BF16

template <typename T>
inline __device__ T add(T a, T b)
{
    return a + b;
}

template <>
inline __device__ half2 add(half2 a, half2 b)
{
    return __hadd2(a, b);
}

template <>
inline __device__ half add(half a, half b)
{
    return __hadd(a, b);
}

#if ENABLE_BF16
template <>
inline __device__ __nv_bfloat162 add(__nv_bfloat162 a, __nv_bfloat162 b)
{
    return bf16hadd2(a, b);
}

template <>
inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b)
{
    return bf16hadd(a, b);
}

inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, float b)
{
    return bf16hadd(a, __float2bfloat16(b));
}
#endif // ENABLE_BF16

// applies to all 4 values addition
template <typename T>
inline __device__ T add(T a, T b, T c)
{
    return a + b + c;
}

#if ENABLE_BF16
inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c)
{
    return bf16hadd(a, b, c);
}

inline __device__ __nv_bfloat162 add(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c)
{
    return bf16hadd2(a, b, c);
}
#endif // ENABLE_BF16

// applies to all 4 values addition
template <typename T>
inline __device__ T add(T a, T b, T c, T d)
{
    return (T) ((float) a + (float) b + (float) c + (float) d);
}

#if ENABLE_BF16
inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c, __nv_bfloat16 d)
{
    return bf16hadd(a, b, c, d);
}
#endif // ENABLE_BF16

template <typename
