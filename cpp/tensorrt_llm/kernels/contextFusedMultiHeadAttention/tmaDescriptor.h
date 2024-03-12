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

#pragma once

namespace tensorrt_llm
{
namespace kernels
{

// TMA desc type.
typedef enum
{
    TILED = 0,    // Tiled mode
    IM2COL      // Im2col mode
} cudaTmaDescType;

// TMA swizzle type.
typedef enum
{
    SWIZZLE_DISABLED,
    SWIZZLE_32B,
    SWIZZLE_64B,
    SWIZZLE_128B,
    SWIZZLE_MAX
} cudaTmaDescSwizzle;

typedef enum
{
    BARRIER64,
    BARRIER128
} cudaTmaDescBarrier;

// TMA interleave type.
typedef enum
{
    INTERLEAVE_DISABLED,
    INTERLEAVE_16B,
    INTERLEAVE_32B,
    INTERLEAVE_MAX
} cudaTmaDescInterleave;

// TMA L2 sector promotion.
typedef enum
{
    PROMOTION_DISABLED = 0,
    PROMOTION_64B,
    PROMOTION_128B,
    PROMOTION_256B
} cudaTmaDescPromotion;

// TMA data type.
typedef enum
{
    U8 = 0,
    U16,
    U32,
    S32,
    U64,
    S64,
    F16_RN,
    F32_RN,
    F32_FTZ_RN,
    F64_RN,
    BF16_RN,
    FORMAT_MAX
} cudaTmaDescFormat;

// TMA cache control.
typedef enum
{
    PREFETCH,      // Prefetch tma descriptor using global memory address
    INVALIDATE,    // Invalidate tma descriptor in l2 cache
    INVALIDATE_ALL // Invalidate tma descriptor and all elements in l2 cache line
} cudaTmaDescCacheCtrl;

// TMA OOB fill modes.
typedef enum
{
    TENSOR_ZFILL,
    TENSOR_CFILL
} cudaTmaDescOobFillMode;

// Constants for tensor size and stride.
constexpr uint64_t k_max_tensor_size = (1llu << 36);
constexpr uint64_t k_max_tensor_stride = (1llu << 36);
constexpr uint64_t k_max_block_size = 256llu;
constexpr uint64_t k_max_traversal_stride = (1llu << 3);

constexpr uint64_t k_min_tensor_size = 1llu;
constexpr uint64_t k_min_tensor_stride = 0llu;
constexpr uint64_t k_min_block_size = 1llu;
constexpr uint64_t k_min_traversal_stride = 1llu;

constexpr uint32_t k_max_cta_id = (1 << 6) - 1;

// The 512 bit of descriptor for tiled mode.
typedef struct
{
    uint64_t tensor_common0;
    uint32_t tensor_common1;

    uint32_t tensor_stride_lower[4]; //< 36b of 64b with 4B aligned
    uint32_t tensor_stride_upper;
    uint32_t tensor_size[5];         //< value -1
    uint32_t traversal_stride_box_0; //< packed 3b (-1)

    uint32_t box_size_end;
} cudaTmaDescTiled;

// The 512 bit of descritptro for im2col mode.
typedef struct
{
    uint64_t tensor_common0;
    uint32_t tensor_common1;

    uint32_t tensor_stride_lower[4];
    uint32_t tensor_stride_upper;
    uint32_t tensor_size[5];
    uint32_t traversal_stride_range_c;

    uint32_t box_corner_dhw;
    uint32_t range_ndhw;
} cudaTmaDescIm2Col;

// TMA desc size
constexpr uint32_t TMA_DESC_SIZE_IN_BYTE = 64;

// TMA desc
typedef struct alignas(64)
{
    uint64_t data[8];
} cudaTmaDesc;

////////////

// manage TMA descriptor host code.
// allocate, deallocate and manipulate tma desc in the host
// copy the tma descriptor from host code to device code
// Multiple TMA desc, one desc per batch.
// Device desc ptr should be allocated outside the class and reused
template <
    // number of dimensions.
    int NUM_DIMS>
class Multiple_tma_descriptor
{
public:
    // ctor
    Multiple_tma_descriptor(int batch_size_)
        : batch_size(batch_size_)
    {
        if (
