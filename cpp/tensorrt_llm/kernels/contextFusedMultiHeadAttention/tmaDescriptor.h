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

#include <cstdint>

namespace tensorrt_llm
{
namespace kernels
{

// TMA desc type.
enum class cudaTmaDescType
{
    TILED = 0,    // Tiled mode
    IM2COL      // Im2col mode
};

// TMA swizzle type.
enum class cudaTmaDescSwizzle
{
    SWIZZLE_DISABLED,
    SWIZZLE_32B,
    SWIZZLE_64B,
    SWIZZLE_128B,
    SWIZZLE_MAX
};

enum class cudaTmaDescBarrier
{
    BARRIER64,
    BARRIER128
};

// TMA interleave type.
enum class cudaTmaDescInterleave
{
    INTERLEAVE_DISABLED,
    INTERLEAVE_16B,
    INTERLEAVE_32B,
    INTERLEAVE_MAX
};

// TMA L2 sector promotion.
enum class cudaTmaDescPromotion
{
    PROMOTION_DISABLED = 0,
    PROMOTION_64B,
    PROMOTION_128B,
    PROMOTION_256B
};

// TMA data type.
enum class cudaTmaDescFormat
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
};

// TMA cache control.
enum class cudaTmaDescCacheCtrl
{
    PREFETCH,      // Prefetch tma descriptor using global memory address
    INVALIDATE,    // Invalidate tma descriptor in l2 cache
    INVALIDATE_ALL // Invalidate tma descriptor and all elements in l2 cache line
};

// TMA OOB fill modes.
enum class cudaTmaDescOobFillMode
{
    TENSOR_ZFILL,
    TENSOR_CFILL
};

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
struct cudaTmaDescTiled
{
    uint64_t tensor_common0;
    uint32_t tensor_common1;

    uint32_t tensor_stride_lower[4]; //< 36b of 64b with 4B aligned
    uint32_t tensor_stride_upper;
    uint32_t tensor_size[5];         //< value -1
    uint32_t traversal_stride_box_0; //< packed 3b (-1)

    uint32_t box_size_end;
};

// The 512 bit of descritptro for im2col mode.
struct cudaTmaDescIm2Col
{
    uint64_t tensor_common0;
    uint32_t tensor_common1;

    uint32_t tensor_stride_lower[4];
    uint32_t tensor_stride_upper;
    uint32_t tensor_size[5];
    uint32_t traversal_stride_range_c;

    uint32_t box_corner_dhw;
    uint32_t range_ndhw;
};

// TMA desc size
constexpr uint32_t TMA_DESC_SIZE_IN_BYTE = 64;

// TMA desc
union cudaTmaDesc
{
    uint64_t data[8];
    cudaTmaDescTiled tiled;
    cudaTmaDescIm2Col im2col;
};

template <int NUM_DIMS>
class Multiple_tma_descriptor
{
public:
    // ctor
    Multiple_tma_descriptor(int batch_size)
        : batch_size(batch_size), desc_count(batch_size), descs(nullptr)
    {
        descs = new cudaTmaDesc[batch_size];
    }

    ~Multiple_tma_descriptor()
    {
        delete[] descs;
    }

    cudaTmaDesc& get_desc(int i)
    {
        return descs
