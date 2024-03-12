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
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/common.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/utility.h"

namespace tensorrt_llm
{
namespace kernels
{
// Define a template struct for storing details about the weight layout
template <WeightOnlyQuantType QType>
struct WeightLayoutDetails
{
    // Every four rows of the original weights are interleaved into a row with stride of 64, so if each thread
    // processes 32 elements(for int4, we can use ldg.128 to load weights), then every group of two adjacent threads
    // will alternately process four different row weights
    // for example
    // every 256 consecutive int4 elements [256*i, 256*(i+1)-1] of row N under interleave layout,
    // the first 64 are from [64*i, 64*(i+1)-1] of row 4N before interleaving,
    // and the second 64 are from [64*i, 64*(i+1)-1] of row 4N+1 before interleaving, and so on.
    // So if each thread loads 32 int4 elements, then the elements of each 2 adjacent threads of each 8
    // consecutive threads will come from row 4N ~ 4N+3 respectively before interleaving.
    static constexpr int kElemBits = 4;
    static constexpr int kInterleave = 4;
    static constexpr int kStride = 64;

    // The index remapping here is to counteracts the effect of cutlass::permute_B_rows_for_mixed_gemm
    // input 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 ... 31
    // weight 0 1 8 9 16 17 24 25 2 3 10 11 18 19 26 27 4 5 12 13 20 21 28 29 6 7 14 15 22 23 30 31
    static constexpr int kShuffleSize = 32;
    static constexpr int kShuffleBasicTile = 2;
    static constexpr int kShuffleContinous = 4;
    static constexpr int kShuffleStrided = 4;

    // The rearrangement here counteracts the effect of cutlass::add_bias_and_interleave_int4s_inplace
    // Input int8 data layout
    //      [elt_7  elt_5  elt_3  elt_1  elt_6  elt_4  elt_2  elt_0] (each elt occupies 4 bits)
    //
    // Converted fp16 data layout
    //      [elt_7  elt_6  elt_5  elt_4  elt_3  elt_2  elt_1  elt_0] (each elt occupies 16 bits)
    static constexpr int kConvertCount = 8;
    using Converter = cutlass::FastInterleavedAndBiasedNumericArrayConverter<cutlass::half_t, cutlass::uint4b_t, kConvertCount>;

    // Each warp completes the internal reduce and writes the [Batch * NPerBlock * Interleave] results to the
    // corresponding address in shared memory
    template <int Num, int WarpSize>
    __device__ __forceinline__ static void sync(float* res, float (*sm)[Num * kInterleave])
    {
#pragma unroll
        for (int i = 0; i < Num; ++i)
        {
            res[i] += __shfl_xor_sync(~0, res[i], 16);
            res[i] += __shfl_xor_sync(~0, res[i], 8);
            res[i] += __shfl_xor_sync(~0, res[i], 1);
        }
        __syncthreads();
        int warp = threadIdx.x / WarpSize, lane = threadIdx.x % WarpSize;
        if (lane == 0 || lane == 2 || lane == 4 || lane == 6)
        {
#pragma unroll
            for (int i = 0; i < Num; ++i)
            {
                sm[warp][i * kInterleave + lane / 2] = res[i];
            }
        }
        __syncthreads();
    }
};

// Define a template struct for storing details about the weight layout for Int8b
template <>
struct WeightLayoutDetails<WeightOnlyQuantType::Int8b>
{
    // Every two rows of the original weights are interleaved into a row with stride of 64, so if each thread
    // processes 16 elements(for int8, we can use ldg.128 to load weights), then every group of four adjacent threads
    // will alternately process two different row weights
    // for example
    // every 128 consecutive int8 elements [128*i, 128*(i+1)-1] of row N under interleave layout,
   
