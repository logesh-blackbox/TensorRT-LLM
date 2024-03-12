/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttentionUtils.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

__inline__ __device__ int target_index(int id1, int id2, int id3, int id4, int dim_1, int dim_2, int dim_3, int dim_4)
{
    return id1 * (dim_2 * dim_3 * dim_4) + id3 * (dim_2 * dim_4) + id2 * dim_4 + id4;
}

template <typename T>
__global__ void addQKVBiasIA3Transpose(T* q_out, T* k_out, T* v_out, const T* __restrict q_in,
    const T* __restrict bias_q, const T* __restrict k_in, const T* __restrict bias_k, const T* __restrict v_in,
    const T* __restrict bias_v, const int* ia3_tasks, const T* ia3_key_weights, const T* ia3_value_weights,
    const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    const int n = head_num * size_per_head;
    const int batch_id = blockIdx.x;
    const int word_id = blockIdx.y;
    const int row_id = batch_id * seq_len + word_id;

    const bool use_ia3 = ia3_tasks != nullptr;
    const int ia3_task = use_ia3 ? ia3_tasks[batch_id] : 0;
    const bool use_ia3_key = use_ia3 && (ia3_key_weights != nullptr);
    const bool use_ia3_value = use_ia3 && (ia3_value_weights != nullptr);

    for (int col_id = threadIdx.x; col_id < n; col_id += blockDim.x)
    {
        const int head_id = col_id / size_per_head;
        const int size_id = col_id % size_per_head;
        const int target_id = batch_id * (head_num * seq_len * size_per_head) + head_id * seq_len * size_per_head
            + word_id * size_per_head + size_id;
        const int src_id = row_id * n + col_id;

        T q = ldg(&q_in[src_id]);
        q_out[target_id] = add(q, ldg(&bias_q[col_id]));

        T k = add(ldg(&k_in[src_id]), ldg(&bias_k[col_id]));
        if (use_ia3_key)
        {
            k = k * ia3_key_weights[ia3_task * n + col_id];
        }
        k_out[target_id] = k;

        T v = add(ldg(&v_in[src_id]), ldg(&bias_v[col_id]));
        if (use_ia3_value)
        {
            v = v * ia3_value_weights[ia3_task * n + col_id];
        }
        v_out[target_id] = v;
    }
}

template <typename T>
__global__ void QKVIA3Transpose(T* q_out, T* k_out, T* v_out, const T* __restrict q_in, const T* __restrict k_in,
    const T* __restrict v_in, const int* ia3_tasks, const T* __restrict ia3_key_weights,
    const T* __restrict ia3_value_weights, const int batch_size, const int seq_len, const int head_num,
    const int size_per_head)
{
    const int n = head_num * size_per_head;
    const int batch_id = blockIdx.x;
    const int word_id = blockIdx.y;
    const int row_id = batch_id * seq_len + word_id;

    const bool use_ia3 = ia3_tasks != nullptr;
    const int ia3_task = use_ia3 ? ia3_tasks[batch_id] : 0;
    const bool use_ia3_key = use_ia3 && (ia3_key_weights != nullptr);
    const bool use_ia3_value = use_ia3 && (ia3_value_weights != nullptr);

    for (int col_id = threadIdx.x; col
