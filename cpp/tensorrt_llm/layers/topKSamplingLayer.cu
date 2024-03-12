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

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"
#include "tensorrt_llm/kernels/samplingTopPKernels.h"
#include "tensorrt_llm/layers/topKSamplingLayer.h"

#include <algorithm>
#include <float.h>
#include <cuda_runtime.h>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;

namespace tensorrt_llm
{
namespace layers
{

template <uint32_t TOP_K_MAX>
__global__ void setup_topk_runtime_args(int batch_size, uint32_t top_k, uint32_t* top_ks, int top_ks_size, float top_p,
    float* top_ps, int top_ps_size, bool* skip_decode)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = index; i < batch_size; i += gridDim.x * blockDim.x)
    {
        uint32_t k = top_ks_size > 1 ? top_ks[i] : top_k;
        float p = top_ps_size > 1 ? top_ps[i] : top_p;
        if (k == 0 && p == 0.0f)
        {
            // TensorRT-LLM's topp implementation does not support topp = 0.0f, but it
            // equivalent to greedy search. So, we set the topk = 1 as an alternative
            // solution.
            k = 1;
        }
        if (k > 0 && p == 0.0f)
        {
            // for compatibility reasons.
            p = 1.0f;
        }
        // Clip k value. A topk sampling kernel supports up to TOP_K_MAX=64.
        top_ks[i] = k > TOP_K_MAX ? TOP_K_MAX : k;
        if (k > TOP_K_MAX)
        {
            printf(
                "[WARNING] topk (%d) is larger than max supported number (%d) for "
                "token %d"
                " clip to max supported number %d. \n",
                k, TOP_K_MAX, i, top_ks[i]);
        }
        // Clip p value if it is out of range. range = [0.0, 1.0].
        top_ps[i] = p < 0.0f ? 0.0f : (p > 1.0f ? 1.0f : p);
        if (p < 0.0f || p > 1.0f)
        {
            printf(
                "[WARNING] topp (%f) is out of range ([0.0, 1.0f]) for token %d"
                " clip to closest number %f.\n",
                p, i, top_ps[i]);
        }
        skip_decode[i] = k == 0;
    }
}

template <typename T>
void TopKSamplingLayer<T>::allocateBuffer(size_t const batch_size, std::vector<uint32_t> const& top_k)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    uint32_t max_top_k = (top_k.size() > 0) ? *std::max_element(std::begin(top_k), std::end(top_k)) : 1;
    if (max_top_k == 0)
    {
        // for safety. TopKSamplingLayer handles a case of top_k=0 and top_p=0 as
        // a greedy decode, i.e. top_k=1, although such case has max_top_k=0.
        max_top_k = 1;
    }
    invokeTopKSampling<T>(nullptr, sampling_workspace_size_, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, max_top_k, 1.0f, vocab_size_padded_, nullptr, stream_, batch_size, skip_decode_buf_);
    sampling_workspace
