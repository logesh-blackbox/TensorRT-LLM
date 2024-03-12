# Copyright (c) 2022-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#include "tensorrt_llm/layers/dynamicDecodeLayer.h"
#include "tensorrt_llm/kernels/banBadWords.h"
#include "tensorrt_llm/kernels/banRepeatNgram.h"
#include "tensorrt_llm/kernels/decodingKernels.h"
#include "tensorrt_llm/kernels/stopCriteriaKernels.h"
#include "tensorrt_llm/layers/baseBeamSearchLayer.h"
#include "tensorrt_llm/layers/onlineBeamSearchLayer.h"
#include "tensorrt_llm/layers/topKSamplingLayer.h"
#include "tensorrt_llm/layers/topPSamplingLayer.h"

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;

namespace tensorrt_llm
{
namespace layers
{

template <typename T>
void DynamicDecodeLayer<T>::initialize()
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    mOnlineBeamsearchDecode = std::make_unique<OnlineBeamSearchLayer<T>>(
        vocab_size_, vocab_size_padded_, stream_, allocator_, is_free_buffer_after_forward_);

    mTopKDecode = std::make_unique<TopKSamplingLayer<T>>(vocab_size_, vocab_size_padded_, stream_, allocator_, false);

    mTopPDecode = std::make_unique<TopPSamplingLayer<T>>(
        vocab_size_, vocab_size_padded_, stream_, allocator_, false, cuda_device_prop_);
}

template <typename T>
DynamicDecodeLayer<T>::DynamicDecodeLayer(size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream,
    IAllocator* allocator, bool is_free_buffer_after_forward, cudaDeviceProp* cuda_device_prop)
    : BaseLayer(stream, allocator, is_free_buffer_after_forward)
    , vocab_size_(vocab_size)
    , vocab_size_padded_(vocab_size_padded)
    , cuda_device_prop_(cuda_device_prop)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    initialize();
}

template <typename T>
DynamicDecodeLayer<T>::~DynamicDecodeLayer()
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    freeBuffer();
}

template <typename T>
DynamicDecodeLayer<T>::DynamicDecodeLayer(DynamicDecodeLayer const& dynamic_decode_layer)
    : BaseLayer(dynamic_decode_layer)
    , vocab_size_(dynamic_decode_layer.vocab_size_)
    , vocab_size_padded_(dynamic_decode_layer.vocab_size_padded_)
    , cuda_device_prop_(dynamic_decode_layer.cuda_device_prop_)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    initialize();
}

namespace
{
template <typename T>
bool allSame(std::optional<std::vector<T>> const& vOpt)
{
    if (!vOpt)
    {
        return true;
    }

    auto const& v = *vOpt;

    if (v.size() <= 1)
    {
        return true;
    }
    auto first = v[0];
    for (std::size_t i = 1; i < v.size(); ++i)
    {
        if (v[i] != first)
        {
            return false;
        }
    }
    return true;
}

bool hasDiffRuntimeArgs(DecodingSetupParams const& params)
{
    return !allSame(params.presence_penalty) || !allSame(params.repetition_penalty) || !allSame(params.temperature)
        || !allSame(params.min_length);
}
} // namespace

template <typename T>
void DynamicDecodeLayer<T>::setup(size_t batch_size, size_t beam_width, SetupParams const& setupParams)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (beam_width == 1)
    { // sampling layers
        typename TopPSamplingLayer<T>::SetupParams samplingParams;

        samplingParams.temperature = setupParams.temperature;
        samplingParams.min_length = setupParams.min_length;
        samplingParams.repetition_penalty = setupParams.repetition_penalty;
        samplingParams.presence_penalty = setupParams.presence_penalty;

        samplingParams.runtime_top_k = setupParams.runtime_top_k;
        samplingParams.runtime_top_p = setupParams.runtime_top_p;
        samplingParams.random_seed = setupParams.random_seed;

        samplingParams.top_p_decay = setupParams.top_p_decay;
        samplingParams.top_p_min = setupParams.top_p_min;
        samplingParams.top_p_reset_ids = setupParams.top_p_reset_ids;

        mTopKDecode->setup(batch_size, samplingParams);
        mTopPDecode->setup(batch_size, samplingParams);
    }
    else
    { // beam search layer
