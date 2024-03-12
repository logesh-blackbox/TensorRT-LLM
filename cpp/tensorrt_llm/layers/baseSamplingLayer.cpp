#include "tensorrt_llm/layers/baseSamplingLayer.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/samplingPenaltyKernels.h"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"

#include <algorithm>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;

namespace tensorrt_llm
{
namespace layers
{
template <typename T>
void BaseSamplingLayer<T>::allocateBuffer(size_t batch_size)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    curandstate_buf_ = allocator_->reMalloc(curandstate_buf_, sizeof(curandState_t) * batch_size, false);
    random_seeds_buf_ = allocator_->reMalloc(random_seeds_buf_, sizeof(unsigned long long) * batch_size, false);
    temperature_buf_ = allocator_->reMalloc(temperature_buf_, sizeof(float) * batch_size, false);
    repetition_penalty_buf_ = allocator_->reMalloc(repetition_penalty_buf_, sizeof(float) * batch_size, false);
    min_lengths_buf_ = allocator_->reMalloc(min_lengths_buf_, sizeof(int) * batch_size, false);
    runtime_logits_buf_ = allocator_->reMalloc(runtime_logits_buf_, sizeof(T) * batch_size * vocab_size_padded_, false);
    skip_decode_buf_ = allocator_->reMalloc(skip_decode_buf_, sizeof(bool) * batch_size, false);

    // host buffers.
    skip_decode_ = (bool*) std::realloc(skip_decode_, sizeof(bool) * batch_size);
    TLLM_CHECK(skip_decode_ != nullptr);

    is_allocate_buffer_ = true;
}

template <typename T>
void BaseSamplingLayer<T>::freeBuffer()
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_)
    {
        allocator_->free((void**) (&curandstate_buf_));
        allocator_->free((void**) (&random_seeds_buf_));
        allocator_->free((void**) (&temperature_buf_));
        allocator_->free((void**) (&repetition_penalty_buf_));
        allocator_->free((void**) (&min_lengths_buf_));
        allocator_->free((void**) (&runtime_logits_buf_));
        allocator_->free((void**) (&skip_decode_buf_));
        std::free(skip_decode_);
        is_allocate_buffer_ = false;
    }
}

template <typename T>
BaseSamplingLayer<T>::BaseSamplingLayer(size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream,
    IAllocator* allocator, bool is_free_buffer_after_forward, cudaDeviceProp* cuda_device_prop)
    : BaseLayer(stream, allocator, is_free_buffer_after_forward, cuda_device_prop)
    , vocab_size_(vocab_size)
    , vocab_size_padded_(vocab_size_padded)
{
}

template <typename T>
BaseSamplingLayer<T>::BaseSamplingLayer(BaseSamplingLayer const& sampling_layer)
    : BaseLayer(sampling_layer)
    , vocab_size_(sampling_layer.vocab_size_)
    , vocab_size_padded_(sampling_layer.vocab_size_padded_)
    , sampling_workspace_size_(sampling_layer.sampling_workspace_size_)
{
}

template <typename T>
BaseSamplingLayer<T>::~BaseSamplingLayer()
{
}

template <typename T>
void BaseSamplingLayer<T>::setupBase(const size_t batch_size, SetupParams const& setupParams)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    allocateBuffer(batch_size);

    // If runtime argument has single random seed, using this random seed to
    // initialize the random table of all sentences. If the argument has
    // [batch_size] random seeds, initializing the random table by different
    // random seeds respectively. If no random seed, initialize the random table
    // of all sentences by 0 directly.
    if (setupParams.random_seed)
    {
        if (setupParams.random_seed->size() == 1)
        {
            invokeCurandInitialize(curandstate_buf_, batch_size, setupParams.random_seed->front(), stream_);
            sync_check_cuda_error();
        }
        else
        {
            TLLM_CHECK_WITH_INFO(setupParams.random_seed->size() == batch_size, "Random seed vector size mismatch.");
            cudaAutoCpy(random_seeds_buf_, setupParams.random_seed->data(), batch_size, stream_);
            invokeCurandBatchInitialize(curandstate_buf_, batch_size, random_seeds_buf_, stream_);
            sync_check_cuda_error();
        }
    }
    else
    {
        // Initialize curand states using the default seed 0.
        invokeCurandInitialize(curandstate_buf_, batch_size, 0, stream_);
    }

    // Setup penalties.
    auto fillBuffers
        = [this, &batch_size](auto const& optParam, auto const defaultValue, auto& hostBuffer, auto& deviceBuffer)
    {
        hostBuffer.resize(batch_size);
        if (!optParam)
        {
            std::fill(std::begin(hostBuffer), std::end(hostBuffer), defaultValue);
        }
        else if (optParam->size() == 1)
        {
            std::fill(std
