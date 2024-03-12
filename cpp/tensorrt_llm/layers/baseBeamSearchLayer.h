


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

#include "tensorrt_llm/common/tensor.h"
#include "tensorrt_llm/kernels/beamSearchTopkKernels.h"
#include "tensorrt_llm/kernels/penaltyTypes.h"
#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/layers/decodingParams.h"

#include <utility>

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{
struct BeamHypotheses;
}

namespace layers
{

template <typename T>
class BaseBeamSearchLayer : public BaseLayer
{
public:
    // Declare a type alias for SetupParams
    using SetupParams = DecodingSetupParams;

    // Constructor with parameters for initializing the base beam search layer
    BaseBeamSearchLayer(size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream, tc::IAllocator* allocator,
        bool is_free_buffer_after_forward);

    // Copy constructor for the base beam search layer
    BaseBeamSearchLayer(BaseBeamSearchLayer<T> const& beam_search_layer);

    // Destructor for the base beam search layer
    ~BaseBeamSearchLayer() override;

    // Type alias for SoftmaxParams
    using SoftmaxParams = DecodingParams;

    // ForwardParams class declaration with mandatory and optional parameters
    class ForwardParams : public SoftmaxParams
    {
    public:
        // Constructor with mandatory and optional parameters
        ForwardParams(
            int step, int ite, tc::Tensor logits, tc::Tensor endIds, tc::Tensor src_cache_indirection, int max_seq_len)
            : SoftmaxParams(step, ite, std::move(logits), std::move(endIds))
            , src_cache_indirection{std::move(src_cache_indirection)}
            , max_seq_len{max_seq_len}
        {
        }

        // Mandatory parameters
        int max_seq_len;
        tc::Tensor src_cache_indirection; // [local_batch_size, beam_width, max_seq_len]

        // Optional parameters
        std::optional<tc::Tensor> embedding_bias; // [vocab_size_padded]
        std::optional<tc::Tensor> input_lengths;  // [local_batch_size * beam_width]
    };

    // BeamSearchOutputParams class declaration with mandatory and optional parameters
    class BeamSearchOutputParams : public DecodingOutputParams
    {
    public:
        // Constructor with mandatory and optional parameters
        BeamSearchOutputParams(tc::Tensor outputIds, tc::Tensor parentIds, tc::Tensor tgt_cache_indirection)
            : DecodingOutputParams{std::move(outputIds)}
            , parent_ids{std::move(parentIds)}
            , tgt_cache_indirection{std::move(tgt_cache_indirection)}
        {
        }

        // Mandatory parameters
        tc::Tensor parent_ids;     // [max_seq_len, batch_size * beam_width], necessary in beam search
        tc::Tensor
            tgt_cache_indirection; // [local_batch_size, beam_width, max_seq_len], the k/v cache index for beam search
        std::shared_ptr<kernels::BeamHypotheses>
            beamHypotheses;        // a special structure which maintains some pointers of beam search

        // Optional parameter
        tc::Tensor
            parent_ids_ptr; // [batch_size] int*, each array is [beam_width, max_seq_len], necessary in beam search
    };

    // Forward method implementation for beam search
    void forward(BeamSearchOutputParams& outputs, ForwardParams const& params);

protected:
    // Meta data declaration
    size_t vocab_size_;
    size_t vocab_size_padded_;

    size_t topk_softmax_workspace_size_;
    void* topk_softmax_workspace_ = nullptr;

    float mTemperature;
    int mMinLength;
    float mRepetitionPenalty;
    tensorrt_llm::kernels::RepetitionPenaltyType mRepetitionPenaltyType;

    // AllocateBuffer method declaration for buffer allocation
    virtual void allocateBuffer(size_t batch_size, size_t beam_width) = 0;

    // InvokeSoftMax method declaration for softmax computation
    virtual void invokeSoftMax(BeamSearchOutputParams& outputs, SoftmaxParams const& params) = 0;

    // SetupBase method implementation for setting up base parameters
    void setupBase(SetupParams const& setupParams);

private:
    // FreeBuffer method implementation for freeing the buffer
    void freeBuffer();
};

// Kernel launcher for updating indirection cache
void update_indir_cache_kernelLauncher(int* tgt_indir_cache, const int* src_indir_cache,
