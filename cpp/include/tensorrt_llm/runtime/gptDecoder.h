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

#include "tensorrt_llm/common/cudaAllocator.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/decodingInput.h"
#include "tensorrt_llm/runtime/decodingOutput.h"
#include "tensorrt_llm/runtime/samplingConfig.h"

#include <cstdint>
#include <memory>

#include <NvInferRuntime.h>

namespace tensorrt_llm
{

namespace layers
{
// Forward declaration
template <typename T>
class DynamicDecodeLayer;
} // namespace layers

namespace runtime
{

class IGptDecoder
{
public:
    // Destructor
    virtual ~IGptDecoder() = default;

    // Set up the decoder with the given sampling configuration and batch size
    virtual void setup(SamplingConfig const& samplingConfig, size_t batchSize) = 0;

    // Perform forward pass through the decoder with the given input and output
    virtual bool forward(DecodingOutput& output, DecodingInput const& input) = 0;

    // Perform forward pass asynchronously through the decoder with the given input and output
    virtual void forwardAsync(DecodingOutput& output, DecodingInput const& input) = 0;

    // Gather tree of final output IDs
    static void gatherTree(ITensor& finalOutputIds, DecodingOutput const& decodingOutput,
        DecodingInput const& decodingInput, BufferManager const& manager);

    // Create a new instance of the decoder with the given data type, vocabulary size, padded vocabulary size, and CUDA stream
    static std::unique_ptr<IGptDecoder> create(
        nvinfer1::DataType dtype, size_t vocabSize, size_t vocabSizePadded, BufferManager::CudaStreamPtr const& stream);
};

// Template specialization of the IGptDecoder interface for a specific data type
template <typename T>
class GptDecoder : public virtual IGptDecoder
{

public:
    // Constructor
    GptDecoder(size_t vocabSize, size_t vocabSizePadded, CudaStreamPtr const& stream);

    // Set up the decoder with the given sampling configuration and batch size
    void setup(SamplingConfig const& samplingConfig, size_t batchSize) override;

    // Perform forward pass through the decoder with the given input and output
    bool forward(DecodingOutput& output, DecodingInput const& input) override;

    // Perform forward pass asynchronously through the decoder with the given input and output
    void forwardAsync(DecodingOutput& output, DecodingInput const& input) override;

private:
    // Buffer manager for handling memory allocation and deallocation
    BufferManager mManager;

    // CUDA allocator for handling memory allocation and deallocation
    common::CudaAllocator mAllocator;

    // Dynamic decode layer for performing the actual decoding
    std::shared_ptr<tensorrt_llm::layers::DynamicDecodeLayer<T>> mDynamicDecodeLayer;
};

// Create a new instance of the IGptDecoder interface for a specific data type
inline std::unique_ptr<IGptDecoder> IGptDecoder::create(
    nvinfer1::DataType dtype, size_t vocabSize, size_t vocabSizePadded, BufferManager::CudaStreamPtr const& stream)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT: return std::make_unique<GptDecoder<float>>(vocabSize, vocabSizePadded, stream);
    case nvinfer1::DataType::kHALF: return std::make_unique<GptDecoder<half>>(vocabSize, vocabSizePadded, stream);
    default: return nullptr;
    }
}
} // namespace runtime
} // namespace tensorrt_llm

