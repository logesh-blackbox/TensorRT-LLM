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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <ATen/ATen.h>

#include <stdexcept>

namespace tensorrt_llm::runtime
{

class Torch
{
public:
    /**
     * Converts an ITensor to an ATen Tensor.
     *
     * @param tensor The ITensor to convert.
     * @return The ATen Tensor equivalent to the input ITensor.
     */
    static at::Tensor tensor(ITensor::SharedPtr<void> tensor)
    {
        // Extract tensor properties
        auto const tensorOptions = at::device(TorchUtils::device((*tensor).data()))
                                       .pinned_memory((*tensor).getMemoryType() == MemoryType::kPINNED)
                                       .dtype(TorchUtils::dataType((*tensor).getDataType()))
                                       .layout(at::kStrided);

        // Create a Tensor for the given data
        return at::for_blob(tensor->data(), TorchUtils::shape((*tensor).getShape())) // NOLINT(*-use-after-move)
            .options(tensorOptions)
            .deleter(
                [ptr = std::move(tensor)](void* data) mutable
                {
                    try
                    {
                        TLLM_CHECK(data == ptr->data());
                        ptr.reset();
                    }
                    catch (std::exception const& e)
                    {
                        TLLM_LOG_EXCEPTION(e);
                    }
                })
            .make_tensor();
    }

    /**
     * Converts an IBuffer to an ATen Tensor.
     *
     * @param buffer The IBuffer to convert.
     * @return The ATen Tensor equivalent to the input IBuffer.
     */
    static at::Tensor buffer(IBuffer::SharedPtr<void> buffer)
    {
        // Create a Tensor for the given data
        auto const shape = ITensor::makeShape({static_cast<runtime::SizeType>(buffer->getSize())});
        return tensor(IBuffer::view(std::move(buffer), shape));
    }

    /**
     * Sets the current CUDA stream for ATen.
     *
     * @param cudaStream The CudaStream to set as the current stream.
     */
    static void setCurrentStream(runtime::CudaStream& cudaStream)
    {
        at::cuda::setCurrentCUDAStream(TorchUtils::stream(cudaStream));
    }

private:
    Torch() = default;
};

} // namespace tensorrt_llm::runtime


