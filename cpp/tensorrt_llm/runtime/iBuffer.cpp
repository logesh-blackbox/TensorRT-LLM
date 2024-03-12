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

#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/bufferView.h"

#include <cuda_runtime_api.h>

#include <memory>

using namespace tensorrt_llm::runtime;

MemoryType IBuffer::memoryType(void const* data)
{
    // Check if the provided data is a valid CUDA pointer
    cudaPointerAttributes attributes{};
    TLLM_CUDA_CHECK(::cudaPointerGetAttributes(&attributes, data));

    // Determine the memory type based on the CUDA pointer attributes
    switch (attributes.type)
    {
    case cudaMemoryTypeHost: return MemoryType::kPINNED;
    case cudaMemoryTypeDevice:
    case cudaMemoryTypeManaged: return MemoryType::kGPU;
    case cudaMemoryTypeUnregistered: return MemoryType::kCPU;
    default: TLLM_THROW("Unsupported memory type");
    }
}

// Create a sliced view of an existing IBuffer
IBuffer::UniquePtr IBuffer::slice(IBuffer::SharedPtr buffer, std::size_t offset, std::size_t size)
{
    // Create a new BufferView with the given offset and size
    return std::make_unique<BufferView>(std::move(buffer), offset, size);
}

// Wrap a given data pointer with an IBuffer
IBuffer::UniquePtr IBuffer::wrap(void* data, nvinfer1::DataType type, std::size_t size, std::size_t capacity)
{
    // Ensure the requested size is within the capacity
    TLLM_CHECK_WITH_INFO(size <= capacity, "Requested size is larger than capacity");

    // Determine the memory type of the provided data
    auto memoryType = IBuffer::memoryType(data);

    // Create a new GenericBuffer with the appropriate allocator
    IBuffer::UniquePtr result;
    auto const capacityInBytes = capacity * BufferDataType(type).getSize();
    switch (memoryType)
    {
    case MemoryType::kPINNED:
        result.reset(new GenericBuffer<PinnedBorrowingAllocator>( // NOLINT(modernize-make-unique)
            capacity, type, PinnedBorrowingAllocator(data, capacityInBytes)));
        break;
    case MemoryType::kCPU:
        result.reset( // NOLINT(modernize-make-unique)
            new GenericBuffer<CpuBorrowingAllocator>(capacity, type, CpuBorrowingAllocator(data, capacityInBytes)));
        break;
    case MemoryType::kGPU:
        result.reset( // NOLINT(modernize-make-unique)
            new GenericBuffer<GpuBorrowingAllocator>(capacity, type, GpuBorrowingAllocator(data, capacityInBytes)));
        break;
    default: TLLM_THROW("Unknown memory type");
    }

    // Resize the buffer to the requested size
    result->resize(size);
    return result;
}

// Output operator for IBuffer
std::ostream& tensorrt_llm::runtime::operator<<(std::ostream& output, IBuffer const& buffer)
{
    // Get a raw pointer to the data within the IBuffer
    auto data = const_cast<IBuffer&>(buffer).data();

    // Wrap the data in an ITensor and output it
    auto tensor = ITensor::wrap(data, buffer.getDataType(),
        ITensor::makeShape({static_cast<SizeType>(buffer.getSize())}), buffer.getCapacity());
    return output << *tensor;
}
