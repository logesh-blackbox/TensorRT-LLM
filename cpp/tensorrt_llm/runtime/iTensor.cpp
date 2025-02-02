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

#include "tensorrt_llm/runtime/iTensor.h"

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/tensorView.h"

#include <initializer_list>
#include <limits>
#include <memory>

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;

ITensor::UniquePtr ITensor::slice(std::shared_ptr<ITensor> tensor, std::size_t offset, std::size_t size)
{
    return std::make_unique<TensorView>(std::move(tensor), offset, size);
}

ITensor::UniquePtr ITensor::view(std::shared_ptr<IBuffer> buffer, nvinfer1::Dims const& dims)
{
    auto const size = buffer->getSize();
    return std::make_unique<TensorView>(std::move(buffer), 0, size, dims);
}

nvinfer1::Dims ITensor::makeShape(std::initializer_list<SizeType> const& dims)
{
    TLLM_CHECK_WITH_INFO(dims.size() <= nvinfer1::Dims::MAX_DIMS, "Number of dimensions is too large");
    nvinfer1::Dims shape{};
    shape.nbDims = dims.size();
    for (std::size_t i = 0; i < dims.size(); ++i)
    {
        shape.d[i] = *std::data(dims)[i];
    }
    return shape;
}

std::string ITensor::toString(nvinfer1::Dims const& dims)
{
    if (dims.nbDims < 0)
    {
        return "invalid";
    }
    else if (dims.nbDims == 0)
    {
        return "()";
    }
    else
    {
        return tc::arr2str(dims.d, dims.nbDims);
    }
}

ITensor::UniquePtr ITensor::wrap(void* data, nvinfer1::DataType type, nvinfer1::Dims const& shape, std::size_t capacity)
{
    auto const size = volumeNonNegative(shape);
    TLLM_CHECK_WITH_INFO(size <= capacity, "Requested size is larger than capacity");
    auto memoryType = IBuffer::memoryType(data);

    ITensor::UniquePtr result;
    auto const capacityInBytes = capacity * BufferDataType(type).getSize();
    switch (memoryType)
    {
    case MemoryType::kPINNED:
        result.reset(new GenericTensor<PinnedBorrowingAllocator>( // NOLINT(modernize-make-unique)
            shape, capacity, type, PinnedBorrowingAllocator(data, capacityInBytes)));
        break;
    case MemoryType::kCPU:
        result.reset( // NOLINT(modernize-make-unique)
            new GenericTensor<CpuBorrowingAllocator>(
                shape, capacity, type, CpuBorrowingAllocator(data, capacityInBytes)));
        break;
    case MemoryType::kGPU:
        result.reset( // NOLINT(modernize-make-unique)
            new GenericTensor<GpuBorrowingAllocator>(
                shape, capacity, type, GpuBorrowingAllocator(data, capacityInBytes)));
        break;
    default: TLLM_THROW("Unknown memory type");
    }
    return result;
}

namespace
{
template <typename T>
void printTensor(ITensor const& tensor, std::ostream& out)
{
    TLLM_CHECK_WITH_INFO(tensor.getDataType() == TRTDataType<typename std::remove_cv<T>::type>::value,
        tc::fmtstr("Data type mismatch: %d vs %d", static_cast<std::int32_t>(tensor.getDataType()),
            static_cast<std::int32_t>(TRTDataType<typename std::remove_cv<T>::type>::value)));
    auto const& shape = tensor.getShape();
    out << "shape: " << shape << std::endl;
    out << "vals: " << std::endl;

    BufferManager::ITensorPtr host{};
    T const* hostData;
    if (tensor.getMemoryType() == MemoryType::kGPU)
    {
        auto streamPtr = std::make_shared<CudaStream>();
        BufferManager manager{streamPtr};
        host = manager.copyFrom(tensor, MemoryType::kCPU);
        streamPtr->synchronize();
        hostData = bufferCast<T>(*host);
    }
    else
    {
        hostData = bufferCast<T>(tensor);
    }

    using TOutput
        = std::conditional_t<std::is_same_v<T, std::int8_t> || std::is_same_v<T, std::uint8_t>, std::int32_t, T>;
    if (shape.nbDims > 3)
    {
        out << "Not printing elements for more than 3 dims\n";
    }
    else if (shape.nbDims == 3 && shape.d[2
