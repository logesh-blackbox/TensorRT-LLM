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
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/tllmBuffers.h"

#include <memory>
#include <string>

namespace tensorrt_llm::runtime
{
class BufferView : virtual public IBuffer
{
public:
    // Constructor for BufferView class, takes a shared pointer to an IBuffer, an offset, and a size.
    // Throws an out_of_range exception if the offset or the slice (offset + size) exceeds the buffer size.
    explicit BufferView(IBuffer::SharedPtr<IBuffer> buffer, std::size_t offset, std::size_t size);

    // Returns a pointer to the data in the buffer. If the size of the buffer is 0, returns nullptr.
    void* data() override;

    // Const version of data() method.
    void const* data() const override;

    // Returns the size of the buffer.
    [[nodiscard]] std::size_t getSize() const override;

    // Returns the capacity of the buffer.
    [[nodiscard]] std::size_t getCapacity() const override;

    // Returns the data type of the buffer.
    [[nodiscard]] nvinfer1::DataType getDataType() const override;

    // Returns the memory type of the buffer.
    [[nodiscard]] MemoryType getMemoryType() const override;

    // Resizes the buffer to a new size. Throws an exception if the new size is larger than the capacity.
    void resize(std::size_t newSize) override;

    // Releases the buffer.
    void release() override;

    // Destructor.
    ~BufferView() override;

private:
    // Shared pointer to an IBuffer.
    IBuffer::SharedPtr<IBuffer> mBuffer;

    // Offset and size of the buffer view.
    std::size_t mOffset, mSize;
};

} // namespace tensorrt_llm::runtime

