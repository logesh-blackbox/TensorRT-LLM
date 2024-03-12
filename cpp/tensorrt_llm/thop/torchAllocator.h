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

#include "tensorrt_llm/common/allocator.h"

#ifdef TORCH_CUDA
#include "torch/extension.h"
#endif

#include <cuda_runtime.h>
#include <string>
#include <unordered_map>

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm
{
namespace thop
{

class TorchAllocator : public tc::IAllocator
{
public:
    // Constructor initializes the stream member variable.
    explicit TorchAllocator(cudaStream_t stream)
        : mStream(stream)
    {
    }

    // Destructor is defaulted.
    ~TorchAllocator() override = default;

    // free method deallocates memory pointed to by the pointer and removes it from the pointer mapping.
    void free(void** ptr) override;

protected:
    // contains method checks if the pointer is present in the pointer mapping.
    bool contains(void const* ptr) const override
    {
        return mPointerMapping.find(ptr) != mPointerMapping.end();
    }

    // reallocType method returns the reallocation type based on the pointer and size.
    tc::ReallocType reallocType(void const* ptr, size_t size) const override;

    // malloc method allocates memory of the specified size and sets it to zero if setZero is true.
    void* malloc(size_t size, bool setZero) override;

    // memSet method sets the memory pointed to by the pointer to the specified value.
    void memSet(void* ptr, int val, size_t size) override;

private:
    // mPointerMapping is an unordered map that stores the mapping between the pointer and the corresponding torch::Tensor.
    std::unordered_map<void const*, torch::Tensor> mPointerMapping{};

    // mStream is a stream that is used for asynchronous memory operations.
    cudaStream_t m
