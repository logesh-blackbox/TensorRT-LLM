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

#include <algorithm>
#include <cstdint>
#include <string>

namespace tensorrt_llm::runtime
{

class MemoryCounters
{
public:
    // Using SizeType as std::size_t for memory size and DiffType as std::ptrdiff_t for memory difference
    using SizeType = std::size_t;
    using DiffType = std::ptrdiff_t;

    // Default constructor
    MemoryCounters() = default;

    // Getters for memory counters
    [[nodiscard]] SizeType getGpu() const
    {
        return mGpu; // Return the current GPU memory usage
    }

    [[nodiscard]] SizeType getCpu() const
    {
        return mCpu; // Return the current CPU memory usage
    }

    [[nodiscard]] SizeType getPinned() const
    {
        return mPinned; // Return the current pinned memory usage
    }

    [[nodiscard]] DiffType getGpuDiff() const
    {
        return mGpuDiff; // Return the difference in GPU memory usage
    }

    [[nodiscard]] DiffType getCpuDiff() const
    {
        return mCpuDiff; // Return the difference in CPU memory usage
    }

    [[nodiscard]] DiffType getPinnedDiff() const
    {
        return mPinnedDiff; // Return the difference in pinned memory usage
    }

    // Allocate memory for a specific memory type
    template <MemoryType T>
    void allocate(SizeType size)
    {
        auto const sizeDiff = static_cast<DiffType>(size);
        if constexpr (T == MemoryType::kGPU)
        {
            mGpu += size; // Increase GPU memory usage
            mGpuDiff = sizeDiff; // Update the GPU memory difference
        }
        else if constexpr (T == MemoryType::kCPU)
        {
            mCpu += size; // Increase CPU memory usage
            mCpuDiff = sizeDiff; // Update the CPU memory difference
        }
        else if constexpr (T == MemoryType::kPINNED)
        {
            mPinned += size; // Increase pinned memory usage
            mPinnedDiff = sizeDiff; // Update the pinned memory difference
        }
        else
        {
            TLLM_THROW("Unknown memory type: %s", MemoryTypeString<T>::value);
        }
    }

    // Allocate memory for a specific memory type
    void allocate(MemoryType memoryType, SizeType size);

    // Deallocate memory for a specific memory type
    template <MemoryType T>
    void deallocate(SizeType size)
    {
        auto const sizeDiff = -static_cast<DiffType>(size);
        if constexpr (T == MemoryType::kGPU)
        {
            mGpu -= std::min(size, mGpu); // Decrease GPU memory usage
            mGpuDiff = sizeDiff; // Update the GPU memory difference
        }
        else if constexpr (T == MemoryType::kCPU)
        {
            mCpu -= std::min(size, mCpu); // Decrease CPU memory usage
            mCpuDiff = sizeDiff; // Update the CPU memory difference
        }
        else if constexpr (T == MemoryType::kPINNED)
        {
            mPinned -= std::min(size, mPinned); // Decrease pinned memory usage
            mPinnedDiff = sizeDiff; // Update the pinned memory difference
        }
        else
        {
            TLLM_THROW("Unknown memory type: %s", MemoryTypeString<T>::value);
        }
    }

    // Deallocate memory for a specific memory type
    void deallocate(MemoryType memoryType, SizeType size);

    // Get the singleton instance of MemoryCounters
    static MemoryCounters& getInstance()
    {
        return mInstance;
    }

    // Convert bytes to a human-readable string
    static std::string bytesToString(SizeType bytes, int precision = 2);

    // Convert bytes to a human-readable string
    static std::string bytesToString(DiffType bytes, int precision = 2);

private:
    SizeType mGpu{}, mCpu{}, mPinned{}; // Memory counters for GPU, CPU, and pinned memory
    DiffType mGpuDiff{}, mCpuDiff{}, mPinnedDiff{}; // Differences in memory usage for GPU, CPU, and pinned memory
    static thread_local MemoryCounters mInstance; // Singleton instance of MemoryCounters
};

} // namespace tensorrt_llm::runtime

