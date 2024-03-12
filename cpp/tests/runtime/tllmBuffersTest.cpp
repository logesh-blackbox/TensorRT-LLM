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

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/runtime/memoryCounters.h"
#include "tensorrt_llm/runtime/tllmBuffers.h"

#include <limits>
#include <memory>
#include <sstream>
#include <type_traits>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

class TllmBuffersTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override
    {
        mDeviceCount = tc::getDeviceCount();

        if (mDeviceCount == 0)
            GTEST_SKIP();
    }

    void TearDown() override {}

    int mDeviceCount;
};

TEST_F(TllmBuffersTest, Stream)
{
    CudaStream stream{};
    EXPECT_NE(stream.get(), nullptr);
    auto ptr = std::make_shared<CudaStream>();
    EXPECT_NE(ptr->get(), nullptr);
    EXPECT_GE(ptr->getDevice(), 0);
    CudaStream lease{ptr->get(), ptr->getDevice(), false};
    EXPECT_EQ(lease.get(), ptr->get());
}

TEST_F(TllmBuffersTest, CudaAllocator)
{
    auto constexpr size = 1024;
    CudaAllocator allocator{};
    auto& counters = MemoryCounters::getInstance();
    EXPECT_EQ(counters.getGpu(), 0);
    auto ptr = allocator.allocate(size);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(counters.getGpu(), size);
    EXPECT_EQ(counters.getGpuDiff(), size);
    EXPECT_NO_THROW(allocator.deallocate(ptr, size));
    EXPECT_EQ(counters.getGpu(), 0);
    EXPECT_EQ(counters.getGpuDiff(), -size);
    EXPECT_EQ(allocator.getMemoryType(), MemoryType::kGPU);
    EXPECT_THROW(allocator.deallocate(ptr, size), std::runtime_error);
}

TEST_F(TllmBuffersTest, PinnedAllocator)
{
    auto constexpr size = 1024;
    PinnedAllocator allocator{};
    auto& counters = MemoryCounters::getInstance();
    EXPECT_EQ(counters.getPinned(), 0);
    auto ptr = allocator.allocate(size);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(counters.getPinned(), size);
    EXPECT_EQ(counters.getPinnedDiff(), size);
    EXPECT_NO_THROW(allocator.deallocate(ptr, size));
    EXPECT_EQ(counters.getPinned(), 0);
    EXPECT_EQ(counters.getPinnedDiff(), -size);
    EXPECT_EQ(allocator.getMemoryType(), MemoryType::kPINNED);
}

TEST_F(TllmBuffersTest, HostAllocator)
{
    auto constexpr size = 1024;
    HostAllocator allocator{};
    auto& counters = MemoryCounters::getInstance();
    EXPECT_EQ(counters.getCpu(), 0);
    auto ptr = allocator.allocate(size);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(counters.getCpu(), size);
    EXPECT_EQ(counters.getCpuDiff(), size);
    EXPECT_NO_THROW(allocator.deallocate(ptr, size));
    EXPECT_EQ(counters.getCpu(), 0);
    EXPECT_EQ(counters.getCpuDiff(), -size);
}

TEST_F(TllmBuffersTest, CudaAllocatorAsync)
{
    auto streamPtr = std::make_shared<CudaStream>();
    auto constexpr size = 1024;
    CudaAllocatorAsync allocator{streamPtr};
    auto& counters = MemoryCounters::getInstance();
    EXPECT_EQ(counters.getGpu(), 0);
    auto ptr = allocator.allocate(size);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(counters.getGpu(), size);
    EXPECT_EQ(counters.getGpuDiff(), size);
    EXPECT_NO_THROW(allocator.deallocate(ptr, size));
    EXPECT_EQ(counters.getGpu(), 0);
    EXPECT_EQ(counters.getGpuDiff(), -size);
    EXPECT_EQ(allocator.getMemoryType(), MemoryType::kGPU);
    streamPtr->synchronize();
    CudaAllocatorAsync allocatorCopy = allocator;
    EXPECT_EQ(allocatorCopy.getCudaStream(), streamPtr);
    CudaAllocatorAsync allocatorMove = std::move(allocatorCopy);
    EXPECT_EQ(allocatorMove.getCudaStream(), streamPtr);
    EXPECT_THROW(allocator.deallocate(ptr, size), std::runtime_error);
}

namespace
{
void testBuffer(IBuffer& buffer, std::int32_t typeSize)
{
    auto const size = buffer.getSize();
   
