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

#include <gmock/gmock.h>  // Include Google Mock header for unit testing
#include <gtest/gtest.h>  // Include Google Test header for unit testing

#include "tensorrt_llm/runtime/bufferManager.h"  // Include the BufferManager class header
#include "tensorrt_llm/runtime/torch.h"          // Include the Torch class header
#include "tensorrt_llm/runtime/torchView.h"      // Include the TorchView class header

#include <memory>  // Include the standard library memory header
#include <vector>  // Include the standard library vector header

using namespace tensorrt_llm::runtime;  // Use the runtime namespace
namespace tc = tensorrt_llm::common;    // Use the common namespace

// Define the TorchTest class that inherits from Google Test's Test class
class TorchTest : public ::testing::Test
{
protected:
    // SetUp() method that is called before each test case
    void SetUp() override
    {
        mDeviceCount = tc::getDeviceCount();  // Get the number of available CUDA devices
        if (mDeviceCount > 0)
        {
            mStream = std::make_unique<CudaStream>();  // Create a new CudaStream object
            Torch::setCurrentStream(*mStream);         // Set the current CUDA stream for TensorRT-LLM
        }
        else
        {
            GTEST_SKIP();  // Skip the test if no CUDA devices are available
        }
    }

    // TearDown() method that is called after each test case
    void TearDown() override {}

    int mDeviceCount;  // The number of available CUDA devices
    BufferManager::CudaStreamPtr mStream;  // The CUDA stream used for asynchronous operations
};

// Define a namespace for local test functions
namespace
{
template <nvinfer1::DataType DType>
void checkFilled(IBuffer& buffer, int fillValue)
{
    if (DType == buffer.getDataType())
    {
        // Check if the buffer is filled with the specified fill value
        // for the given data type
        EXPECT_THAT(BufferRange<typename CppDataType<DType>::type>(buffer), ::testing::Each(fillValue));
    }
}
} // namespace

// Define the TEST_F macro for the TorchTest class
TEST_F(TorchTest, Aten)
{
    BufferManager manager(mStream);  // Create a new BufferManager object with the current CUDA stream

    // Create a shape for a TRT-LLM tensor
    auto const shapeTllm = ITensor::makeShape({1, 2, 3, 4});

    // Convert the TRT-LLM shape to an ATen shape
    auto const shapeAten = TorchUtils::shape(shapeTllm);

    // Create small and large shapes for comparison
    auto const shapeSmall = ITensor::makeShape({1, 2, 3, 2});
    auto const shapeLarge = ITensor::makeShape({1, 2, 3, 8});

    // Compare the TRT-LLM shape and the ATen shape
    for (int i = 0; i < shapeAten.size(); ++i)
    {
        EXPECT_EQ(shapeAten[i], shapeTllm.d[i]) << i;
    }

    // Define a fill value and allocate a host buffer for the TRT-LLM tensor
    const int fillValue = 1;
    auto tensorHostBase = manager.allocate(MemoryType::kPINNED, shapeTllm, nvinfer1::DataType::kINT64);

    // Loop through all memory types and data types
    for (auto memoryType : {MemoryType::kCPU, MemoryType::kGPU, MemoryType::kPINNED})
    {
        for (auto dtype : {nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kHALF, nvinfer1::DataType::kINT8,
                 nvinfer1::DataType::kUINT8, nvinfer1::DataType::kINT32, nvinfer1::DataType::kINT64,
                 nvinfer1::DataType::kBF16, nvinfer1::DataType::kFP8, nvinfer1::DataType::kBOOL})
        {
            // Create a TRT-LLM tensor with the given memory type and data type
            ITensor::SharedPtr tensorTllm{manager.allocate(memoryType, shapeTllm, dtype)};

            // Convert the TRT-LLM tensor to an ATen tensor
            auto tensorAten = Torch::tensor(tensorTllm);

            // Check the device type and memory pinning of the ATen tensor
            EXPECT_TRUE(
                (memoryType == MemoryType::kGPU && tensorAten.device().is_cuda()) || tensorAten.device().is_cpu());
            EXPECT_EQ(memoryType == MemoryType::kPINNED, tensorAten.is_pinned());

            // Check the data type and shape of the ATen tensor
            EXPECT_EQ(TorchUtils::dataType(dtype), tensorAten.dtype());
            EXPECT_THAT(tensorAten.sizes(), ::testing::ElementsAreArray(shapeAten));

            // Check if the data pointers of the TRT-LLM tensor and the ATen tensor are the same
            EXPECT_EQ(tensorAten.data_ptr(), tensor
