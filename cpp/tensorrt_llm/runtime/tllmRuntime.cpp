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
#include "tllmRuntime.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/common/tensor.h"
#include "tllmBuffers.h"
#include "tllmLogger.h"

#include <limits>
#include <type_traits>

using namespace tensorrt_llm::runtime;

namespace
{
using DimType = std::remove_reference_t<decltype(std::declval<nvinfer1::Dims>().d[0])>;
static_assert(sizeof(SizeType) >= sizeof(DimType), "SizeType is too small");
static_assert(std::is_signed<SizeType>::value, "SizeType must be signed");

nvinfer1::Dims shapeToDims(std::vector<std::size_t> const& shape)
{
    TLLM_CHECK(shape.size() <= nvinfer1::Dims::MAX_DIMS);
    nvinfer1::Dims dims;
    auto constexpr dim_max = std::numeric_limits<DimType>::max();
    dims.nbDims = static_cast<std::int32_t>(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i)
    {
        // shape[i] >= 0 because it has unsigned type. Check upper bound:
        TLLM_CHECK(shape[i] <= static_cast<std::size_t>(dim_max));
        dims.d[i] = static_cast<DimType>(shape[i]);
    }
    return dims;
}

std::vector<std::size_t> dimsToShape(nvinfer1::Dims const& dims)
{
    TLLM_CHECK(dims.nbDims >= 0);
    std::vector<std::size_t> shape(dims.nbDims);
    for (std::int32_t i = 0; i < dims.nbDims; ++i)
    {
        TLLM_CHECK(dims.d[i] >= 0);
        shape[i] = static_cast<std::size_t>(dims.d[i]);
    }
    return shape;
}

tensorrt_llm::runtime::TllmLogger defaultLogger{};

} // namespace

TllmRuntime::TllmRuntime(void const* engineData, std::size_t engineSize, nvinfer1::ILogger& logger)
    : mStream(std::make_shared<CudaStream>())
    , mBufferManager{mStream}
    , mRuntime{nvinfer1::createInferRuntime(logger)}
    , mEngine{mRuntime->deserializeCudaEngine(engineData, engineSize)}
{
    TLLM_CHECK_WITH_INFO(mEngine != nullptr, "Failed to deserialize cuda engine");
    auto const devMemorySize = mEngine->getDeviceMemorySize();
    mEngineBuffer = mBufferManager.gpu(devMemorySize);
}

TllmRuntime::TllmRuntime(void const* engineData, std::size_t engineSize)
    : TllmRuntime{engineData, engineSize, defaultLogger}
{
}

nvinfer1::IExecutionContext& TllmRuntime::addContext(std::int32_t profileIndex)
{
    TLLM_CHECK(0 <= profileIndex && profileIndex < mEngine->getNbOptimizationProfiles());
    mContexts.emplace_back(mEngine->createExecutionContextWithoutDeviceMemory());
    auto& context = *mContexts.back();
    context.setDeviceMemory(mEngineBuffer->data());
    context.setOptimizationProfileAsync(profileIndex, mStream->get());
    return context;
}

void TllmRuntime::clearContexts()
{
    for (auto& context : mContexts)
    {
        context.reset();
    }
    mContexts.clear();
}

bool TllmRuntime::executeContext(SizeType contextIndex) const
{
    NVTX3_FUNC_RANGE();
    auto& context = getContext(contextIndex);
    return context.enqueueV3(mStream->get());
}

void TllmRuntime::setInputTensors(SizeType contextIndex, TensorMap const& tensorMap)
{
    NVTX3_FUNC_RANGE();
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& context = getContext(contextIndex);
    for (std::int32_t i = 0; i < mEngine->getNbIOTensors(); ++i)
    {
        auto const name = mEngine->getIOTensorName(i);
        if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            NVTX3_SCOPED_RANGE(input_tensor);
            auto pos = tensorMap.find(name);
            if (pos == tensorMap.end())
            {
                auto expectedShape = mEngine->getTensorShape(name);
                TLLM_THROW(
                    "Input tensor '%s' not found; expected shape: %s", name, ITensor::toString(expectedShape).c_str());
            }
            auto const& tensor = pos->second;
            auto const tensorDtype = tensor
