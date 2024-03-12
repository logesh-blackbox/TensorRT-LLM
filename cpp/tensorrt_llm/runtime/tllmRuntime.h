#pragma once

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include <NvInferRuntime.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace tensorrt_llm::runtime
{
class TllmRuntime
{
public:
    using TensorMap = StringPtrMap<ITensor>;

    explicit TllmRuntime(void const* engineData, std::size_t engineSize, nvinfer1::ILogger& logger);

    explicit TllmRuntime(void const* engineData, std::size_t engineSize);

    SizeType addContext(std::int32_t profileIndex) const;

    void clearContexts() const;

    void setTensors(SizeType contextIndex, TensorMap const& tensorMap) const;

    bool executeContext(SizeType contextIndex) const;

    CudaStream const& getStream() const;

    nvinfer1::ICudaEngine& getEngine()
    {
        return *mEngine;
    }

    nvinfer1::ICudaEngine const& getEngine() const
    {
        return *mEngine;
    }

    BufferManager& getBufferManager()
    {
        return mBufferManager;
    }

    BufferManager const& getBufferManager() const
    {
        return mBufferManager;
    }

private:
    BufferManager::CudaStreamPtr mStream;
    BufferManager mBufferManager;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    BufferManager::IBufferPtr mEngineBuffer;
    mutable std::vector<std::unique_ptr<nvinfer1::IExecutionContext>> mContexts;
    std::unique_ptr<ITensor> mDummyTensor;
};
} // namespace tensorrt_llm::runtime

