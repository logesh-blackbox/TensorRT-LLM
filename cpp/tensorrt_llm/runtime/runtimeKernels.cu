#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/runtime/runtimeKernels.h"

#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

namespace tensorrt_llm::runtime::kernels
{

namespace
{

template <typename T>
__global__ void fill(T* data, std::size_t size, T const value)
{
    auto const tidx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (auto idx = tidx; idx < size; idx += stride)
    {
        data[idx] = value;
    }
}

template <typename T>
__global__ void fillBatch(T* data, std::int32_t const* indices, std::size_t size, T const* values)
{
    auto const batchIdx = indices[blockIdx.y];
    const T value = values[blockIdx.y];
    auto const tidx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    auto const startIdx = batchIdx * size;
    auto const endIdx = startIdx + size;

    for (auto idx = startIdx + tidx; idx < endIdx; idx += stride)
    {
        data[idx] = value;
    }
}

template <typename T>
__global__ void copyBatch(
    const T* srcData, T* dstData, std::int32_t const* srcIndices, std::int32_t const* dstIndices, std::size_t size)
{
    auto const srcBatchIdx = srcIndices[blockIdx.y];
    auto const dstBatchIdx = dstIndices[blockIdx.y];
    auto const tidx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    auto const srcStartIdx = srcBatchIdx * size;
    auto const dstStartIdx = dstBatchIdx * size;
    auto const srcEndIdx = srcStartIdx + size;

    auto srcIdx = srcStartIdx + tidx;
    auto dstIdx = dstStartIdx + tidx;

    for (; srcIdx < srcEndIdx; srcIdx += stride, dstIdx += stride)
    {
        dstData[dstIdx] = srcData[srcIdx];
    }
}

template <typename T>
__global__ void reduceSum(T* output, T const* input, std::size_t size)
{
    T threadSum = 0;
    for (auto index = threadIdx.x; index < size; index += blockDim.x)
    {
        threadSum += input[index];
    }

    T blockSum = 0;
    if (blockDim.x <= 32)
    {
        blockSum = tc::warpReduceSum(threadSum);
    }
    else
    {
        blockSum = tc::blockReduceSum(threadSum);
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        *output = blockSum;
    }
}

template <typename T>
__global__ void transpose(SizeType* output, SizeType const* input, SizeType const batchSize, SizeType const rowSize)
{
    SizeType const tidx = blockIdx.x * blockDim.x + threadIdx.x;
    SizeType const tidy = blockIdx.y * blockDim.y + threadIdx.y;

    for (SizeType batchIdx = tidy; batchIdx < batchSize; batchIdx += blockDim.y * gridDim.y)
    {
        for (SizeType tokenIdx = tidx; tokenIdx < rowSize; tokenIdx += blockDim.x * gridDim.x)
        {
            auto const inputIdx = batchIdx * rowSize + tokenIdx;
            auto const outputIdx = tokenIdx * batchSize + batchIdx;
            output[outputIdx] = input[inputIdx];
        }
    }
}

template <typename T>
__global__ void transposeWithOutputOffset(SizeType* output, SizeType const* input, SizeType const nbInputRows,
    SizeType const inputRowSize, SizeType const outputRowSize, SizeType const outputOffset)
{
    SizeType const tidx = blockIdx.x * blockDim.x + threadIdx.x;
    SizeType const tidy = blockIdx.y * blockDim.y + threadIdx.y;

    for (SizeType batchIdx = tidy; batchIdx < nbInputRows; batchIdx += blockDim.y * gridDim.y)
    {
        for (SizeType tokenIdx = tidx; tokenIdx < inputRowSize; tokenIdx += blockDim.x * gridDim.x)
        {
            auto const inputIdx = batchIdx * inputRowSize + tokenIdx;
            auto const outputIdx = tokenIdx * outputRowSize + outputOffset + batchIdx;
            output[outputIdx] = input[inputIdx];
        }
    }
}

template <typename T>
__global__ void transposeWithInputOffset(SizeType* output, SizeType const* input, SizeType const outputRowSize,
    SizeType const nbOutputRows, SizeType const inputRowSize, SizeType const inputOffset)
{
    SizeType const tidx = blockIdx.x * blockDim.x + threadIdx.x;
    SizeType const tidy = blockIdx.y * blockDim.y + threadIdx.y;

