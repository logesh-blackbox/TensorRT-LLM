#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/tensor.h"
#include <curand_kernel.h>
#include <sys/stat.h>
#include <unordered_map>

namespace tensorrt_llm
{
namespace common
{

template <typename T>
void deviceMalloc(T** ptr, size_t size, bool is_random_initialize)
{
    check_cuda_error(cudaMalloc((void**) (ptr), sizeof(T) * size));
    if (is_random_initialize)
    {
        cudaRandomUniform(*ptr, size);
    }
}

template <typename T>
void deviceMemSetZero(T* ptr, size_t size)
{
    check_cuda_error(cudaMemset(static_cast<void*>(ptr), 0, sizeof(T) * size));
}

template <typename T>
void deviceFree(T*& ptr)
{
    if (ptr != NULL)
    {
        check_cuda_error(cudaFree(ptr));
        ptr = NULL;
    }
}

template <typename T>
void deviceFill(T* devptr, size_t size, T value, cudaStream_t stream)
{
    T* arr = new T[size];
    std::fill(arr, arr + size, value);
    check_cuda_error(cudaMemcpyAsync(devptr, arr, sizeof(T) * size, cudaMemcpyHostToDevice, stream));
    delete[] arr;
}

template <typename T>
void cudaD2Hcpy(T* tgt, const T* src, const size_t size)
{
    check_cuda_error(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyDeviceToHost));
}

template <typename T>
void cudaH2Dcpy(T* tgt, const T* src, const size_t size)
{
    check_cuda_error(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyHostToDevice));
}

template <typename T>
void cudaD2Dcpy(T* tgt, const T* src, const size_t size)
{
    check_cuda_error(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyDeviceToDevice));
}

template <typename T_OUT, typename T_IN>
__global__ void cudaCast(T_OUT* dst, T_IN* src, const size_t size)
{
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x)
    {
        dst[tid] = (T_OUT) ((float) (src[tid]));
    }
}

template <typename T_OUT, typename T_IN>
void invokeCudaCast(T_OUT* dst, T_IN const* const src, const size_t size, cudaStream_t stream)
{
    cudaCast<<<256, 256, 0, stream>>>(dst, src, size);
}

template <typename T>
void cudaAutoCpy(T* tgt, const T* src, const size_t size, cudaStream_t stream)
{
    if (stream != NULL)
    {
        check_cuda_error(cudaMemcpyAsync(tgt, src, sizeof(T) * size, cudaMemcpyDefault, stream));
    }
    else
    {
        check_cuda_error(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyDefault));
    }
}

template <typename T>
int loadWeightFromBinFunc(T* ptr, std::vector<size_t> shape, std::string filename)
{
    std::vector<T> host_array = loadWeightFromBinHelper<T>(shape, filename);

    if (host_array.empty())
    {
        return 0;
    }

    if (std::is_same<T, T>::value == true)
    {
        cudaH2Dcpy(ptr, host_array.data(), host_array.size());
    }
    else
    {
        T* ptr_2 = nullptr;
        deviceMalloc(&ptr_2, host_array.size(), false);
        cudaH2Dcpy(ptr_2, host_array.data(), host_array.size());
        invokeCudaD2DcpyConvert(ptr, ptr_2, host_array.size());
        deviceFree(ptr_2);
    }
    return 0;
}

template <typename T>
int loadWeightFromBin(T* ptr, std::vector<size_t> shape, std::string filename, TRTLLMCudaDataType model_file_type)
{
    switch (model_file_type)
    {
    case TRTLLMCudaDataType::FP32: loadWeightFromBinFunc<T>(ptr, shape, filename); break;
    case TRTLLMCudaDataType::FP16: loadWeightFromBinFunc<T>(ptr, shape, filename); break;
    case TRTLLMCudaDataType::INT8: loadWeightFromBinFunc<T>(ptr, shape, filename); break;
#ifdef ENABLE_BF16
    case TRTLLMCudaDataType::BF16: loadWeightFromBinFunc<T>(ptr, shape, filename); break;
#endif
#ifdef ENABLE_FP8
    case TRTLLMCudaDataType::FP8: loadWeightFromBinFunc<T>(ptr, shape, filename); break;
#endif
    default: TLLM_LOG_ERROR("Does not support TRTLLMCudaDataType=%d", model_file_type); TLLM_CHECK(false);
    }
    return 0;
}

template <typename T_IN, typename T_OUT>
__global
