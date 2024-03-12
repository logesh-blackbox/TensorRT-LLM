#pragma once

#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/plugins/common/checkMacrosPlugin.h"

#include <NvInferRuntime.h>

#include <cstring>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <memory>
#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif // ENABLE_MULTI_DEVICE
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>

namespace tensorrt_llm::plugins
{

class BasePlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    void setPluginNamespace(const char* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    [[nodiscard]] char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

protected:
    std::string mNamespace{api::kDefaultNamespace};
};

class BaseCreator : public nvinfer1::IPluginCreator
{
public:
    void setPluginNamespace(const char* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    [[nodiscard]] char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

protected:
    std::string mNamespace{api::kDefaultNamespace};
};

// Write values into buffer
template <typename T>
void write(char*& buffer, const T& val)
{
    std::memcpy(buffer, &val, sizeof(T));
    buffer += sizeof(T);
}

// Read values from buffer
template <typename T>
void read(const char*& buffer, T& val)
{
    std::memcpy(&val, buffer, sizeof(T));
    buffer += sizeof(T);
}

inline cudaDataType_t trtToCublasDtype(nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT: return CUDA_R_32F;
    case nvinfer1::DataType::kHALF: return CUDA_R_16F;
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 9
    case nvinfer1::DataType::kBF16: return CUDA_R_16BF;
#endif
    default: TLLM_THROW("Not supported data type for cuBLAS");
    }
}

std::uintptr_t constexpr kCudaMemAlign = 128;

int8_t* alignPtr(int8_t* ptr, uintptr_t to);

int8_t* nextWorkspacePtrCommon(int8_t* ptr, uintptr_t previousWorkspaceSize, const uintptr_t alignment);

int8_t* nextWorkspacePtr(
    int8_t* const base, uintptr_t& offset, const uintptr_t size, const uintptr_t alignment = kCudaMemAlign);

int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize);

int8_t* nextWorkspacePtrWithAlignment(int8_t* ptr, uintptr_t previousWorkspaceSize, const uintptr_t alignment);

size_t calculateTotalWorkspaceSize(size_t* workspaces, int count, const uintptr_t alignment = kCudaMemAlign);

// Like std::unique_ptr, but does not prevent generation of default copy constructor when used as class members.
// The copy constructor produces nullptr. So the plugin default copy constructor will not really copy this, and
// your clone() implementation is responsible for initializing such data members.
// With this we can simplify clone() implementation when there are many data menbers including at least one unique_ptr.
template <typename T, typename Del = std::default_delete<T>>
class UniqPtrWNullCopy : public std::unique_ptr<T, Del>
{
public:
    using std::unique_ptr<T, Del>::unique_ptr;

    // for compatibility with std::make_unique
    explicit UniqPtrWNullCopy(std::unique_ptr<T, Del>&& src)
        : std::unique_ptr<T, Del>::unique_ptr{std::move(src)}
    {
    }

    // copy constructor produces nullptr
    UniqPtrWNullCopy(UniqPtrWNullCopy const&)
        : std::unique_ptr<T, Del>::unique_ptr{}
    {
    }
};

} // namespace tensorrt_llm::plugins

inline bool isBuilding()
{
    auto constexpr key = "IS_BUILDING";
    auto const val = getenv(key);
    return val != nullptr && std::string(val) == "1";
}

#if ENABLE_MULTI_DEVICE
#define NCCLCHECK(cmd)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        ncclResult_t r = cmd;                                                                                          \
        if (r != ncclSuccess)                                                                                          \
        {                                                                                                              \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r));                      \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

std::unordered_map<nvinfer1::DataType, ncclDataType_t>* getDtypeMap();

std::map<std::set<int>, ncclComm_t>* getCommMap();
#endif // ENABLE_MULTI_DEVICE

//! To save GPU memory, all the plugins share the same cublas and
