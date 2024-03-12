/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "cutlass_extensions/gemm_configs.h"
#include "tensorrt_llm/common/quantization.h"
#include <cuda_runtime_api.h>

namespace tk = tensorrt_llm::common;
namespace tkc = tensorrt_llm::cutlass_extensions;

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

/*
  This runner supports:
  int8_t inputs (A and B)
  float alpha scalings (either per-col, or per-col x per-row)
  T output (D) where T = {float, half, __nv_bfloat16} // TODO

  Activations, biases, scales and outputs are all assumed to be row-major.
  Weights are assumed to be column-major.
*/

class CutlassInt8GemmRunnerInterface
{
public:
    CutlassInt8GemmRunnerInterface() {}

    virtual ~CutlassInt8GemmRunnerInterface() {}

    virtual void gemm(const int8_t* A, const int8_t* B, tk::QuantMode quantOption, const float* alphaCol,
        const float* alphaRow, void* C, int m, int n, int k, tkc::CutlassGemmConfig gemmConfig, char* workspacePtr,
        const size_t workspaceBytes, cudaStream_t stream) = 0;

    // Returns desired workspace size in bytes.
    virtual size_t getWorkspaceSize(const int m, const int n, const int k) = 0;

    virtual std::vector<tkc::CutlassGemmConfig> getConfigs() const = 0;

protected:
    static constexpr int SPLIT_K_LIMIT = 7;
    static constexpr int MIN_M_TILE = 32;
    static constexpr int MIN_N_TILE = 64;
};

template <typename T>
class CutlassInt8GemmRunner : public virtual CutlassInt8GemmRunnerInterface
{
public:
    CutlassInt8GemmRunner() : mSm(0), mMultiProcessorCount(0) {}

    ~CutlassInt8GemmRunner() {}

    void gemm(const int8_t* A, const int8_t* B, tk::QuantMode quantOption, const float* alphaCol,
        const float* alphaRow, void* C, int m, int n, int k, tkc::CutlassGemmConfig gemmConfig, char* workspacePtr,
        const size_t workspaceBytes, cudaStream_t stream) override
    {
        dispatchToArch<T>(A, B, quantOption, alphaCol, alphaRow, static_cast<T*>(C), m, n, k, gemmConfig,
            workspacePtr, workspaceBytes, stream);
    }

    // Returns desired workspace size in bytes.
    size_t getWorkspaceSize(const int m, const int n, const int k) override
    {
        return 0;
    }

    std::vector<tkc::CutlassGemmConfig> getConfigs() const override
    {
        return {};
    }

private:
    int mSm;
    int mMultiProcessorCount;

    template <typename U>
    void dispatchToArch(const int8_t* A, const int8_t* B, tk::QuantMode quantOption, const float* alphaCol,
        const float* alphaRow, U* C, int m, int n, int k, tkc::CutlassGemmConfig gemmConfig, char* workspacePtr,
        const size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr)
    {
        // Set the number of SMs and multi-processors.
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        mSm = prop.multiProcessorCount;
        mMultiProcessorCount = prop.multiProcessorCount;

        // Implement the gemm function here.
    }
};

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm

