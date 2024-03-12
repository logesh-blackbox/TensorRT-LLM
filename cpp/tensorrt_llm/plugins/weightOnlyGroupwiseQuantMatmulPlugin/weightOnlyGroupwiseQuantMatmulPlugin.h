/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "tensorrt_llm/kernels/preQuantScaleKernel.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/kernelLauncher.h"
#include "tensorrt_llm/plugins/common/gemmPluginProfiler.h"
#include "tensorrt_llm/plugins/common/plugin.h"

#include <cutlass/numeric_types.h>

#include <cassert>
#include <memory>
#include <set>
#include <string>
#include <vector>

// The blank line here is to avoid clang-format -sort-includes option reordering these two cutlass header files and
// breaking dependencies
#include "cutlass/integer_subbyte.h"

namespace tensorrt_llm::plugins
{

using WeightOnlyGemmRunner = tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface;
using WeightOnlyGemmRunnerPtr = std::shared_ptr<WeightOnlyGemmRunner>;

// WeightOnlyGroupwiseQuantGemmPluginProfiler class
// This class is responsible for profiling different tactics (configurations) of the groupwise quantized GEMM algorithm.
// It inherits from the GemmPluginProfiler class, which provides a common interface for profiling GEMM algorithms.
class WeightOnlyGroupwiseQuantGemmPluginProfiler
    : public GemmPluginProfiler<tensorrt_llm::cutlass_extensions::CutlassGemmConfig, WeightOnlyGemmRunnerPtr,
          GemmIdCore, GemmIdCoreHash>
{
public:
    // Using WeightOnlyGroupwiseQuantGemmPluginProfiler::Config as a type alias for the CutlassGemmConfig type.
    using Config = tensorrt_llm::cutlass_extensions::CutlassGemmConfig;

    // Setters for the quantization algorithm and group size.
    void setQuantAlgo(int quantAlgo)
    {
        mQuantAlgo = quantAlgo;
    }

    void setGroupSize(int groupSize)
    {
        mGroupSize = groupSize;
    }

protected:
    // Implementation of the runTactic method from the GemmPluginProfiler class.
    // This method runs a specific time-consuming tactic (configuration) of the groupwise quantized GEMM algorithm.
    void runTactic(int m, int n, int k, const Config& tactic, char* workspace, const cudaStream_t& stream) override;

    // Implementation of the computeTmpSize method from the GemmPluginProfiler class.
    // This method computes the temporary size required for the specified tactic.
    void computeTmpSize(int maxM, int n, int k) override;

    // Implementation of the getTactics method from the GemmPluginProfiler class.
    // This method returns a vector of tactics (configurations) that can be used for the specified dimensions.
    std::vector<Config> getTactics(int m, int n, int k) const override;

private:
    int mQuantAlgo;
    int mGroupSize;
};

// WeightOnlyGroupwiseQuantMatmulPlugin class
// This class is responsible for implementing the groupwise quantized matrix multiplication plugin.
// It inherits from the BasePlugin class, which provides a common interface for all plugins.
class WeightOnlyGroupwiseQuantMatmulPlugin : public BasePlugin
{
public:
    // Using WeightOnlyGroupwiseQuantMatmulPlugin::PluginProfilerPtr as a type alias for the shared_ptr of the
    // WeightOnlyGroupwiseQuantGemmPluginProfiler class.
    using PluginProfilerPtr = std::shared_ptr<WeightOnlyGroupwiseQuantGemmPluginProfiler>;

    // Default constructor is deleted.
    WeightOnlyGroupwiseQuantMatmulPlugin() = delete;

    // Constructor with parameters.
    WeightOnlyGroupwiseQuantMatmulPlugin(
        nvinfer1::DataType type, int quant_algo, int group_size, const PluginProfilerPtr& profiler);

    // Constructor with serialized data and a custom profiler.
    WeightOnlyGroupwiseQuantMatmulPlugin(const void* data, size_t length, const PluginProfilerPtr& profiler);

    // Destructor.
    ~WeightOnlyGroupwiseQuantMatmulPlugin() override = default;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::
