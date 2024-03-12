/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "gemmPlugin.h"

// Include necessary TensorRT headers
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cublasLt.h>

// Include custom headers for the plugin
#include "common.h"
#include "plugins.h"

// Define the namespace for the plugin
namespace tensorrt_llm::plugins {

// Define the GemmPlugin class
class GemmPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    // Constructor
    GemmPlugin(int transA, int transB, nvinfer1::DataType type, bool useFp8, const PluginProfilerPtr& pluginProfiler);

    // Parameterized constructor
    GemmPlugin(const void* data, size_t length, const PluginProfilerPtr& pluginProfiler);

    // Destructor
    ~GemmPlugin();

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(
        int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    // IPluginV2 Methods
    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    int getNbOutputs() const noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

private:
    // Helper methods
    void init();
    void setGemmConfig();
    void configGemm();

    // Member variables
    int mTransA;
    int mTransB;
    nvinfer1::DataType mType;
    bool mUseFp8;
    PluginProfilerPtr mPluginProfiler;
    CublasMMWrapperPtr mCublasWrapper;
    GemmIdCublas mGemmId;
    Dims mDims;
};

// Constructor
GemmPlugin::GemmPlugin(int transA, int transB, nvinfer1::DataType type, bool useFp8, const PluginProfilerPtr& pluginProfiler)
    : mTransA(transA)
    , mTransB(transB)
    , mType(type)
    , mUseFp8(useFp8)
    , mPluginProfiler(pluginProfiler)
{
    init();
}

// Parameterized constructor
GemmPlugin::GemmPlugin(const void* data, size_t length, const PluginProfilerPtr& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
{
    const char *d = re.interpret_cast<const char*>(data), *a = d;
    read(d, mTransA);
    read(d, mTransB);
    read(d, mType);
    read(d, mUseFp8);
    read(d, mDims);

    init();

    mPluginProfiler->deserialize(d, mDims, mGemmId);

    TLLM_CHECK(d == a + length);
}

// Destructor
GemmPlugin::~GemmPlugin()
{
    // Free any allocated resources here
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* GemmPlugin::clone() const noexcept
{
    auto* plugin = new GemmPlugin(*this);
    return plugin;
}

nvinfer1::DimsExprs GemmPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        TLLM_CHECK(nbInputs == 2);
        TLLM_CHECK(outputIndex == 0);
        const int nbDimsA =
