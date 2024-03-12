/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <NvInferRuntime.h>

#include <cassert>
#include <set>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

namespace openai_triton::plugin
{

class TritonFlashAttentionPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    // Constructor for TritonFlashAttentionPlugin
    TritonFlashAttentionPlugin(int numHeads, int headSize, float softmaxScale, nvinfer1::DataType type);

    // Constructor for TritonFlashAttentionPlugin that accepts a serialized buffer
    TritonFlashAttentionPlugin(const void* data, size_t length);

    // Destructor for TritonFlashAttentionPlugin
    ~TritonFlashAttentionPlugin() override = default;

    // IPluginV2DynamicExt Methods
    // Clone the plugin
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    // Get the output dimensions for a given input
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    // Check if the plugin supports a given format combination
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;

    // Configure the plugin with input and output descriptions
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;

    // Get the workspace size required for the plugin
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;

    // Enqueue the plugin for execution
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // Implementation of enqueue for specific data types
    template <typename T>
    int enqueueImpl(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

    // IPluginV2Ext Methods
    // Get the output data type for a given input data type
    nvinfer1::DataType getOutputDataType(
        int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    // IPluginV2 Methods
    // Get the plugin type
    const char* getPluginType() const noexcept override;

    // Get the plugin version
    const char* getPluginVersion() const noexcept override;

    // Get the number of outputs
    int getNbOutputs() const noexcept override;

    // Initialize the plugin
    int initialize() noexcept override;

    // Terminate the plugin
    void terminate() noexcept override;

    // Get the serialization size of the plugin
    size_t getSerializationSize() const noexcept override;

    // Serialize the plugin
    void serialize(void* buffer) const noexcept override;

    // Destroy the plugin
    void destroy() noexcept override;

    // Set the plugin namespace
    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    // Get the plugin namespace
    const char* getPluginNamespace() const noexcept override;

private:
    // Layer name
    const std::string mLayerName;

    // Namespace
    std::string mNamespace;

    // Number of heads
    int mNumHeads;

    // Head size
    int mHeadSize;

    // Softmax scale
    float mSoftmaxScale;

    // Data type
    nvinfer1::DataType mType;

    // CUDA module
    CUmodule mModule;

    // CUDA function
    CUfunction mKernel;
};

// Creator for TritonFlashAttentionPlugin
class TritonFlashAttentionPluginCreator : public nvinfer1::IPluginCreator
{
public:
    // Constructor for TritonFlashAttentionPluginCreator
    TritonFlashAttentionPluginCreator();

    // Get the plugin name
    const char* getPluginName() const noexcept override;

    // Get the plugin version
    const char* getPluginVersion() const noexcept override;

    // Get the plugin field names
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    // Create the plugin
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    // Deserialize the plugin
    nvinfer1::IPluginV2* deserializePlugin(
       
