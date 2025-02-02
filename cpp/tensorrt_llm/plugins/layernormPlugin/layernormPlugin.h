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

#include "tensorrt_llm/plugins/common/plugin.h"
#include <cassert>
#include <mpi.h>
#include <set>
#include <string>
#include <vector>

namespace tensorrt_llm::plugins
{

class LayernormPlugin : public BasePlugin
{
public:
    // Constructor with epsilon, useDiffOfSquares, and data type as parameters
    LayernormPlugin(float eps, bool useDiffOfSquares, nvinfer1::DataType type);

    // Constructor with serialized data
    LayernormPlugin(const void* data, size_t length);

    // Destructor
    ~LayernormPlugin() override = default;

    // IPluginV2DynamicExt Methods
    // Clone the plugin
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    // Get output dimensions based on input dimensions and expression builder
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    // Check if the plugin supports the given format combination
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;

    // Configure the plugin with input and output descriptions
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;

    // Get the workspace size required for the plugin
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;

    // Enqueue the plugin for execution with given inputs, outputs, workspace, and stream
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext Methods
    // Get the output data type based on input data types
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

    // Get the serialization size
    size_t getSerializationSize() const noexcept override;

    // Serialize the plugin
    void serialize(void* buffer) const noexcept override;

    // Destroy the plugin
    void destroy() noexcept override;

private:
    float mEps;
    bool mUseDiffOfSquares;
    nvinfer1::DataType mType;

    // Layer name
    const std::string mLayerName;
};

class LayernormPluginCreator : public BaseCreator
{
public:
    // Constructor
    LayernormPluginCreator();

    // Get the plugin name
    const char* getPluginName() const noexcept override;

    // Get the plugin version
    const char* getPluginVersion() const noexcept override;

    // Get the plugin field names
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    // Create the plugin with given name and field collection
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    // Deserialize the plugin with given name, serialized data, and length
    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    // Plugin field collection and attributes
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

} // namespace tensorrt_llm::plugins
