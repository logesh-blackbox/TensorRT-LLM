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
#include "tensorrt_llm/plugins/common/plugin.h"

// Include necessary headers for assert, memory, set, string, and vector.
#include <cassert>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace tensorrt_llm::plugins
{

// Declare the QuantizePerTokenPlugin class, which inherits from BasePlugin.
class QuantizePerTokenPlugin : public BasePlugin
{
public:
    // Default constructor.
    QuantizePerTokenPlugin();

    // Constructor with data and length parameters.
    QuantizePerTokenPlugin(const void* data, size_t length);

    // Destructor.
    ~QuantizePerTokenPlugin() override = default;

    // Implement IPluginV2DynamicExt methods.
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // Implement IPluginV2Ext methods.
    nvinfer1::DataType getOutputDataType(
        int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    // Implement IPluginV2 methods.
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;

private:
    // Declare the mLayerName member variable.
    const std::string mLayerName;
};

// Declare the QuantizePerTokenPluginCreator class, which inherits from BaseCreator.
class QuantizePerTokenPluginCreator : public BaseCreator
{
public:
    // Default constructor.
    QuantizePerTokenPluginCreator();

    // Implement IPluginCreator methods.
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    // Declare the mFC and mPluginAttributes member variables.
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

} // namespace tensorrt_llm::plugins
