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

#include "sendPlugin.h"

// Include required TensorRT headers
#include <cuda_runtime_api.h>
#include <nccl.h>
#include <plugin/plugin_common.h>
#include <plugin/plugin_factory.h>
#include <plugin/plugin_impl.h>
#include <plugin/plugin_tensor_desc.h>
#include <plugin/plugin_util.h>
#include <vector>

// Declare the namespace for the plugin
using namespace nvinfer1;
using tensorrt_llm::plugins::SendPluginCreator;
using tensorrt_llm::plugins::SendPlugin;

// Define the plugin version and name
static const char* SEND_PLUGIN_VERSION{"1"};
static const char* SEND_PLUGIN_NAME{"Send"};

// Define the PluginFieldCollection and PluginField for the plugin constructor
PluginFieldCollection SendPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> SendPluginCreator::mPluginAttributes;

// Define the SendPlugin class
class SendPlugin : public IPluginV2DynamicExt
{
public:
    // Constructor for the plugin
    SendPlugin(int tgtRank, nvinfer1::DataType type);

    // Parameterized constructor for the plugin
    SendPlugin(const void* data, size_t length);

    // IPluginV2DynamicExt methods
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

    // IPluginV2Ext methods
    nvinfer1::DataType getOutputDataType(
        int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    // IPluginV2 methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;

private:
    // Helper functions for reading and writing data
    template <typename T>
    void read(const char*& d, T& value);
    template <typename T>
    void write(char*& d, const T& value);

    // Data members for the plugin
    int mTgtRank;
    nvinfer1::DataType mType;
    ncclComm_t mComm;
};

// Define the SendPluginCreator class
class SendPluginCreator : public IPluginCreator
{
public:
    // Constructor for the plugin creator
    SendPluginCreator();

    // IPluginCreator methods
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    // Data member for the plugin namespace
    std::string mNamespace;
};

// Implement the SendPlugin class methods

// Constructor for the plugin
SendPlugin::SendPlugin(int tgtRank, nvinfer1::DataType type)
    : mTgtRank(tgtRank)
    , mType(type)
{
}

// Parameterized constructor for the plugin
SendPlugin::SendPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mType);
    read(d, mTgtRank);
    TLLM_CHECK(d == a + length);
}

// IPluginV2DynamicExt methods
nvinfer1::IPluginV2DynamicExt* SendPlugin::clone() const noexcept
{
    auto* plugin = new SendPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs SendPlugin::getOutputDimensions(
    int outputIndex, const
