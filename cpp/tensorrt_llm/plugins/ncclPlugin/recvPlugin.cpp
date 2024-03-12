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

#include "recvPlugin.h"

using namespace nvinfer1;
using tensorrt_llm::plugins::RecvPluginCreator;
using tensorrt_llm::plugins::RecvPlugin;

// Constants
const char* RECV_PLUGIN_VERSION = "1";
const char* RECV_PLUGIN_NAME = "Recv";

// PluginFieldCollection and PluginField declarations
static PluginFieldCollection gPluginFieldsCollection{};
static std::vector<PluginField> gPluginAttributes{
    PluginField("src_rank", nullptr, PluginFieldType::kINT32, 1),
    PluginField("type_id", nullptr, PluginFieldType::kINT32, 1)
};

// RecvPlugin class
RecvPlugin::RecvPlugin(int srcRank, DataType type)
    : mSrcRank(srcRank)
    , mType(type)
{
}

RecvPlugin::RecvPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mType);
    read(d, mSrcRank);
    TLLM_CHECK(d == a + length);
}

nvinfer1::IPluginV2DynamicExt* RecvPlugin::clone() const noexcept
{
    auto* plugin = new RecvPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs RecvPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    return inputs[0];
}

bool RecvPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void RecvPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t RecvPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int RecvPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    if (isBuilding())
    {
        return 0;
    }
    int size = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        size *= inputDesc[0].dims.d[i];
    }
    NCCLCHECK(ncclRecv(outputs[0], size, (*getDtypeMap())[inputDesc[0].type], 0, mComm, stream));

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType RecvPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return inputTypes[0];
}

// IPluginV2 Methods

const char* RecvPlugin::getPluginType() const noexcept
{
    return RECV_PLUGIN_NAME;
}

const char* RecvPlugin::getPluginVersion() const noexcept
{
    return RECV_PLUGIN_VERSION;
}

int RecvPlugin::getNbOutputs() const noexcept
{
    return 1
