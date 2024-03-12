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

#include <cstdio>

#include "lookupPlugin.h"
#include "tensorrt_llm/kernels/lookupKernels.h"
#include "tensorrt_llm/plugins/common/plugin.h"

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::LookupPluginCreator;
using tensorrt_llm::plugins::LookupPlugin;

// Constants for the plugin
static const char* LOOKUP_PLUGIN_VERSION{"1"};
static const char* LOOKUP_PLUGIN_NAME{"Lookup"};

// PluginFieldCollection and PluginField for the plugin's attributes
PluginFieldCollection LookupPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> LookupPluginCreator::mPluginAttributes;

// Constructor for the LookupPlugin class
LookupPlugin::LookupPlugin(nvinfer1::DataType type, int rank)
    : mType(type)
    , mRank(rank)
{
}

// Parameterized constructor for the LookupPlugin class
LookupPlugin::LookupPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mType);
    read(d, mRank);
    TLLM_CHECK(d == a + length);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* LookupPlugin::clone() const noexcept
{
    auto* plugin = new LookupPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
}

// Implementation of getOutputDimensions method
nvinfer1::DimsExprs LookupPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        TLLM_CHECK(nbInputs == 2);
        TLLM_CHECK(outputIndex == 0);
        DimsExprs ret;
        const int nbDimsInput = inputs[0].nbDims;
        const int nbDimsWeight = inputs[1].nbDims;
        ret.nbDims = nbDimsInput + 1;

        for (int i = 0; i < nbDimsInput; ++i)
        {
            ret.d[i] = inputs[0].d[i];
        }
        ret.d[nbDimsInput] = inputs[1].d[nbDimsWeight - 1];

        return ret;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

// Implementation of supportsFormatCombination method
bool LookupPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    bool res = false;
    switch (pos)
    {
    case 0: res = ((inOut[0].type == DataType::kINT32) && (inOut[0].format == TensorFormat::kLINEAR)); break;
    case 1: res = ((inOut[1].type == mType) && (inOut[1].format == TensorFormat::kLINEAR)); break;
    case 2: res = ((inOut[2].type == mType) && (inOut[2].format == TensorFormat::kLINEAR)); break;
    default: // should NOT be here!
        res = false;
    }

    return res;
}

// Implementation of configurePlugin method
void LookupPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

// Implementation of getWorkspaceSize method
size_t LookupPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

// Implementation of enqueue method
int LookupPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // inputs
    //     input  [batchSize]
    //     weight [localVocabSize, hidden]
    // outputs
    //     embedding [batchSize, hidden]

    int batchSize = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        batchSize *= inputDesc[0].dims.d[i];
    }

    const int localVocabSize = inputDesc[1].dims.d[0];
    const int hidden = inputDesc[1].dims.d[inputDesc[1].dims.nbDims - 1];
    const int* input = re
