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

#include "rmsnormPlugin/rmsnormPlugin.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/kernels/rmsnormKernels.h"

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::RmsnormPluginCreator;
using tensorrt_llm::plugins::RmsnormPlugin;

const char* RMSNORM_PLUGIN_VERSION = "1";
const char* RMSNORM_PLUGIN_NAME = "Rmsnorm";

PluginFieldCollection RmsnormPluginCreator::mFC;
std::vector<PluginField> RmsnormPluginCreator::mPluginAttributes;

RmsnormPlugin::RmsnormPlugin(float eps, DataType type)
    : mEps(eps)
    , mType(type)
{
    TLLM_CHECK_WITH_INFO((getSMVersion() >= 80) || (mType != DataType::kBF16),
        "Unsupported data type, pre SM 80 GPUs do not support bfloat16");
}

RmsnormPlugin::RmsnormPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mEps);
    read(d, mType);
    TLLM_CHECK(d == a + length);
    TLLM_CHECK_WITH_INFO((getSMVersion() >= 80) || (mType != DataType::kBF16), "Unsupported data type");
}

IPluginV2DynamicExt* RmsnormPlugin::clone() const noexcept
{
    auto* plugin = new RmsnormPlugin(mEps, mType);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

DimsExprs RmsnormPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    return inputs[outputIndex];
}

bool RmsnormPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    TLLM_CHECK(0 <= pos && pos < 5);
    return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void RmsnormPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs,
    const DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t RmsnormPlugin::getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs,
    const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int RmsnormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int m = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims - 1; ++i)
    {
        m *= inputDesc[0].dims.d[i];
    }
    const int n = inputDesc[1].dims.d[0];

    if (mType == DataType::kHALF)
    {
        const half* input = reinterpret_cast<const half*>(inputs[0]);
        const half* weight = reinterpret_cast<const half*>(inputs[1]);
        half* output = reinterpret_cast<half*>(outputs[0]);
        invokeGeneralRmsNorm(output, input, weight, (half*) nullptr, mEps, m, n, stream);
    }
    else if (mType == DataType::kFLOAT)
    {
        const float* input = reinterpret_cast<const float*>(inputs[0]);
        const float* weight = reinterpret_cast<const float*>(inputs[1]);
        float* output = reinterpret_cast<float*>(outputs[0]);
        invokeGeneralRmsNorm(output, input, weight, (float*) nullptr, mEps, m, n, stream);
    }

    return 0;
}
