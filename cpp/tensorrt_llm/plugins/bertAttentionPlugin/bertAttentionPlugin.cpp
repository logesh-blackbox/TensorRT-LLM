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
#include "bertAttentionPlugin.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;

using tensorrt_llm::plugins::BertAttentionPluginCreator;
using tensorrt_llm::plugins::BertAttentionPlugin;

static const char* BERT_ATTENTION_PLUGIN_VERSION{"1"};
static const char* BERT_ATTENTION_PLUGIN_NAME{"BertAttention"};
PluginFieldCollection BertAttentionPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> BertAttentionPluginCreator::mPluginAttributes;

BertAttentionPlugin::BertAttentionPlugin(int num_heads, int head_size, float q_scaling, bool qk_half_accum,
    ContextFMHAType context_fmha_type, nvinfer1::DataType type, bool do_relative_attention, int max_distance)
    : mNumHeads(num_heads)
    , mHeadSize(head_size)
    , mQScaling(q_scaling)
    , mQKHalfAccum(qk_half_accum)
    , mEnableContextFMHA(context_fmha_type != ContextFMHAType::DISABLED)
    , mFMHAForceFP32Acc(context_fmha_type == ContextFMHAType::ENABLED_WITH_FP32_ACC)
    , mType(type)
    , mRelativeAttention(do_relative_attention)
    , mMaxDistance(max_distance)
{
    // pre-check whether FMHA is supported in order to save memory allocation
    mEnableContextFMHA = mEnableContextFMHA && (mType == DataType::kHALF) && MHARunner::fmha_supported(mHeadSize, mSM);
}

// Parameterized constructor
BertAttentionPlugin::BertAttentionPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mNumHeads);
    read(d, mHeadSize);
    read(d, mQScaling);
    read(d, mQKHalfAccum);
    read(d, mEnableContextFMHA);
    read(d, mFMHAForceFP32Acc);
    read(d, mType);
    read(d, mRelativeAttention);
    read(d, mMaxDistance);
    TLLM_CHECK(d == a + length);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* BertAttentionPlugin::clone() const noexcept
{
    auto* plugin = new BertAttentionPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
}

nvinfer1::DimsExprs BertAttentionPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    TLLM_CHECK(outputIndex == 0);
    auto ret = inputs[0];
    ret.d[2] = exprBuilder.constant(ret.d[2]->getConstantValue() / 3);
    return ret;
}

bool BertAttentionPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (pos == 1)
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32;
    }
    else
    {
        return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
}

void BertAttentionPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t BertAttentionPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    const int batch_size = inputs[0].dims.d[0];
    const int input_seq_len = inputs[0].dims.d[1];
    const int local_hidden_units_ = inputs[0].dims.d[2] / 3;
    const int beam_width = 1;
    const int max_input_length = input_seq_len;

    size_t size{0U};
    if (inputs[0].type == DataType::kHALF)
    {
        size = sizeof(half);
    }
    else if (inputs[0].type == DataType::kFLOAT)
    {
        size = sizeof(float);
    }

    const size_t attention
