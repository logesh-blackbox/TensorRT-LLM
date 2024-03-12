/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/runtime/common.h"
#include <NvInferRuntime.h>

namespace tensorrt_llm::runtime
{

class GptModelConfig
{
public:
    enum class ModelVariant : std::int32_t
    {
        kGpt = 0,
        kGlm = 1, // https://github.com/THUDM/GLM
    };

    constexpr explicit GptModelConfig(
        SizeType vocabSize, SizeType nbLayers, SizeType nbHeads, SizeType hiddenSize, nvinfer1::DataType dtype,
        bool useGptAttentionPlugin = false, bool inputPacked = false, bool pagedKvCache = false,
        SizeType tokensPerBlock = 64, common::QuantMode quantMode = common::QuantMode::none(),
        SizeType maxBatchSize = 0, SizeType maxInputLen = 0, SizeType maxOutputLen = 0,
        std::optional<SizeType> maxNumTokens = std::nullopt, bool computeContextLogits = false,
        ModelVariant modelVariant = ModelVariant::kGpt, bool useCustomAllReduce = false)
        : mVocabSize(vocabSize)
        , mNbLayers(nbLayers)
        , mNbHeads(nbHeads)
        , mNbKvHeads(nbHeads)
        , mHiddenSize(hiddenSize)
        , mDataType(dtype)
        , mUseGptAttentionPlugin(useGptAttentionPlugin)
        , mInputPacked(inputPacked)
        , mPagedKvCache(pagedKvCache)
        , mTokensPerBlock(tokensPerBlock)
        , mQuantMode(quantMode)
        , mMaxBatchSize(maxBatchSize)
        , mMaxInputLen(maxInputLen)
        , mMaxOutputLen(maxOutputLen)
        , mMaxNumTokens(maxNumTokens)
        , mComputeContextLogits(computeContextLogits)
        , mModelVariant(modelVariant)
        , mUseCustomAllReduce(useCustomAllReduce)
    {
    }

    [[nodiscard]] SizeType constexpr getVocabSize() const noexcept
    {
        return mVocabSize;
    }

    [[nodiscard]] SizeType constexpr getVocabSizePadded(SizeType worldSize) const noexcept
    {
        return (mVocabSize + worldSize - 1) / worldSize * worldSize;
    }

    [[nodiscard]] SizeType constexpr getNbLayers(SizeType pipelineParallelism = 1) const
    {
        TLLM_CHECK(mNbLayers % pipelineParallelism == 0);
        return mNbLayers / pipelineParallelism;
    }

    [[nodiscard]] SizeType constexpr getNbHeads() const noexcept
    {
        return mNbHeads;
    }

    [[nodiscard]] SizeType constexpr getNbKvHeads() const noexcept
    {
        return mNbKvHeads;
    }

    void constexpr setNbKvHeads(SizeType nbKvHeads) noexcept
    {
        mNbKvHeads = nbKvHeads;
    }

    [[nodiscard]] SizeType constexpr getHiddenSize() const noexcept
    {
        return mHiddenSize;
    }

    [[nodiscard]] SizeType constexpr getSizePerHead() const noexcept
    {
        return mHiddenSize / mNbHeads;
    }

    [[nodiscard]] SizeType constexpr getTokensPerBlock() const noexcept
    {
        return mTokensPerBlock;
    }

    void constexpr setTokensPerBlock(SizeType TokensPerBlock) noexcept
    {
        mTokensPerBlock = TokensPerBlock;
    }

    [[nodiscard]] nvinfer1::DataType constexpr getDataType() const noexcept
    {
        return mDataType;
    }

    [[nodiscard]] bool constexpr useGptAttentionPlugin() const noexcept
    {
        return mUseGptAttentionPlugin;
    }

    void constexpr useGptAttentionPlugin(bool useGptAttentionPlugin) noexcept
    {
        mUseGptAttentionPlugin = useGptAttentionPlugin;
    }

    [[nodiscard]] bool constexpr usePackedInput() const noexcept
    {
        return mInputPacked;
    }

    void constexpr usePackedInput(bool inputPacked) noexcept
    {
        mInputPacked = inputPacked;
    }

    [[nodiscard]] bool constexpr usePagedKvCache() const noexcept
    {
        return mPagedKvCache;
    }

    void constexpr usePagedKvCache(bool pagedKvCache) noexcept
    {
        mPagedKvCache = pagedKvCache;
    }

    [[nodiscard]] common::QuantMode constexpr getQuantMode() const noexcept
    {
        return mQuantMode;
    }

    void constexpr setQuantMode(common::QuantMode QuantMode) noexcept
    {
        mQuantMode = QuantMode;
    }

    [[nod
