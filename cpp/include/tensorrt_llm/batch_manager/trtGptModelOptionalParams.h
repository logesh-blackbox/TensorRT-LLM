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

#include "tensorrt_llm/runtime/common.h"

#include <optional>

namespace tensorrt_llm::batch_manager
{

/**
 * @brief TrtGptModelOptionalParams class holds optional parameters for the GPT model.
 */
class TrtGptModelOptionalParams
{
public:
    /**
     * @brief SizeType is the data type used for sizes.
     */
    using SizeType = tensorrt_llm::runtime::SizeType;

    /**
     * @brief Default constructor.
     */
    TrtGptModelOptionalParams()
        : mMaxNumSequences(std::nullopt)
        , mMaxTokensInPagedKvCache(std::nullopt)
        , mKvCacheFreeGpuMemFraction(std::nullopt)
        , mEnableTrtOverlap(std::nullopt)
    {
    }

    /**
     * @brief Constructor with optional parameters.
     *
     * @param maxNumSequences Maximum number of sequences.
     * @param maxTokensInPagedKvCache Maximum tokens in paged KV cache.
     * @param kvCacheFreeGpuMemFraction Fraction of GPU memory to keep free for KV cache.
     * @param enableTrtOverlap Enable TRT overlap.
     */
    TrtGptModelOptionalParams(std::optional<SizeType> maxNumSequences, std::optional<SizeType> maxTokensInPagedKvCache,
        std::optional<float> kvCacheFreeGpuMemFraction, std::optional<bool> enableTrtOverlap)
        : mMaxNumSequences(maxNumSequences)
        , mMaxTokensInPagedKvCache(maxTokensInPagedKvCache)
        , mKvCacheFreeGpuMemFraction(kvCacheFreeGpuMemFraction)
        , mEnableTrtOverlap(enableTrtOverlap)
    {
    }

    /**
     * @brief Get the maximum tokens in paged KV cache.
     *
     * @return std::optional<SizeType> Maximum tokens in paged KV cache.
     */
    [[nodiscard]] std::optional<SizeType> getMaxTokensInPagedKvCache() const
    {
        return mMaxTokensInPagedKvCache;
    }

    /**
     * @brief Get the fraction of GPU memory to keep free for KV cache.
     *
     * @return std::optional<float> Fraction of GPU memory to keep free for KV cache.
     */
    [[nodiscard]] std::optional<float> getKvCacheFreeGpuMemFraction() const
    {
        return mKvCacheFreeGpuMemFraction;
    }

    /**
     * @brief Get the maximum number of sequences.
     *
     * @return std::optional<float> Maximum number of sequences.
     */
    [[nodiscard]] std::optional<float> getMaxNumSequences() const
    {
        return mMaxNumSequences;
    }

    /**
     * @brief Get the enable TRT overlap.
     *
     * @return std::optional<bool> Enable TRT overlap.
     */
    [[nodiscard]] std::optional<bool> getEnableTrtOverlap() const
    {
        return mEnableTrtOverlap;
    }

private:
    std::optional<SizeType> mMaxNumSequences;
    std::optional<SizeType> mMaxTokensInPagedKvCache;
    std::optional<float> mKvCacheFreeGpuMemFraction;
    std::optional<bool> mEnableTrtOverlap;
};

} // namespace tensorrt_llm::batch_manager

