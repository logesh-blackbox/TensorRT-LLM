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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <optional>
#include <utility>

namespace tensorrt_llm::runtime
{

class GenerationInput
{
public:
    using TensorPtr = ITensor::SharedPtr;

    GenerationInput(
        SizeType const endId, SizeType const padId, TensorPtr ids, TensorPtr lengths, bool packed = false)
        : endId{endId}
        , padId{padId}
        , ids{std::move(ids)}
        , lengths{std::move(lengths)}
        , packed{packed}
        , maxNewTokens{}
        , embeddingBiasOpt{}
        , badWordsList{}
        , stopWordsList{}
    {
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->ids), "Invalid ids tensor");
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->lengths), "Invalid lengths tensor");
    }

    // mandatory parameters
    SizeType endId;
    SizeType padId;
    TensorPtr ids;     // [1, packedLength] or [batchSize, maxInputLength], on gpu
    TensorPtr lengths; // [batchSize], on gpu
    bool packed;       // indicates if ids are packed or padded to maxInputLength

    // optional parameters
    std::optional<SizeType> maxNewTokens; // max number of tokens to generate
    std::optional<TensorPtr> embeddingBiasOpt;           // [vocabSizePadded], on gpu
    std::optional<TensorPtr> badWordsList;               // [2, badWordsLength] or [batchSize, 2, badWordsLength], on gpu
    std::optional<TensorPtr> stopWordsList;              // [batchSize, 2, stopWordsLength], on gpu
};

} // namespace tensorrt_llm::runtime

