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

#include <gtest/gtest.h>

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/gptDecoder.h"
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;

namespace
{

void testDecoder(nvinfer1::DataType const dtype, SamplingConfig const& samplingConfig)
{
    constexpr SizeType tensorParallelism{1};
    constexpr SizeType pipelineParallelism{1};
    constexpr SizeType localRank{0};
    const WorldConfig worldConfig{tensorParallelism, pipelineParallelism, localRank};

    constexpr SizeType vocabSize{51200};
    constexpr SizeType nbLayers{2};
    constexpr SizeType nbHeads{16};
    constexpr SizeType hiddenSize{1024};
    GptModelConfig modelConfig{vocabSize, nbLayers, nbHeads, hiddenSize, dtype};
    modelConfig.useGptAttentionPlugin(false);

    auto streamPtr = std::make_shared<CudaStream>();
    BufferManager manager(streamPtr);

    // create decoder
    constexpr SizeType vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());
    auto decoder = IGptDecoder::create(modelConfig.getDataType(), vocabSize, vocabSizePadded, streamPtr);
    ASSERT_TRUE(static_cast<bool>(decoder));

    // setup decoder
    constexpr SizeType beamWidth = samplingConfig.beamWidth;
    constexpr SizeType batchSize{4};

    decoder->setup(samplingConfig, batchSize);

    constexpr int endId{50257};
    constexpr SizeType maxInputLength{8};
    constexpr SizeType maxNewTokens{2};
    constexpr SizeType maxSeqLength = maxInputLength + maxNewTokens;

    // set up inputs
    auto logits = manager.gpu<float>(tc::ITensor::makeShape({batchSize, beamWidth, vocabSizePadded}));
    manager.setZero(*logits);

    std::vector<int> endIdsVec(batchSize * beamWidth, endId);
    auto endIds = manager.copyFrom(endIdsVec, tc::ITensor::makeShape({batchSize, beamWidth}), MemoryType::kGPU);

    DecodingInput inputs{maxInputLength, batchSize, logits, endIds};
    std::vector<std::int32_t> sequenceLimitLengthsVec(batchSize, maxSeqLength);
    inputs.sequenceLimitLength = manager.copyFrom(sequenceLimitLengthsVec, tc::ITensor::makeShape({batchSize}), MemoryType::kGPU);

    if (beamWidth > 1)
    {
        auto srcCacheIndirection = manager.gpu<int32_t>(tc::ITensor::makeShape({batchSize, beamWidth, maxSeqLength}));
        manager.setZero(*srcCacheIndirection);
        inputs.cacheIndirection = srcCacheIndirection;
    }

    // set up outputs
    auto outputIds = manager.gpu<int32_t>(tc::ITensor::makeShape({batchSize, beamWidth, maxSeqLength}));
    manager.setZero(*outputIds);
    DecodingOutput outputs{outputIds};
    auto newTokens = manager.gpu<int32_t>(tc::ITensor::makeShape({batchSize, beamWidth}));
    manager.setZero(*newTokens);
    outputs.newTokens = newTokens;

    std::vector<int> sequenceLengthsVec(batchSize * beamWidth, maxInputLength);
    outputs.lengths = manager.copyFrom(sequenceLengthsVec, tc::ITensor::makeShape({batchSize, beamWidth}), MemoryType::kGPU);
    outputs.finished = manager.gpu<bool>(tc::ITensor::makeShape({batchSize, beamWidth}));
    manager.setZero(*outputs.finished);
    outputs.finishedSum = BufferManager::pinned<int32_t>(tc::ITensor::makeShape({1}));
    auto* finishedSumHost = bufferCast<std::int32_t>(*outputs.finishedSum);
    *finishedSumHost = -1;

    if (beamWidth > 1)
    {
        auto tgtCacheIndirection = manager.gpu<int32_t>(tc::ITensor::makeShape({batchSize, beamWidth, maxSeqLength}));
        manager.setZero(*tgtCacheIndirection);
        outputs.cacheIndirection = tgtCacheIndirection;

        auto cumLogProbs = manager.gpu<float>(tc::ITensor::makeShape({batchSize, beamWidth}));
        manager.setZero(*cumLogProbs);
        outputs.cumLogProbs = cumLogProbs;

        auto parentIds = manager.gpu<int32_t>(tc::ITensor::makeShape({batchSize, beamWidth, maxSeqLength}));
        manager.setZero(*parentIds);
        outputs.parentIds = parentIds;
    }

    // run decoder
    EXPECT_FALSE(decoder->forward(outputs, inputs));
    inputs.step += 1;
    EXPECT_EQ(*
