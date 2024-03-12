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
#ifndef TOP_LEVEL_DIR
#error "Define TOP_LEVEL_DIR"
#endif

#include <gtest/gtest.h>

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/stlUtils.h"
#include "tensorrt_llm/common/tensor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/gptSession.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/utils/multiDeviceUtils.h"

#include <algorithm>
#include <filesystem>
#include <mpi.h>

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;
namespace fs = std::filesystem;

namespace
{
auto const TEST_RESOURCE_PATH = fs::path{TOP_LEVEL_DIR} / "cpp/tests/resources";
auto const ENGINGE_PATH = TEST_RESOURCE_PATH / "models/rt_engine";
auto const DATA_PATH = TEST_RESOURCE_PATH / "data";

auto const GPT_MODEL_DIR = "gpt2";
auto const GPTJ_MODEL_DIR = "gpt-j-6b";
auto const LLAMA_MODEL_DIR = "llama-7b-hf";

// Engines need to be generated using cpp/tests/resources/scripts/build_gpt_engines.py.
auto const FP32_GPT_DIR = "fp32-default";
auto const FP32_GPT_ATTENTION_DIR = "fp32-plugin";
auto const FP16_GPT_DIR = "fp16-default";
auto const FP16_GPT_ATTENTION_DIR = "fp16-plugin";
auto const FP16_GPT_ATTENTION_PACKED_DIR = FP16_GPT_ATTENTION_DIR + std::string("-packed");
auto const FP16_GPT_ATTENTION_PACKED_PAGED_DIR = FP16_GPT_ATTENTION_PACKED_DIR + std::string("-paged");

// Expected outputs need to be generated using cpp/tests/resources/scripts/generate_expected_gpt_output.py.
auto const FP32_RESULT_FILE = "output_tokens_fp32_tp1_pp1.npy";
auto const FP32_PLUGIN_RESULT_FILE = "output_tokens_fp32_plugin_tp1_pp1.npy";
auto const FP16_RESULT_FILE = "output_tokens_fp16_tp1_pp1.npy";
auto const FP16_PLUGIN_RESULT_FILE = "output_tokens_fp16_plugin_tp1_pp1.npy";
auto const FP16_PLUGIN_PACKED_PAGED_RESULT_TP1_PP4_FILE = "output_tokens_fp16_plugin_packed_paged_tp1_pp4.npy";
auto const FP16_PLUGIN_PACKED_PAGED_RESULT_TP4_PP1_FILE = "output_tokens_fp16_plugin_packed_paged_tp4_pp1.npy";
auto const FP16_PLUGIN_PACKED_PAGED_RESULT_TP2_PP2_FILE = "output_tokens_fp16_plugin_packed_paged_tp2_pp2.npy";
auto const FP16_PLUGIN_PACKED_RESULT_FILE = "output_tokens_fp16_plugin_packed_tp1_pp1.npy";
auto const FP16_PLUGIN_PACKED_PAGED_RESULT_FILE = "output_tokens_fp16_plugin_packed_paged_tp1_pp1.npy";

struct ModelIds
{
    int endId;
    int padId;
};

struct ModelParams
{
    char const* baseDir;
    ModelIds ids;
};

class ModelSpec
{
public:
    ModelSpec(fs::path modelPath, fs::path resultsFile, nvinfer1::DataType dtype)
        : mModelPath{std::move(modelPath)}
        , mResultsFile{std::move(resultsFile)}
        , mDataType{dtype}
        , mUseGptAttentionPlugin{false}
        , mUsePackedInput{false}
        , mUsePagedKvCache{false}
        , mDecoderPerRequest{false}
        , mPPSize(1)
        , mTPSize(1)
    {
    }

    ModelSpec& useGptAttentionPlugin()
    {
        mUseGptAttentionPlugin = true;
        return *this;
    }

    ModelSpec& usePackedInput()
    {
        mUsePackedInput = true;
        return *this;
    }

    ModelSpec& usePagedKvCache()
    {
        mUsePagedKvCache = true;
        return *this;
    }

    ModelSpec& useDecoderPerRequest()
    {
        mDecoderPerRequest = true;
        return *this;
   
