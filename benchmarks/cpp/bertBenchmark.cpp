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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/worldConfig.h"

using namespace tensorrt_llm::runtime;

namespace
{

// follows https://github.com/NVIDIA/TensorRT/blob/release/8.6/samples/common/sampleEngines.cpp
std::vector<uint8_t> loadEngine(std::string const& enginePath)
{
    std::ifstream engineFile(enginePath, std::ios::binary);
    TLLM_CHECK(engineFile.good());
    engineFile.seekg(0, std::ifstream::end);
    auto const size = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<uint8_t> engineBlob(size);
    engineFile.read(reinterpret_cast<char*>(engineBlob.data()), size);
    TLLM_CHECK(engineFile.good());
    return engineBlob;
}

std::string engineFilename(
    std::filesystem::path const& dataPath, WorldConfig const& worldConfig, std::string const& model)
{
    auto constexpr allowExceptions = true;
    auto constexpr ingoreComments = true;
    auto const jsonFilePath = dataPath / "config.json";
    TLLM_CHECK(std::filesystem::exists(jsonFilePath));
    std::ifstream jsonStream(jsonFilePath);
    auto const json = nlohmann::json::parse(jsonStream, nullptr, allowExceptions, ingoreComments);
    auto const& builderConfig = json.at("builder_config");
    auto const precision = builderConfig.at("precision").template get<std::string>();
    auto const worldSize = builderConfig.at("tensor_parallel").template get<SizeType>();

    TLLM_CHECK(worldSize == worldConfig.getSize());
    return model + "_" + precision + "_tp" + std::to_string(worldConfig.getSize()) + "_rank"
        + std::to_string(worldConfig.getRank()) + ".engine";
}

void benchmarkBert(std::string const& modelName, std::filesystem::path const& dataPath,
    std::vector<int> const
