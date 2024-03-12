/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/plugins/common/gemmPluginProfiler.h"
#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "tensorrt_llm/kernels/cutlass_kernels/int8_gemm/int8_gemm.h"

namespace tensorrt_llm::plugins
{

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::GemmPluginProfiler()
{
    // Initialize the MNKProfileMap shared pointer
    mMNKProfileMap = std::make_shared<MNKProfileMap>();

    // Set the skip flag based on the environment variable
    const auto skipEnv = std::getenv("SKIP_GEMM_PLUGIN_PROFILINGS");
    mSkip = (skipEnv != NULL && std::stoi(skipEnv));
    if (mSkip)
    {
        // Log a debug message if the SKIP_GEMM_PLUGIN_PROFILINGS environment variable is set
        TLLM_LOG_DEBUG(
            "SKIP_GEMM_PLUGIN_PROFILINGS is set. Skipping GEMM plugin profilings. It could result in runtime error "
            "if default tactic is not defined.");
    }
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::serialize(
    char*& buffer, const GemmIdType& gemmId) const
{
    // Get the MProfileMap for the given gemmId
    auto mProfileMap = mMNKProfileMap->getMProfileMap(gemmId);

    // Write the size of the profile map to the buffer
    write(buffer, static_cast<int>(mProfileMap->size()));

    // Iterate over the entries in the profile map and write them to the buffer
    for (const auto& pair : *mProfileMap)
    {
        // Write the pair of M to the best GEMM config
        write(buffer, pair);
    }
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::deserialize(
    const char*& data, GemmDims& dims, const GemmIdType& gemmId)
{
    // Lock the mutex for the MNKProfileMap
    writer_lock lock(mMNKProfileMap->mutex);

    // Set the dims member variable
    mDims = dims;

    // Create the MProfileMap for the given gemmId if it doesn't already exist
    if (!mMNKProfileMap->existsMProfileMap(gemmId))
    {
        mMNKProfileMap->createMProfileMap(gemmId);
    }

    // Populate the MProfileMap for the given gemmId
    auto profileMap = mMNKProfileMap->getMProfileMap(gemmId);

    // Read the size of the selected map from the data
    int selectedMapSize;
    read(data, selectedMapSize);

    // Iterate over the entries in the selected map and insert them into the profileMap
    for (int ii = 0; ii < selectedMapSize; ++ii)
    {
        std::pair<int, std::optional<Config>> config;
        read(data, config);
        profileMap->insert(config);
    }
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
size_t GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::getSerializationSize(
    const GemmIdType& gemmId) const
{
    reader_lock lock(mMNKProfileMap->mutex);

    // Calculate the size of the serialized data for the given gemmId
    return sizeof(int) +                                 // size of the tactics map
        mMNKProfileMap->getMProfileMap(gemmId)->size()
        * sizeof(std::pair<int, std::optional<Config>>); // size of the tactics map
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::profileTactics(
    const RunnerPtr& runner, const nvinfer1::DataType& type, const GemmDims& dims, const GemmIdType& gemmId)
{
    // Lock the mutex for the MNKProfileMap
    writer_lock lock(mMNKProfileMap->mutex);

    // Set the member variables for the runner, type, and dims
    mRunner = runner;
    mType = type;

    // Calculate the maximum M value and allocate temporary workspace for GEMMs
    const int maxM = std::min(
