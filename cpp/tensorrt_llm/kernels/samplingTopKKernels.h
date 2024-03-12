/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include <curand_kernel.h>

namespace tensorrt_llm
{
namespace kernels
{

/**
 * @brief Performs Top-K sampling on the given logits.
 *
 * This function invokes the Top-K sampling kernel for a single sequence.
 *
 * @tparam T The data type of the logits.
 * @param workspace The pointer to the workspace memory.
 * @param workspace_size The size of the workspace memory.
 * @param log_probs The pointer to the log probabilities.
 * @param ids The pointer to the output IDs.
 * @param sequence_length The length of the sequence.
 * @param finished_buf The buffer to store the finished flag.
 * @param cum_log_probs The pointer to the cumulative log probabilities.
 * @param output_log_probs The pointer to the output log probabilities.
 * @param curandstate The pointer to the curand state.
 * @param top_k The value of top-k.
 * @param top_p The value of top-p.
 * @param vocab_size_padded The padded vocabulary size.
 * @param end_ids The pointer to the end IDs.
 * @param stream The CUDA stream.
 * @param batch_size The batch size.
 * @param skip_decode A boolean flag indicating whether to skip decoding.
 */
template <typename T>
void invokeTopKSampling(void* workspace, size_t& workspace_size, const T* log_probs, int** ids, int* sequence_length,
    bool* finished_buf, float* cum_log_probs, float* output_log_probs, curandState_t* curandstate, const int top_k,
    const float top_p, const int vocab_size_padded, const int* end_ids, cudaStream_t stream, const int batch_size,
    const bool* skip_decode)
{
    // Check if the workspace size is sufficient.
    if (workspace_size < tensorrt_llm::kernels::getWorkspaceSize<T>())
    {
        throw std::runtime_error("Insufficient workspace size for TopKSampling.");
    }

    // Call the Top-K sampling kernel.
    tensorrt_llm::kernels::topKSampling<<<1, 1, 0, stream>>>(
        workspace, log_probs, ids, sequence_length, finished_buf, cum_log_probs, output_log_probs, curandstate, top_k,
        top_p, vocab_size_padded, end_ids, skip_decode);

    // Update the workspace size.
    workspace_size = tensorrt_llm::kernels::getWorkspaceSize<T>();
}

/**
 * @brief Performs batch Top-K sampling on the given logits.
 *
 * This function invokes the batch Top-K sampling kernel for multiple sequences.
 *
 * @tparam T The data type of the logits.
 * @param workspace The pointer to the workspace memory.
 * @param workspace_size The size of the workspace memory.
 * @param log_probs The pointer to the log probabilities.
 * @param ids The pointer to the output IDs.
 * @param sequence_lengths The lengths of the sequences.
 * @param finished The buffer to store the finished flags.
 * @param cum_log_probs The pointer to the cumulative log probabilities.
 * @param output_log_probs The pointer to the output log probabilities.
 *
