/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_fp16.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/penaltyTypes.h"

namespace tensorrt_llm
{
namespace kernels
{

/**
 * @brief Applies repetition penalty to the given logits.
 *
 * This function applies a repetition penalty to the given logits based on the specified penalty type.
 *
 * @tparam T The data type of the logits.
 * @param logits The logits to apply the repetition penalty to.
 * @param penalties The penalties to apply to the logits.
 * @param output_ids The output IDs of the sequence.
 * @param sequence_lengths The lengths of the sequences in the batch.
 * @param batch_size The size of the batch.
 * @param local_batch_size The local batch size.
 * @param vocab_size The size of the vocabulary.
 * @param input_lengths The lengths of the input sequences.
 * @param penalty_type The type of repetition penalty to apply.
 * @param max_seq_len The maximum sequence length.
 * @param stream The CUDA stream to execute the operation on.
 */
template <typename T>
void invokeBatchApplyRepetitionPenalty(T* logits, const float* penalties, const int* output_ids,
    const int* sequence_lengths, const int batch_size, const int local_batch_size, const int vocab_size,
    const int* input_lengths, const RepetitionPenaltyType penalty_type, int max_seq_len, cudaStream_t stream)
{
    constexpr int kNumPenalties = 2;
    constexpr int kPenaltyIndex = 0;
    constexpr int kOutputIdIndex = 1;

    const int* penalties_ptr = penalties;
    const int* output_ids_ptr = output_ids;

    if (penalty_type == RepetitionPenaltyType::kRepetitionPenalty)
    {
        // Apply repetition penalty
        for (int i = 0; i < batch_size; ++i)
        {
            const int seq_len = sequence_lengths[i];
            const int* input_len_ptr = input_lengths + i * local_batch_size;
            const int* input_len_ptr_end = input_len_ptr + local_batch_size;
            for (; input_len_ptr != input_len_ptr_end; ++input_len_ptr)
            {
                const int input_len = *input_len_ptr;
                if (input_len >= seq_len)
                {
                    break;
                }

                const int* penalties_ptr_seq = penalties_ptr + i * vocab_size * kNumPenalties;
                const int* output_ids_ptr_seq = output_ids_ptr + i * seq_len;
                for (int j = input_len; j < seq_len; ++j)
                {
                    const int id = output_ids_ptr_seq[j];
                    const float penalty = penalties_ptr_seq[id * kNumPenalties + kPenaltyIndex];
                    logits[id * seq_len + j] -= penalty;
                }
            }
        }
    }
    else if (penalty_type == RepetitionPenaltyType::kOutputIdPenalty)
    {
        // Apply output ID penalty
        for (int i = 0; i < batch_size; ++i)

