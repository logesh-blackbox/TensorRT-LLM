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
void invokeBatchApplyRepetitionPenalty(T* logits, const float* penalties, const int** output_ids,
    const int* sequence_lengths, const int batch_size, const int local_batch_size, const int vocab_size,
    const int* input_lengths, const RepetitionPenaltyType penalty_type, int max_seq_len, cudaStream_t stream);

/**
 * @brief Applies temperature penalty to the given logits.
 *
 * This function applies a temperature penalty to the given logits based on the specified temperature.
 *
 * @tparam T The data type of the logits.
 * @param logits The logits to apply the temperature penalty to.
 * @param bias The bias to apply to the logits.
 * @param temperature The temperature to apply to the logits.
 * @param batch_size The size of the batch.
 * @param vocab_size The size of the vocabulary.
 * @param vocab_size_padd The padded size of the vocabulary.
 * @param stream The CUDA stream to execute the operation on.
 */
template <typename T>

