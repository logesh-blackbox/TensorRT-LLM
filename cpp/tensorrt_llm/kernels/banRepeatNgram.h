/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
#include <cuda_runtime.h>

namespace tensorrt_llm {
namespace kernels {

/**
 * @brief Invokes the BanRepeatNgram function with the given parameters.
 *
 * @tparam T The data type of the logits and output_ids_buf.
 *
 * @param logits The logits array.
 * @param output_ids_buf The output IDs buffer.
 * @param finished_buf The finished buffer.
 * @param parent_ids_buf The parent IDs buffer.
 * @param batch_size The batch size.
 * @param local_batch_size The local batch size.
 * @param beam_width The beam width.
 * @param no_repeat_ngram_size_buf The no repeat ngram size buffer.
 * @param id_offset The ID offset.
 * @param vocab_size_padded The vocabulary size padded.
 * @param step The step.
 * @param stream The CUDA stream.
 */
template <typename T>
__global__ void banRepeatNgramKernel(T* logits, const int* output_ids_buf, const bool* finished_buf,
                                     const int* parent_ids_buf, int batch_size, int local_batch_size, int beam_width,
                                     const int* no_repeat_ngram_size_buf, int id_offset, int vocab_size_padded,
                                     int step, cudaStream_t stream) {
  int batch_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_index >= batch_size) return;

  int local_batch_index = batch_index / local_batch_size;
  int local_index = batch_index % local_batch_size;

  int* output_ids = &output_ids_buf[local_batch_index * beam_width];
  bool finished = finished_buf[local_batch_index];
  int parent_id = parent_ids_buf[local_batch_index];
  int no_repeat_ngram_size = no_repeat_ngram_size_buf[local
