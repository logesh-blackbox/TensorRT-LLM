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

#include "decoderMaskedMultiheadAttentionLaunch.h"

namespace tensorrt_llm
{
namespace kernels
{

namespace
{
// Define the size per head as a constant
constexpr int kSizePerHead = 224;

// Define a macro to instantiate the MMHA launchers for different data types
#define INSTANTIATE_MMHA_LAUNCHERS(T, sizePerHead)                                                  \
  template void decoderMaskedMultiheadAttentionLaunch<T>(                                             \
      const int batchSize, const int sequenceLength, const int numHeads, const int headSize,       \
      const T* query, const T* key, const T* value, const T* mask, const T* bias, T* output,      \
      cudaStream_t stream, bool isLastKernelInvocation);                                            \
  template void decoderMaskedMultiheadAttentionLaunch<T>(                                             \
      const int batchSize, const int sequenceLength, const int numHeads, const int headSize,       \
      const T* query, const T* key, const T* value, const T* mask, const T* bias, T* output,      \
      cudaStream_t stream, bool isLastKernelInvocation, int64_t workspaceSize, void* workspace);  \
