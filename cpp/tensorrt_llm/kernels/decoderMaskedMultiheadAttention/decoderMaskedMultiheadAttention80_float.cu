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
constexpr int kSizePerHead = 80;
} // namespace

namespace mmha
{

template <typename T>
void launchDecoderMaskedMultiheadAttention(const int batchSize, const int seqLength, const int numHeads, const int headSize, const T* input, const T* mask, const T* query, const T* key, const T* value, T* output, T* workspace, cudaStream_t stream)
{
    launch<T, kSizePerHead>(batchSize, seqLength, numHeads, headSize, input, mask, query, key, value, output, workspace, stream);
}

INSTANTIATE_MMHA_LAUNCHERS(float, kSizePerHead)

} // namespace mmha

} // namespace kernels
} // namespace tensorrt_llm

