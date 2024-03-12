/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include "checkMacrosPlugin.h"
#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaRunner.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include "tensorrt_llm/plugins/gptAttentionCommon/gptAttentionCommon.h"
#include <cassert>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace tensorrt_llm::plugins
{
// batch_size = num_ctx_requests + num_gen_requests * beam_width
// num_ctx_requests = number of context requests (single sequence per request).
// num_gen_requests = number of generation requests (beam_width sequences per request).
// Context sequences have to appear first, generation sequences after

// inputs
//     0.  input_tensor [batch_size, seq_len, local_hidden_size + 2 * local_num_kv_heads * head_size] or
//                      [1, num_tokens, local_hidden_size + 2 * local_num_kv_heads * head_size] when
//                      enable_remove_input_padding
//     1.  sequence_length [batch_size]
//     2.  host_past_key_value_lengths [batch_size] (int32)
//     3.  context_lengths [batch_size]
//     4.  cache_indir [num_gen_requests, beam_width, memory_max_len] (required in beamsearch)
//     5.  host_request_types [batch_size] int32. 0: context; 1: generation: 2: none. When not in inflight-batching
//     mode,
//                      all elements must be identical.
//     6.  past_key_value_pool [batch_size, 2, local_num_kv_heads, max_seq_len, head_size] or
//         block_pointers [batch_size, 2, max_blocks_per_seq] if paged kv cache
//     7.  kv_cache_quantization_scale [1] (optional)
//     8.  kv_cache_dequantization_scale [1] (optional)
//     9.  alibi_slopes [num_heads] (optional for ALiBi position embedding)
//     10. host_context_lengths [batch_size] int32. (optional, required when remove_input_padding is true)
//     11. qkv_bias (optional) [local_hidden_size * 3]
//
// outputs
//     output_tensor [batch_size, seq_len, local_hidden_size]
//     present_key_value_pool (optional if not paged kv cache) [batch_size, 2, local_num_kv_heads, max_seq_len,
//     head_size]

// This class implements the GPTAttentionPlugin, which is a plugin for the TensorRT framework.
// The plugin is used to perform multi-head attention operations in the GPT language model.
// The plugin takes in a batch of sequences, and for each sequence, it computes the attention
// weights between each token in the sequence and all other tokens in the sequence.
// The attention weights are then used to compute a weighted sum of the values of the tokens,
// which is used as the output for the sequence.
// The plugin also supports the use of a key-value cache, which can be used to speed up the
// computation of the attention weights for sequences that have been seen before.
// The plugin is implemented as a template class, with the template parameter specifying the
// data type of the input and output tensors.
class GPTAttentionPlugin : public GPTAttentionPluginCommon
{
public:
    // The constructor takes in various parameters that are used to configure the behavior of
    // the plugin.
    // The parameters include the number of attention heads, the size of each attention head,
    // whether the attention is unidirectional, the scaling factor for the queries, the type
    // of position embedding to use, the dimension and base for the rotary embedding, the
    // type of rotary scaling to use, the scale factor for the rotary embedding, the maximum
    // number of positions for the rotary embedding, the size and rank of the token padding,
    // the type of context fused multi-head attention to use, whether to use multi-block mode,
    // the quantization mode for the key-value cache, whether to remove input padding, the
    // type of attention mask to use, whether to use a paged key-value cache, the number of
    // tokens per block, the data type of the input and output tensors, the maximum context
    // length, whether to enable the query-key-value bias, and whether to use cross attention.
    GPTAttentionPlugin(int num_heads, int num_kv_heads, int head_size, int unidirectional, float q_scaling,
        tensorrt_llm::kernels::PositionEmbeddingType position_
