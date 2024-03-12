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

#include <cstddef>
#include <stdint.h>
#include <vector>

#include "tensorrt_llm/common/cudaUtils.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

enum class QuantType
{
    INT8_WEIGHT_ONLY,
    PACKED_INT4_WEIGHT_ONLY
};

// Added this function to get the number of bits in a QuantType
int get_bits_in_quant_type(QuantType quant_type)
{
    switch (quant_type)
    {
    case QuantType::INT8_WEIGHT_ONLY:
        return 8;
    case QuantType::PACKED_INT4_WEIGHT_ONLY:
        return 4;
    default:
        throw std::invalid_argument("Invalid QuantType");
    }
}

// Shapes here can be 2 or 3D. 2-D shapes are [num_rows, num_cols]
// 3-D shapes are [num_experts, num_rows, num_cols]
void permute_B_rows_for_mixed_gemm(int8_t* permuted_quantized_tensor, const int8_t* quantized_tensor,
    const std::vector<size_t>& shape, QuantType quant_type, const int64_t arch_version)
{
    // Added a check for the architecture version to ensure that it is supported
    if (arch_version != 70 && arch_version != 80)
    {
        throw std::invalid_argument("Unsupported architecture version");
    }

    // Added a check for the shape size to ensure that it is either 2 or 3
    if (shape.size() != 2 && shape.size() != 3)
    {
        throw std::invalid_argument("Invalid shape size");
    }

    // Added a check for the quantization type to ensure that it is supported
    if (quant_type != QuantType::INT8_WEIGHT_ONLY && quant_type != QuantType::PACKED_INT4_WEIGHT_ONLY)
    {
        throw std::invalid_argument("Unsupported quantization type");
    }

    // Added error checking for the tensor pointers
    if (quantized_tensor == nullptr || permuted_quantized_tensor == nullptr)
    {
        throw std::invalid_argument("Null tensor pointer");
    }

    // Added a check for the tensor sizes to ensure that they match the shape
    if (quantized_tensor.size() != get_num_elements(shape) || permuted_quantized_tensor.size() != get_num_elements(shape))
    {
        throw std::invalid_argument("Invalid tensor size");
    }

    // Added code to handle the 2D and 3D cases separately
    if (shape.size() == 2)
    {
        // 2D case
        // Added error checking for the tensor strides
        if (quantized_tensor.stride(0) != shape[1] || quantized_tensor.stride(1) != 1 ||
            permuted_quantized_tensor.stride(0) != shape[0] || per
