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

#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/thop/thUtils.h"

// Check if the Torch version is at least v1.9.0
#if defined(TORCH_VERSION_MAJOR) && ((TORCH_VERSION_MAJOR > 1) || ((TORCH_VERSION_MAJOR == 1) && (TORCH_VERSION_MINOR >= 9)))
#define TORCH_IS_AT_LEAST_v190
#endif

namespace torch_ext
{
// Use torch::Tensor for input and output tensors.
using torch::Tensor;
// Import the QuantizeMode enum from tensorrt_llm::common namespace.
using namespace tensorrt_llm::common;

// Function to quantize the input tensor using e4m3 format.
// The quantization mode can be PER_TOKEN, PER_CHANNEL, or PER_TENSOR.
std::vector<Tensor> e4m3_quantize_helper(Tensor input, QuantizeMode quantize_mode)
{
    // Check if the input tensor is contiguous and has a non-zero number of elements.
    CHECK_CONTIGUOUS(input);
    TORCH_CHECK(input.numel() != 0, "input should not be empty tensor");
    // Check if the input tensor dimension is valid based on the quantization mode.
    TORCH_CHECK(input.dim() >= 2 && (quantize_mode != QuantizeMode::PER_CHANNEL || input.dim() == 2),
        "Invalid dim. The dim of input should be greater than or equal to 2");

    // Get the scalar type of the input tensor.
    auto _st = input.scalar_type();
    // Check if the input tensor datatype is valid.
    TORCH_CHECK(_st == torch::kFloat32 || _st == torch::kFloat16 || _st == torch::kBFloat16,
        "Invalid datatype. input must be FP16 or BF16 or FP32");

    // Initialize the quantized_input_shape and scale_shape vectors based on the input tensor shape and quantization mode.
    std::vector<int64_t> quantized_input_shape;
    for (int i = 0; i < input.dim(); i++)
        quantized_input_shape.push_back(input.size(i));
    std::vector<int64_t> scale_shape;
    if (quantize_mode == QuantizeMode::PER_TOKEN)
    {
        for (int i = 0; i < input.dim() - 1; i++)
            scale_shape.push_back(input.size(i));
        scale_shape.push_back(1);
    }
    else if (quantize_mode == QuantizeMode::PER_CHANNEL)
    {
        for (int i = 0; i < input.dim() - 2; i++)
            scale_shape.push_back(input.size(i));
        scale_shape.push_back(1);
        scale_shape.push_back(input.size(-1));
    }
    else // must be PER_TENSOR
    {
        scale_shape.assign(input.dim(), 1);
    }

    // Check if the input tensor is on the CUDA device.
    const auto is_cuda = input.is_cuda();
    // Move the input tensor to the CUDA device if it is not already there.
    input = input.cuda();

    // Create the quantized_input and scales tensors with the appropriate shapes and data types.
    Tensor quantized_input
        = torch::empty(quantized_input_shape, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));

    Tensor scales = torch::empty(scale_shape, torch::dtype(input.dtype()).device(torch::kCUDA).requires_grad(false));

    // Get the pointer to the quantized_input tensor data.
    auto quantized_input_ptr = reinterpret_cast<__nv_fp8_e4m3*>(get_ptr<int8_t>(quantized_input));

    // Get the default CUDA stream.
    auto stream = at::cuda::getDefaultCUDAStream();

    // Perform the quantization based on the input tensor datatype and quantization mode.
    if (input.scalar_type() == at::ScalarType::Float)
    {
        invokeComputeScalesAndQuantizeMatrix(quantized_input_ptr, get_ptr<float>(scales), get_ptr<const float>(input),
            input.numel(), input.size(-1), quantize_mode, stream);
    }
    else if (input.scalar_type() == at::ScalarType::Half)
    {
        invokeComputeScalesAndQuantizeMatrix(quantized_input_ptr, get_ptr<half>(scales), get_ptr<const half>(input),
            input.numel(), input.size(-1), quantize_mode, stream);
    }
#ifdef ENABLE_BF16
    else if (input.scalar_type() == at::ScalarType::BFloat16)
    {
        invokeComputeScalesAndQuantizeMatrix(quantized_input_ptr, get_ptr<__nv_bfloat16
