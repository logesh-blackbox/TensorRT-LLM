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

#include "tensorrt_llm/thop/thUtils.h"

namespace torch_ext
{

/**
 * Convert the given Tensor's shape to a vector of size_t.
 *
 * @param tensor The Tensor to convert the shape of.
 * @return A vector of size_t representing the shape of the given Tensor.
 */
std::vector<size_t> convert_shape(torch::Tensor tensor)
{
    std::vector<size_t> v_shape;
    for (int i = 0; i < tensor.dim(); i++)
    {
        v_shape.push_back(tensor.size(i));
    }
    return v_shape;
}

/**
 * Convert the given Tensor to a TensorRT common::Tensor with the specified memory type.
 *
 * @tparam T The data type of the Tensor to convert.
 * @param tensor The Tensor to convert.
 * @param memory_type The memory type of the resulting TensorRT common::Tensor.
 * @return A TensorRT common::Tensor with the specified memory type and data type,
 *         containing the same data as the given Tensor.
 */
template <typename T>
tensorrt_llm::common::Tensor convert_tensor(torch::Tensor tensor, tensorrt_llm::common::MemoryType memory_type)
{
    return tensorrt_llm::common::Tensor{
        memory_type, tensorrt_llm::common::getTensorType<T>(), convert_shape(tensor), get_ptr<T>(tensor)};
}

// Template instantiations
template tensorrt_llm::common::Tensor convert_tensor<int8_t>(
    torch::Tensor tensor, tensorrt_llm::common::MemoryType memory_type);
template tensorrt_llm::common::Tensor convert_tensor<float>(
    torch::Tensor tensor, tensorrt_llm::common::MemoryType memory_type);
template tensorrt_llm::common::Tensor convert_tensor<half>(
    torch::Tensor tensor, tensorrt_llm::common::MemoryType memory_type);
#ifdef ENABLE_BF16
template tensorrt_llm::common::Tensor convert_tensor<__nv_bfloat16>(
    torch::Tensor tensor, tensorrt_llm::common::MemoryType memory_type);
#endif
template tensorrt_llm::common::Tensor convert_tensor<int>(
    torch::Tensor tensor, tensorrt_llm::common::MemoryType memory_type);
template tensorrt_llm::common::Tensor convert_tensor<unsigned long long int>(
    torch::Tensor tensor, tensorrt_llm::common::MemoryType memory_type);
template tensorrt_llm::common::Tensor convert_tensor<unsigned int>(
    torch::Tensor tensor, tensorrt_llm::common::MemoryType memory_type);
template tensorrt_llm::common::Tensor convert_tensor<bool>(
    torch::Tensor tensor, tensorrt_llm::common::MemoryType memory_type);

/**
 * Convert the given Tensor to a TensorRT common::Tensor with the default memory type.
 *
 * @tparam T The data type of the Tensor to convert.
 * @param tensor The Tensor to convert.
 * @return A TensorRT common::Tensor with the default memory type and data type,
 *         containing the same data as the given Tensor.
 */
template <typename T>
tensorrt_llm::common::Tensor convert_tensor(torch::Tensor tensor)
{
    tensorrt_llm::common::MemoryType mtype
        = tensor.is_cuda() ? tensorrt_llm::common::MEMORY_GPU : tensorrt_llm::common::MEMORY_CPU;
    return convert_tensor<T>(tensor, mtype);
}

// Template instantiations
template tensorrt_llm::common::Tensor convert_tensor<int8_t>(torch::Tensor tensor);
template tensorrt_llm::common::Tensor convert_tensor<float>(torch::Tensor tensor);
template tensorrt_llm::common::Tensor convert_tensor<half>(torch::Tensor tensor);
#ifdef ENABLE_BF16
template tensorrt_llm::common::Tensor convert_tensor<__nv_bfloat16>(torch::Tensor tensor);
#endif
template tensorrt_llm::common::Tensor convert_tensor<int>(torch::Tensor tensor);
template tensorrt_llm::common::Tensor convert_tensor<unsigned long long int>(torch::Tensor tensor);
template tensorrt_llm::common::Tensor convert_tensor<unsigned int>(torch::Tensor tensor);
template tensorrt_llm::common::Tensor convert_tensor<bool>(torch::Tensor tensor);

/**
 * Get the size of the given Tensor in bytes.
 *
 * @param tensor The Tensor to get the size of.
 * @return The size of the given Tensor in bytes.
 */
size_t sizeBytes(torch::Tensor tensor)
{
    return tensor.numel() * torch::elementSize(torch::typeMetaToScalarType(tensor.dtype()));
}

} // namespace torch_ext

