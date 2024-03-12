/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iTensor.h"

namespace tensorrt_llm::runtime::kernels
{

/**
 * @template T
 * @brief Fills a buffer with a specified value.
 * @param buffer The buffer to fill.
 * @param value The value to fill the buffer with.
 * @param stream The CUDA stream to use for the operation.
 */
template <typename T>
void invokeFill(IBuffer& buffer, T value, CudaStream const& stream);

/**
 * @template T
 * @brief Fills a batch of buffers with specified values based on indices and strides.
 * @param buffer The base buffer to fill.
 * @param indices The indices of the values to fill in the base buffer.
 * @param stride The stride between each index in the base buffer.
 * @param values The buffer containing the values to fill in the base buffer.
 * @param stream The CUDA stream to use for the operation.
 */
template <typename T>
void invokeFillBatch(
    IBuffer& buffer, IBuffer const& indices, std::size_t stride, IBuffer const& values, CudaStream const& stream);

/**
 * @template T
 * @brief Copies a batch of buffers with specified offsets.
 * @param srcBuffer The source buffer to copy from.
 * @param destBuffer The destination buffer to copy to.
 * @param srcIndices The indices of the source buffer to copy from.
 * @param destIndices The indices of the destination buffer to copy to.
 * @param stride The stride between each index in the source and destination buffers.
 * @param stream The CUDA stream to use for the operation.
 */
template <typename T>
void invokeCopyBatch(IBuffer const&, IBuffer&, IBuffer const&, IBuffer const&, std::size_t, CudaStream const&);

/**
 * @template T
 * @brief Adds a value to a buffer.
 * @param buffer The buffer to add the value to.
 * @param value The value to add to the buffer.
 * @param stream The CUDA stream to use for the operation.
 */
template <typename T>
void invokeAdd(IBuffer& buffer, T value, CudaStream const& stream);

/**
 * @brief Reduces a buffer using a reduction operation.
 * @param output The output buffer for the reduction operation.
 * @param input The input buffer for the reduction operation.
 * @param stream The CUDA stream to use for the operation.
 */
void reduce(IBuffer& output, IBuffer const& input, CudaStream const& stream);

/**
 * @brief Transposes an ITensor object.
 * @param output The output ITensor object.
 * @param input The input ITensor object.
 * @param stream The CUDA stream to use for the operation.
 */
void invokeTranspose(ITensor& output, ITensor const& input, CudaStream const& stream);

/**
 * @brief Transposes an ITensor object with an output offset.
 * @param output The output ITensor object.
 * @param input The input ITensor object.
 * @param outputOffset The output offset for the transposed ITensor object.
 * @param stream The CUDA stream to use for the operation.
 */
void invokeTransposeWithOutputOffset(
    ITensor& output, ITensor const& input, SizeType outputOffset, CudaStream const& stream);

/**
 * @brief Transposes an ITensor object with an input offset.
 * @param output The output ITensor object.
 * @param input The input ITensor object.
 * @param inputOffset The input offset for the ITensor object.
 * @param stream The CUDA stream to use for the operation.
 */
void invokeTransposeWithInputOffset(
    ITensor& output, ITensor const& input, SizeType inputOffset, CudaStream const& stream);

/**
 * @brief Performs an inclusive sum on a buffer.
 * @param output The output buffer for the inclusive sum operation.
 * @param input The input buffer for the inclusive sum operation.
 * @param manager The buffer manager for the operation.
 * @param stream The CUDA stream to use for the operation.
 */
void invokeInclusiveSum(IBuffer& output, IBuffer const& input, BufferManager const& manager, CudaStream const& stream);

/**
 * @brief Builds a token mask ITensor object.
 * @param tokenMask The output token mask ITensor object.
 * @param inputLengths The input lengths ITensor object.
 * @param maxInputLength The maximum input length.
 * @param stream The CUDA stream to use for the operation.
 */
void invokeBuildTokenMask(
    ITensor& tokenMask, ITensor const& inputLengths, SizeType maxInputLength, CudaStream const& stream);

