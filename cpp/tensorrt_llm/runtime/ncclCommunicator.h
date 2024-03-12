/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/worldConfig.h"

struct ncclComm;
typedef struct ncclComm* ncclComm_t;

namespace tensorrt_llm::runtime
{

class NcclCommunicator
{
public:
    /**
     * @template T
     * @param sendbuff Buffer with data to send
     * @param count Number of elements to send
     * @param peer ID of the receiving process
     * @param stream CUDA stream to use for the operation
     * @param logger Logger for reporting errors
     *
     * This function sends data over the network to the process with the given ID.
     */
    template <typename T>
    void send(T* sendbuff, size_t count, int peer, CudaStream const& stream, nvinfer1::ILogger& logger) const;

    /**
     * @template T
     * @param buf Buffer to send
     * @param peer ID of the receiving process
     * @param stream CUDA stream to use for the operation
     * @param logger Logger for reporting errors
     *
     * This function sends the contents of the given buffer over the network to the process with the given ID.
     */
    template <typename T>
    void send(IBuffer const& buf, int peer, CudaStream const& stream, nvinfer1::ILogger& logger) const
    {
        send(bufferCast<T>(buf), buf.getSize(), peer, stream, logger);
    }

    /**
     * @template T
     * @param recvbuff Buffer to receive data into
     * @param count Number of elements to receive
     * @param peer ID of the sending process
     * @param stream CUDA stream to use for the operation
     * @param logger Logger for reporting errors
     *
     * This function receives data over the network from the process with the given ID.
     */
    template <typename T>
    void receive(T* recvbuff, size_t count, int peer, CudaStream const& stream, nvinfer1::ILogger& logger) const;

    /**
     * @template T
     * @param
