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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"

#include <cuda_runtime_api.h>

#include <memory>

namespace tensorrt_llm::runtime
{

class CudaEvent
{
public:
    // Creates a new cuda event. The event will be destroyed in the destructor.
    //
    // \param flags Flags for event creation. By default, event timing is disabled.
    explicit CudaEvent(unsigned int flags = cudaEventDisableTiming)
    {
        // Create a new CUDA event
        TLLM_CUDA_CHECK(::cudaEventCreate(&mEvent, flags));
        TLLM_LOG_TRACE("Created event %p", mEvent);
    }

    // Pass an existing cuda event to this object.
    //
    // \param event The event to pass to this object.
    // \param ownsEvent Whether this object owns the event and destroys it in the destructor.
    explicit CudaEvent(cudaEvent_t event, bool ownsEvent = true)
    {
        TLLM_CHECK_WITH_INFO(event != nullptr, "event is nullptr");
        mEvent = event;
        mOwnsEvent = ownsEvent;
    }

    // Returns the event associated with this object.
    [[nodiscard]] cudaEvent_t get() const
    {
        return mEvent;
    }

    // Synchronizes the event.
    void synchronize() const
    {
        TLLM_CUDA_CHECK(::cudaEventSynchronize(get()));
    }

    // Destructor, destroys the event if this object owns it.
    ~CudaEvent()
    {
        if (mOwnsEvent)
        {
            TLLM_CUDA_CHECK(::cudaEventDestroy(mEvent));
            TLLM_LOG_TRACE("Destroyed event %p", mEvent);
        }
    }

private:
    cudaEvent_t mEvent;
    bool mOwnsEvent;
};

} // namespace tensorrt_llm::runtime

