/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <iostream>
#include <stdexcept>
#include <string>

#include "tensorrt_llm_libutils.h"

int main(int argc, char* argv[])
{
    // Initialize TensorRT logger
    class TRTLogger : public nvinfer1::ILogger
    {
    public:
        void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
        {
            if (severity <= nvinfer1::ILogger::Severity::kERROR)
                std::cerr << "[TensorRT-LLM ERR]: " << msg << std::endl;
            else if (severity == nvinfer1::ILogger::Severity::kWARNING)
                std::cerr << "[TensorRT-LLM WARNING]: " << msg << std::endl;
            else
                std::cout << "[TensorRT-LLM LOG]: " << msg << std::endl;
        }
    };

    TRTLogger* trtLogger = new TRTLogger();

    // Load the TensorRT LLM plugin library
    std::string libname = "libtensorrt_llm_plugin.so";

    // Initialize the TensorRT plugin library
    bool initLibNvInferPlugins_status = initLibNvInferPlugins(trtLogger, "tensorrt_llm");
    std::cout << "initLibNvInferPlugins Success Status: " << initLibNvInferPlugins_status << std::endl;

    // Get the TensorRT library version
    std::cout << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;
    int32_t getInferLibVersion_version = getInferLibVersion();
    std::cout << "Version: " << getInferLibVersion_version << std::endl;

    return 0;
}




