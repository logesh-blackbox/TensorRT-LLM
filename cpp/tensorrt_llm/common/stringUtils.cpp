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

#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/common/assert.h"

#include <cerrno>
#include <cstdarg>
#include <cstring>
#include <string>

namespace tensorrt_llm::common
{

/**
 * @brief Formats a string using a given format and variable arguments.
 *
 * This function takes a format string and a variable number of arguments,
 * and returns a formatted string. It uses the standard C++ vsnprintf function
 * to format the string.
 *
 * @param format The format string to use for formatting.
 * @param args The variable arguments to use for formatting.
 *
 * @return A formatted string.
 */
std::string vformat(char const* fmt, va_list args)
{
    va_list args0;
    va_copy(args0, args);
    auto const size = vsnprintf(nullptr, 0, fmt, args0);
    if (size <= 0)
        return "";

    std::string stringBuf(size, char{});
    auto const size2 = std::vsnprintf(&stringBuf[0], size + 1, fmt, args);

    TLLM_CHECK_WITH_INFO(size2 == size, std::strerror(errno));

    return stringBuf;
}

/**
 * @brief Formats a string using a given format and variable arguments.
 *
 * This function takes a format string and a variable number of arguments,
 * and returns a formatted string. It uses the vformat function to format the string.
 *
 * @param format The format string to use for formatting.
 * @param ... The variable arguments to use for formatting.
 *
 * @
