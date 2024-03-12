/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#if ENABLE_BF16
#include <cuda_bf16.h>
#endif // ENABLE_BF16
#include <cuda_fp16.h>

#include <memory>  // std::make_unique
#include <sstream> // std::stringstream
#include <string>
#include <vector>

namespace tensorrt_llm::common
{
#if ENABLE_BF16
    // This function converts a bfloat16 value to a float value
    inline float __bfloat162float(const __nv_bfloat16& val)
    {
        return __nv_bfloat16_to_float(val);
    }
#endif // ENABLE_BF16

    // This function converts a half-precision float value to a single-precision float value
    inline float __half2float(const __half& val)
    {
        return __half_to_float(val);
    }

#if defined(_MSC_VER)
    // Formats a string with the given format and variable arguments
    std::string fmtstr(char const* format, ...);
#else
    // Formats a string with the given format and variable arguments
    std::string fmtstr(char const* format, ...) __attribute__((format(printf, 1, 2)));
#endif

// __PRETTY_FUNCTION__ is used for neat debugging printing but is not supported on Windows
// The alternative is __FUNCSIG__, which is similar but not identical
#if defined(_WIN32)
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

    // This template function writes an array of values to an output stream
    template <typename U, typename TStream, typename T>
    inline TStream& arr2outCasted(TStream& out, T* arr, size_t size)
    {
        out << "(";
        if (size > 0)
        {
            for (size_t i = 0; i < size - 1; ++i)
            {
                out << static_cast<U>(arr[i]) << ", ";
            }
            out << static_cast<U>(arr[size - 1]);
        }
        out << ")";
        return out;
    }

    // This template function writes an array of values to an output stream
    template <typename TStream, typename T>
    inline TStream& arr2out(TStream& out, T* arr, size_t size)
    {
        return arr2outCasted<T>(out, arr, size);
    }

    // This function converts an array of values to a string
    template <typename T>
    inline std::string arr2str(T* arr, size_t size)
    {
        std::stringstream ss;
        return arr2out(ss, arr, size).str();
    }

    // This function converts a vector of values to a string
    template <typename T>
    inline std::string vec2str(std::vector<T> vec)
    {
        return arr2str(vec.data(), vec.size());
    }

    // This function checks if a string starts with a given prefix
    inline bool strStartsWith(std::string const& str, std::string const& prefix)
    {
        return str.rfind(prefix, 0) == 0;
    }

} // namespace tensorrt_llm::common
