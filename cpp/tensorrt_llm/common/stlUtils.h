/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

// Includes
#include "tensorrt_llm/common/assert.h"
#include <functional>
#include <numeric>

namespace tensorrt_llm::common::stl_utils
{

/**
 * @brief Performs an inclusive scan on a range of elements, computing the cumulative prefix sum.
 *
 * This function is similar to std::inclusive_scan, but provides a fallback implementation for older compilers.
 *
 * @tparam TInputIt The type of the input iterator.
 * @tparam TOutputIt The type of the output iterator.
 * @tparam TBinOp The type of the binary operation used to combine elements.
 *
 * @param first The beginning of the input range.
 * @param last The end of the input range.
 * @param dFirst The beginning of the output range.
 * @param op The binary operation used to combine elements.
 *
 * @return The end of the output range.
 */
template <typename TInputIt, typename TOutputIt, typename TBinOp>
constexpr TOutputIt basicInclusiveScan(TInputIt first, TInputIt last, TOutputIt dFirst, TBinOp op)
{
    if (first != last)
    {
        auto val = *first;
        while (true)
        {
            *dFirst = val;
            ++dFirst;
            ++first;
            if (first == last)
            {
                break;
            }
            val = op(std::move(val), *first);
        }
    }
    return dFirst;
}

/**
 * @brief Performs an inclusive scan on a range of elements, computing the cumulative prefix sum.
 *
 * This function is similar to std::inclusive_scan, but provides a fallback implementation for older compilers.
 *
 * @tparam TInputIt The type of the input iterator.
 * @tparam TOutputIt The type of the output iterator.
 *
 * @param first The beginning of the input range.
 * @param last The end of the input range.
 * @param dFirst The beginning of the output range.
 *
 * @return The end of the output range.
 */
template <typename TInputIt, typename TOutputIt>
constexpr TOutputIt inclusiveScan(TInputIt first, TInputIt last, TOutputIt dFirst)
{
#if defined(__GNUC__) && __GNUC__ <= 8
    return basicInclusiveScan(first, last, dFirst, std::plus<>{});
#else
    return std::inclusive_scan(first, last, dFirst);
#endif
}

/**
 * @brief Performs an exclusive scan on a range of elements, computing the cumulative suffix sum.
 *
 * This function is similar to std::exclusive_scan, but provides a fallback implementation for older compilers.
 *
 * @tparam TInputIt The type of the input iterator.
 * @tparam TOutputIt The type of the output iterator.
 * @tparam T The type of the initial value.
 * @tparam TBinOp The type of the binary operation used to combine elements.
 *
 * @param first The beginning of the input range.
 * @param last The end of the input range.
 * @param dFirst The beginning of the output range.
 * @param init The initial value.
 * @param op The binary operation used to combine elements.
 *
 * @return The end of the output range.
 */
template <typename TInputIt, typename TOutputIt, typename T, typename TBin
