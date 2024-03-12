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

#include "tensorrt_llm/common/memoryUtils.h"
#include <array>
#include <gtest/gtest.h>

namespace tc = tensorrt_llm::common;

TEST(memory_utils, flat_index)
{
    // All testing happens at compile time.
    std::array<int, 4> constexpr dims{4, 3, 2, 1};
    static_assert(tc::flat_index(dims, 0) == 0);
    static_assert(tc::flat_index(dims, 1) == 1);
    static_assert(tc::flat_index(dims, 2) == 2);
    static_assert(tc::flat_index(dims, 3) == 3);
    static_assert(tc::flat_index(dims, 0, 0) == 0);
    static_assert(tc::flat_index(dims, 0, 1) == 1);
    static_assert(tc::flat_index(dims, 1, 0) == 3);
    static_assert(tc::flat_index(dims, 1, 1) == 4);
    static_assert(tc::flat_index(dims, 1, 2) == 5);
    static_assert(tc::flat_index(dims, 2, 0) == 6);
    static_assert(tc::flat_index(dims, 2, 1) == 7);
    static_assert(tc::flat_index(dims, 2, 2) == 8);
    static_assert(tc::flat_index(dims, 0, 0, 0) == 0);
    static_assert(tc::flat_index(dims, 0, 0, 1) == 1);
    static_assert(tc::flat_index(dims, 0, 1, 0) == 2);
    static_assert(tc::flat_index(dims, 0, 1, 1) == 3);
    static_assert(tc::flat_index(dims, 0, 2, 0) == 4);
    static_assert(tc::flat_index(dims, 0, 2, 1) == 5);
    static_assert(tc::flat_index(dims, 1, 0, 0) == 6);
    static_assert(tc::flat_index(dims, 1, 0, 1) == 7);
    static_assert(tc::flat_index(dims, 1, 1, 0) == 8);
    static_assert(tc::flat_index(dims, 1, 1, 1) == 9);
    static_assert(tc::flat_index(dims, 1, 2, 0) == 10);
    static_assert(tc::flat_index(dims, 1, 2, 1) == 11);
    static_assert(tc::flat_index(0, dims, 1, 2, 1) == 11);
    static_assert(tc::flat_index(1, dims, 1, 2, 1) == 35);
    static_assert(tc::flat_index(2, dims, 1, 2, 1) == 59);
    static_assert(tc::flat_index(dims, 0, 0, 0) == tc::flat_index(0, &dims[1], 0, 0));
    static_assert(tc::flat_index(dims, 0, 0, 1) == tc::flat_index(0, &dims[1], 0, 1));
    static_assert(tc::flat_index(dims, 0, 1, 0) == tc::flat_index(0, &dims[1], 1, 0));
    static_assert(tc::flat_index(dims, 0, 1, 1) == tc::flat_index(0, &dims[1], 1, 1));
    static_assert(tc::flat_index(dims, 0, 2, 0) == tc::flat_index(0, &dims[1], 2, 0));
    static_assert(tc::flat_index(dims, 0, 2, 1) == tc::flat_index(0, &dims[1], 2, 1));
    static_assert(tc::flat_index(dims, 1, 0, 0) == tc::flat_index(1, &dims[1], 0, 0));
    static_assert(tc::flat_index(dims, 1, 0, 1) == tc::flat_index(1, &dims[1], 0, 1));
    static_assert(tc::flat_index(dims, 1, 1, 0) == tc::flat_index(1, &dims[1], 1, 0));
    static_assert(tc::flat_index(dims, 1, 1, 1) == tc::flat_index(1, &dims[1], 1, 1));
    static_assert(
