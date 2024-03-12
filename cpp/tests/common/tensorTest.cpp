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

#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "tensorrt_llm/common/tensor.h"

using namespace tensorrt_llm::common;

// Define a macro for comparing two tensors
#define EXPECT_EQUAL_TENSORS(t1, t2)                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        EXPECT_TRUE(t1.where == t2.where);                                                                             \
        EXPECT_TRUE(t1.type == t2.type);                                                                               \
        EXPECT_TRUE(t1.shape == t2.shape);                                                                             \
        EXPECT_TRUE(t1.data == t2.data);                                                                               \
    } while (false)

// Test for checking if a key exists in the TensorMap
TEST(TensorMapTest, HasKeyCorrectness)
{
    bool* v1 = new bool(true);
    float* v2 = new float[6]{1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f};
    Tensor t1 = Tensor{MEMORY_CPU, TYPE_BOOL, {1}, v1};
    Tensor t2 = Tensor{MEMORY_CPU, TYPE_FP32, {3, 2}, v2};

    TensorMap map({{"t1", t1}, {"t2", t2}});

    // Check if the keys "t1" and "t2" exist in the map
    EXPECT_TRUE(map.contains("t1"));
    EXPECT_TRUE(map.contains("t2"));

    // Check if the key "t3" does not exist in the map
    EXPECT_FALSE(map.contains("t3"));

    delete v1;
    delete[] v2;
}

// Test for inserting a tensor into the TensorMap
TEST(TensorMapTest, InsertCorrectness)
{
    int* v1 = new int[4]{1, 10, 20, 30};
    float* v2 = new float[2]{1.0f, 2.0f};
    Tensor t1 = Tensor(MEMORY_CPU, TYPE_INT32, {4}, v1);
    Tensor t2 = Tensor(MEMORY_CPU, TYPE_INT32, {2}, v2);

    TensorMap map({{"t1", t1}});

    // Check the size and contents of the map after insertion
    EXPECT_TRUE(map.size() == 1);
    EXPECT_TRUE(map.contains("t1"));
    EXPECT_EQUAL_TENSORS(map.at("t1"), t1);
    EXPECT_FALSE(map.contains("t2"));

    delete[] v1;
    delete[] v2;
}

// Test for inserting a tensor with a nonexistent key into the TensorMap
TEST(TensorMapTest, InsertDoesNotAllowNoneTensor)
{
    TensorMap map;

    // Attempt to insert a nonexistent tensor and check for an exception
    EXPECT_THROW(map.insert("none", {}), std::runtime_error);

    // Attempt to insert a tensor with a null data pointer and check for an exception
    Tensor none_data_tensor = Tensor(MEMORY_CPU, TYPE_INT32, {}, nullptr);
    EXPECT_THROW(map.insert("empty", none_data_tensor), std::runtime_error);
}

// Test for inserting a tensor with a duplicate key into the TensorMap
TEST(TensorMapTest, InsertDoesNotAllowDuplicatedKey)
{
    int* v1 = new int[4]{1, 10, 20, 30};
    Tensor t1 = Tensor(MEMORY_CPU, TYPE_INT32, {4}, v1);
    Tensor t2 = Tensor(MEMORY_CPU, TYPE_INT32, {2}, v1);
    TensorMap map({{"t1", t1}});

    // Attempt to insert a tensor with a duplicate key and check for an exception
    EXPECT_THROW(map.insert("t1", t2), std::runtime_error);

    delete[] v1;
}

// Test for getting a value from the TensorMap
TEST(TensorMapTest, GetValCorrectness)
{
    int* v1 = new int[4]{1, 10, 20, 30};
    Tensor t1 = Tensor(MEMORY_CPU, TYPE_INT32, {4}, v1);

    TensorMap map({{"t1", t1}});

    // Check for an exception when trying to get a nonexistent key
    EXPECT_THROW(map.getVal<int>("t3"), std::runtime_error);

    // Check the value of the tensor with key "t1"
    EXPECT_TRUE(map.getVal<int>("t1") == 1);

    // Check the value of the tensor with key "t1" using a default value
    EXPECT_TRUE(map.getVal<int>("t1", 3) == 1);

    // Check the default value when the key does not exist
    EXPECT_TRUE(map.getVal<int>("t2", 3) == 3);

    // Update the value of the tensor and check the new value
    v1[0] += 1;
    EXPECT_TRUE(map.getVal<int>("t1
