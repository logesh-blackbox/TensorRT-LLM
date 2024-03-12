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

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/tllmException.h"

#include <string>

using ::testing::HasSubstr;

TEST(TllmException, StackTrace)
{
    // This test checks the behavior of the TllmException constructor and the 'what()' method.
    // It creates a TllmException object with a formatted error message and checks if the 'what()' method
    // returns the expected error message, including the line number, file name, and the provided error message.

    // clang-format off
    auto ex = NEW_TLLM_EXCEPTION("TestException %d", 1);
    std::string const what{ex.what()};
    // clang-format on

    // Check if the 'what()' method returns the line number where the exception was thrown.
    EXPECT_THAT(what, HasSubstr(std::to_string(__LINE__ - 2)));

    // Check if the 'what()' method returns the provided error message.
    EXPECT_THAT(what, HasSubstr("TestException 1"));

    // Check if the 'what()' method returns the file name where the exception was thrown.
    EXPECT_THAT(what, HasSubstr(__FILE__));

#if !defined(_MSC_VER)
    // Check if the 'what()' method does not include the constructor's function name.
    EXPECT_THAT(what, ::testing::Not(HasSubstr("tensorrt_llm::common::TllmException::TllmException")));

    // Check if the 'what()' method returns the name of the test file.
    EXPECT_THAT(what, HasSubstr("tests/tllmExceptionTest"));

    // Check if the 'what()' method returns the name of the main function.
    EXPECT_THAT(what, HasSubstr("main"));
#endif
}

TEST(TllmException, Logger)
{
    // This test checks the behavior of the TLLM_THROW macro and the TLLM_LOG_EXCEPTION function.
    // It creates a TllmException object using the TLLM_THROW macro and logs the exception using the TLLM_LOG_EXCEPTION function.
    // It then checks if the logged output includes the expected error message, line number, file name, and function name.

    try
    {
        // clang-format off
        TLLM_THROW("TestException %d", 1);
        // clang-format on
    }
    catch (const std::exception& e)
    {
        // Capture the standard output.
        testing::internal::CaptureStdout();

        // Log the exception.
        TLLM_LOG_EXCEPTION(e);

        //
