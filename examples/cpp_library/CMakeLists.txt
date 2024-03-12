# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# This CMakeLists.txt file is for building the TRT LLM C++ load example.
# It uses CMake to build the example.

cmake_minimum_required(VERSION 3.1)

# Set the minimum C++ standard to C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Define the project name
set(TARGET_NAME trt_llm_plugins_cpp_load_example)
project(${TARGET_NAME})

# Enable verbose makefiles
set(CMAKE_VERBOSE_MAKEFILE 1)

# Set compiler flags
set(CMAKE_C_FLAGS "-Wall -pthread ")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -lstdc++")
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})

# Set the build type to release
set(CMAKE_BUILD_TYPE release)

# Find CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA library status:")
message(STATUS "    config: ${CUDA_DIR}")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

# Declare the executable target built from your sources
add_executable(${TARGET_NAME} main.cpp)

# Link your application with CUDA libraries
target_link_libraries(${TARGET_NAME} PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(${TARGET_NAME} PRIVATE cudnn)
target_link_libraries(${TARGET_NAME} PRIVATE nvinfer)
target_link_libraries(${TARGET_NAME} PRIVATE nvinfer_plugin_tensorrt_llm)

# Add include directories
target_include_directories(${TARGET_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})

