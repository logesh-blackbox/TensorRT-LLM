# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.1)

project(trt_llm_custom_plugins)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_VERBOSE_MAKEFILE 1)

find_package(CUDA REQUIRED)

add_library(
  trt_llm_custom_plugins
  SHARED
  tritonPlugins.cpp
  TritonFlashAttentionPlugin.cpp
  aot/fmha_kernel_fp16.c
  aot/fmha_kernel_fp32.c
  aot/fp16/fmha_kernel_d64_fp16.fbf0f274_0d1d2d3d4d5d6789.c
  aot/fp32/fmha_kernel_d64_fp32.f30323ef_0d1d2d3d4d5d6789.c)

target_link_libraries(
  trt_llm_custom_plugins
  PUBLIC
    cuda::cudart
    cuda::cuda_runtime_api
    tensorrt
    tensorrt_llm_common
    tensorrt_llm)

target_include_directories(
  trt_llm_custom_plugins
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/aot>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/aot/fp16>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/aot/fp32>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/aot>
    $<INSTALL_INTERFACE:include/aot/fp16>
    $<INSTALL_INTERFACE:include/aot/fp32>
    ${CUDA_INCLUDE_DIRS}
    ${TRT_INCLUDE_DIR}
    ${TRT_LLM_INCLUDE_DIR})

set(CMAKE_BUILD_TYPE release)
