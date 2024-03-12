/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_runtime.h>

#define TLLM_CUBLAS_VERSION_CALC(MAJOR, MINOR, PATCH) (MAJOR * 10000 + MINOR * 100 + PATCH)
#define TLLM_CUBLAS_VER_LE(MAJOR, MINOR, PATCH)                                                                        \
    TLLM_CUBLAS_VERSION_CALC(CUBLAS_VER_MAJOR, CUBLAS_VER_MINOR, CUBLAS_VER_PATCH)                                     \
    <= TLLM_CUBLAS_VERSION_CALC(MAJOR, MINOR, PATCH)
#define TLLM_CUBLAS_VER_LT(MAJOR, MINOR, PATCH)                                                                        \
    TLLM_CUBLAS_VERSION_CALC(CUBLAS_VER_MAJOR, CUBLAS_VER_MINOR, CUBLAS_VER_PATCH)                                     \
        < TLLM_CUBLAS_VERSION_CALC(MAJOR, MINOR, PATCH)
#define TLLM_CUBLAS_VER_GE(MAJOR, MINOR, PATCH)                                                                        \
    TLLM_CUBLAS_VERSION_CALC(CUBLAS_VER_MAJOR, CUBLAS_VER_MINOR, CUBLAS_VER_PATCH)                                     \
    >= TLLM_CUBLAS_VERSION_CALC(MAJOR, MINOR, PATCH)
#define TLLM_CUBLAS_VER_GT(MAJOR, MINOR, PATCH)                                                                        \
    TLLM_CUBLAS_VERSION_CALC(CUBLAS_VER_MAJOR, CUBLAS_VER_MINOR, CUBLAS_VER_PATCH)                                     \
        > TLLM_CUBLAS_VERSION_CALC(MAJOR, MINOR, PATCH)

#if CUDA_VERSION >= 11030
#define TLLM_CUBLAS_API_VERSION 11
#elif CUDA_VERSION >= 11000
#define TLLM_CUBLAS_API_VERSION 11
#elif CUDA_VERSION >= 10200
#define TLLM_CUBLAS_API_VERSION 10
#elif CUDA_VERSION >= 10100
#define TLLM_CUBLAS_API_VERSION 10
#elif CUDA_VERSION >= 9200
#define TLLM_CUBLAS_API_VERSION 9
#elif CUDA_VERSION >=
