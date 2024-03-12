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

// Include header for the WeightOnlyBatchedGemvKernelLauncher template class.
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/kernel.h"

// Namespace for the tensorrt_llm library and its kernels module.
namespace tensorrt_llm
{
namespace kernels
{

// Template specializations for the WeightOnlyBatchedGemvKernelLauncher class.
// These specializations cover various combinations of quantization types,
// per-channel or group-wise weight layouts, activation functions, and
// thread block configurations.

// Specialization for Int4b quantization, per-channel weight layout, Gelu activation,
// and a thread block configuration of (1, 1, 192).
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyPerChannel, GeluActivation,
    true, true, 1, 1, 192>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyPerChannel, GeluActivation,
    true, false, 1, 1, 192>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyPerChannel, GeluActivation,
    false, true, 1, 1, 192>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyPerChannel, GeluActivation,
    false, false, 1, 1, 192>;

// Specialization for Int4b quantization, per-channel weight layout, Relu activation,
// and a thread block configuration of (1, 1, 192).
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyPerChannel, ReluActivation,
    true, true, 1, 1, 192>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyPerChannel, ReluActivation,
    true, false, 1, 1, 192>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyPerChannel, ReluActivation,
    false, true, 1, 1, 192>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyPerChannel, ReluActivation,
    false, false, 1, 1, 192>;

// Specialization for Int4b quantization, per-channel weight layout, Identity activation,
// and a thread block configuration of (1, 1, 192).
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyPerChannel,
    IdentityActivation, true, true, 1, 1, 192>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyPerChannel,
    IdentityActivation, true, false, 1, 1, 192>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyPerChannel,
    IdentityActivation, false, true, 1, 1, 192>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyPerChannel,
    IdentityActivation, false, false, 1, 1, 192>;

// Specialization for Int4b quantization, group-wise weight layout with a group size of 64,
// Gelu activation, and a thread block configuration of (2, 1, 256).
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<64>, GeluActivation,
    true, true, 2, 1, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<64>, GeluActivation,
    true, false, 2, 1, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<64>, GeluActivation,
    false, true, 2, 1, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<64>, GeluActivation,
    false, false, 2, 1, 256>;

// Specialization for Int4b quantization, group-wise weight layout with a group size of 64,
// Relu activation, and a thread block configuration of (2, 1, 256).
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<64>, ReluActivation,
    true, true, 2, 1, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<64>, ReluActivation,
    true, false, 2, 1, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int4b, WeightOnlyGroupWise<64>, ReluActivation,
    false, true, 2, 1, 256
