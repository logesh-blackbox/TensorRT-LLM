/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"

#include "cutlass_extensions/arch/mma.h"
#include "cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.h"

namespace cutlass
{
namespace gemm
{
namespace kernel
{

// Base template for MixedGemmArchTraits
template <typename TypeA, typename TypeB, typename arch, typename Enable = void>
struct MixedGemmArchTraits
{
};

// Specialization for float, float, arch
template <typename arch>
struct MixedGemmArchTraits<float, float, arch>
{
    // Number of stages for the kernel
    static constexpr int Stages = 2;

    // Operator class for the kernel
    using OperatorClass = cutlass::arch::OpClassSimt;

    // Accumulator type for the kernel
    using AccType = float;

    // Layout for matrix B
    using LayoutB = cutlass::layout::RowMajor;

    // Elements per access for matrix A
    static constexpr int ElementsPerAccessA = 1;

    // Elements per access for matrix B
    static constexpr int ElementsPerAccessB = 1;

    // Elements per access for matrix C
    static constexpr int ElementsPerAccessC = 1;

    // ThreadblockK for the kernel
    static constexpr int ThreadblockK = 8;

    // Instruction shape for the kernel
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

    // Operator for the kernel
    using Operator = cutlass::arch::OpMultiplyAdd;
};

// Specialization for Volta architecture
template <typename TypeA, typename TypeB>
struct MixedGemmArchTraits<TypeA, TypeB, cutlass::arch::Sm70,
    typename cutlass::platform::enable_if<cutlass::platform::is_same<TypeA, cutlass::half_t>::value
        || cutlass::platform::is_same<TypeA, cutlass::bfloat16_t>::value>::type>
{
private:
    // Layout details for matrix B
    using LayoutDetails = LayoutDetailsB<TypeB, cutlass::arch::Sm70>;

public:
    // ThreadblockK for the kernel
    static constexpr int ThreadblockK = LayoutDetails::ThreadblockK;

    // Operator class for the kernel
    using OperatorClass = cutlass::arch::OpClassTensorOp;

    // Accumulator type for the kernel
    using AccType = float;

    // Layout for matrix B
    using LayoutB = typename LayoutDetails::Layout;

    // Elements per access for matrix A
    static constexpr int ElementsPerAccessA = 128 / cutlass::sizeof_bits<TypeA>::value;

    // Elements per access for matrix B
    static constexpr int ElementsPerAccessB = LayoutDetails::ElementsPerAccess;

    // Elements per access for matrix C
    static constexpr int ElementsPerAccessC = 128 / cutlass::sizeof_bits<TypeA>::value;

    // Instruction shape for the kernel
    using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;

    // Operator for the kernel
    using Operator = typename LayoutDetails::Operator;
};

// Specialization for Turing architecture
template <typename TypeA, typename TypeB>
struct MixedGemmArchTraits<TypeA, TypeB, cutlass::arch::Sm75,
    typename cutlass::platform::enable_if<cutlass::platform::is_same<TypeA, cutlass::half_t>::value
        || cutlass::platform::is_same<TypeA, cutlass::bfloat16_t>::value>::type>
{
private:
    // Layout details for matrix B
    using LayoutDetails = LayoutDetailsB<TypeB, cutlass::arch::Sm75>;

public:
    // ThreadblockK for the kernel
    static constexpr int ThreadblockK = LayoutDetails::ThreadblockK;

    // Operator class for the kernel
    using OperatorClass = cutlass::arch::OpClassTensorOp;

    // Accumulator type for the kernel
    using AccType = float;

    // Layout for matrix B
    using LayoutB = typename LayoutDetails::Layout;

    // Elements per access for matrix A
    static constexpr int ElementsPerAccessA = 128 / cutlass::sizeof_bits<TypeA>::value;

    // Elements per access for matrix B
    static constexpr int ElementsPerAccessB = LayoutDetails::ElementsPerAccess;

    // Elements per access for matrix C
    static constexpr int ElementsPerAccessC = 128 / cutlass::sizeof_bits<TypeA>::value;

    // Instruction shape for the kernel
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

    // Operator for the kernel
