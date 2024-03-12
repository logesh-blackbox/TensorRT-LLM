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

#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass_extensions/arch/mma.h"

#include "cutlass_extensions/gemm/threadblock/dq_mma_pipelined.h"
#include "cutlass_extensions/gemm/warp/default_mma_tensor_op.h"
#include "cutlass_extensions/gemm/warp/mma_tensorop_compute_B_with_f16.h"
#include "cutlass_extensions/tile_interleaved_layout.h"

#include "cutlass_extensions/gemm/threadblock/default_dq_mma.h"

namespace cutlass
{
namespace gemm
{
namespace threadblock
{

////////////////////////////////////////////////////////////////////////////////

// This is a template specialization for the DqMma struct.
// It is used to define the threadblock-scoped pipelined matrix multiply
// for specific data types, layouts, and architectures.
template <
    /// Type for element A
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Type for element B
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for the input scale
    typename ElementScale,
    /// Layout for the scale operand
    typename LayoutScale,
    /// Access granularity of Scales in unit of elements
    int kAlignmentScale,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator_,
    /// Option for shared memory clear
    SharedMemoryClearOption::Value shared_memory_clear_option,
    /// Additional template parameters for the specialization
    typename EnableIf = void>
struct DqMma;

// This is a specialization of the DqMma struct for the case when the architecture
// is less than compute capability 8.0 and the B matrix is not column-major interleaved.
template <
    /// Type for element A
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Type for element B
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for the input scale
    typename ElementScale,
    /// Layout for the scale operand
    typename LayoutScale,
    /// Access granularity of Scales in unit of elements
    int kAlignmentScale,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator_,
    typename EnableIf = void>
struct DqMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementScale, LayoutScale, kAlignmentScale,
    ElementAccumulator, layout::RowMajor, OperatorClass, ArchTag, ThreadblockShape, WarpShape, InstructionShape, 2,
    Operator_, SharedMemoryClearOption::kNone,
    typename platform::enable_if<(
        ArchTag::kMinComputeCapability < 80 && !layout::IsColumnMajorTileInterleave<LayoutB>::value)>::type>
{

    static_assert(platform::is_same<ElementA, half_t>::value || platform::is_same<ElementA, bfloat16_t>::value,
        "Element A must be fp16 or bf16");

    static_assert(platform::is_same<ElementB, uint8_t>::value || platform::is_same<ElementB, uint4b_t>::value,
        "Element B must be uint8 or uint4");

    // This is a tag that contains information about the operator used in the GEMM.
    using OperatorInfo = arch::DetagOperator<Operator_>;

    // This is the operator used in the GEMM.
    using Operator = typename OperatorInfo::Operator;

    // This is a static assertion that checks if the quantization operator is per-column scale only.
    static_assert(OperatorInfo::QuantOp == WeightOnlyQuantOp::
