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
#include "cutlass_extensions/gemm/threadblock/default_dq_mma_multistage.h"
#include "cutlass_extensions/gemm/threadblock/default_dq_mma_pipelined.h"

namespace cutlass
{
namespace gemm
{
namespace threadblock
{

////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output (OperatorClass TensorOp), bf16 activation & bf16 weight
template <
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB>
struct DefaultMma<bfloat16_t, LayoutA, kAlignmentA, bfloat16_t, LayoutB, kAlignmentB, ElementAccumulator,
    layout::RowMajor, arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape, InstructionShape, 2, Operator, false,
    SharedMemoryClear, GatherA, GatherB>
{

private:
    // Conversions only needed pre-ampere. This will trigger mma pipeline, so we convert before STS.
    static constexpr bool arch_has_bf16_mma = ArchTag::kMinComputeCapability >= 80;
    using MmaElementA = typename platform::conditional<arch_has_bf16_mma, bfloat16_t, half_t>::type;
    using MmaElementB = typename platform::conditional<arch_has_bf16_mma, bfloat16_t, half_t>::type;

public:
    // Define the MmaCore components
    using MmaCore =
        typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape, WarpShape, InstructionShape, MmaElementA,
            LayoutA, MmaElementB, LayoutB, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp, 2, Operator>;

    using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>, bfloat16_t, LayoutA, 1,
        typename MmaCore::IteratorThreadMapA, kAlignmentA, GatherA>;

    // Define iterators over tiles from the B operand
    using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>, bfloat16_t, LayoutB, 0,
        typename MmaCore::IteratorThreadMapB, kAlignmentB, GatherB>;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<typename MmaCore::Shape, IteratorA,
        typename MmaCore::SmemIteratorA, IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
        layout::RowMajor, typename MmaCore::MmaPolicy>;
};

// bf16 x bf16 specialization on Ampere to use mma multistage for 2 stage. Helps avoid reg spills on
// large tile when not enough shared mem is present to do 3+ stage
template <
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index
