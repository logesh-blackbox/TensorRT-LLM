// SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass_extensions/arch/mma.h"

#include "cutlass_extensions/gemm/threadblock/dq_mma_multistage.h"
#include "cutlass_extensions/gemm/warp/default_mma_tensor_op.h"
#include "cutlass_extensions/gemm/warp/mma_tensorop_compute_B_with_f16.h"
#include "cutlass_extensions/tile_interleaved_layout.h"

#include "cutlass_extensions/gemm/threadblock/default_dq_mma.h"
#include "cutlass_extensions/transform/threadblock/fine_grained_scale_zero_iterator.h"

namespace cutlass
{
namespace gemm
{
namespace threadblock
{

////////////////////////////////////////////////////////////////////////////////

// This template specialization defines the `DefaultScaleIterators` class for the `DqMma` class.
// The `DefaultScaleIterators` class provides iterators over tiles from the scale operand.
// It has two specializations: one for fine-grained iterators and another for per-column iterators.
// The fine-grained iterators are used when the quantization operation is fine-grained, and the per-column
// iterators are used when the quantization operation is not fine-grained.
template <typename MmaShape, typename Element, typename Layout, WeightOnlyQuantOp QuantOp, int Alignment,
    typename Enable = void>
struct DefaultScaleIterators;

// Specialization for fine-grained iterators
template <typename MmaShape, typename Element, typename Layout, WeightOnlyQuantOp QuantOp, int Alignment>
struct DefaultScaleIterators<MmaShape, Element, Layout, QuantOp, Alignment, std::enable_if_t<isFinegrained(QuantOp)>>
{
    // Define the `IteratorScale` type as `FineGrainedScaleZeroIterator`.
    // The `FineGrainedScaleZeroIterator` class provides an iterator over tiles from the scale operand
    // with fine-grained quantization.
    using IteratorScale
        = cutlass::transform::threadblock::FineGrainedScaleZeroIterator<cutlass::MatrixShape<1, MmaShape::kN>, Element,
            Layout, 0, Alignment>;

    // Define the `SmemIteratorScale` type as `IteratorScale`.
    // This allows the `SmemIteratorScale` to be used in shared memory.
    using SmemIteratorScale = IteratorScale;
};

// Specialization for per-column iterators
template <typename MmaShape, typename Element, typename Layout, WeightOnlyQuantOp QuantOp, int Alignment>
struct DefaultScaleIterators<MmaShape, Element, Layout, QuantOp, Alignment, std::enable_if_t<!isFinegrained(QuantOp)>>
{
    // Define the `IteratorScaleThreadMap` type as `PitchLinearStripminedThreadMap`.
    // The `PitchLinearStripminedThreadMap` class provides a thread map for iterating over tiles from the scale operand
    // in a strip-mined manner.
    static_assert((MmaShape::kN % Alignment) == 0, "");

private:
    using IteratorScaleThreadMap = transform::PitchLinearStripminedThreadMap<layout::PitchLinearShape<MmaShape::kN, 1>,
        MmaShape::kN / Alignment, Alignment>;

public:
    // Define the `IteratorScale` type as `PredicatedTileIterator`.
    // The `PredicatedTileIterator` class provides an iterator over tiles from the scale operand
    // with per-column quantization.
    using IteratorScale = cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<1, MmaShape::kN>,
        Element, Layout, 0, IteratorScaleThreadMap, Alignment>;

    // Define the `SmemIteratorScale` type as `IteratorScale`.
    // This allows the `SmemIteratorScale` to be used in shared memory.
    using SmemIteratorScale = IteratorScale;
};

////////////////////////////////////////////////////////////////////////////////

// This is the main template specialization for the `DqMma` class.
// It defines a threadblock-scoped pipelined matrix multiply that performs dequantization and interleaving of the B operand
// after loading it from shared memory.
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

