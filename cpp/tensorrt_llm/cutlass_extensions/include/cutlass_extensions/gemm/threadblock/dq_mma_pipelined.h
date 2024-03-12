/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass_extensions/gemm/threadblock/dq_mma_base.h"
#include "cutlass_extensions/gemm/warp/mma_tensorop_dequantizer.h"
#include "cutlass_extensions/interleaved_numeric_conversion.h"

#include "cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.h"
#include "cutlass_extensions/gemm_configs.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass
{
namespace gemm
{
namespace threadblock
{

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Data type for the scales
    typename IteratorScale_,
    /// Iterators over scales in shared memory
    typename SmemIteratorScale_,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Converter for B matrix applied immediately after the LDG (before STS)
    typename TransformBAfterLDG_,
    /// Converter for B matrix applited immediately after the LDS
    typename TransformBAfterLDS_,
    /// The quantization operator being used
    WeightOnlyQuantOp QuantOp_,
    /// Used for partial specialization
    typename Enable = bool>
class DqMmaPipelined : public DqMmaBase<Shape_, Policy_, typename SmemIteratorScale_::Element, 2, QuantOp_>
{
public:
    ///< Base class
    using Base = DqMmaBase<Shape_, Policy_, typename SmemIteratorScale_::Element, 2, QuantOp_>;

    using Shape = Shape_;         ///< Size of the Gemm problem - concept: gemm::GemmShape<>
    using IteratorA = IteratorA_; ///< Iterates over tiles of A operand in global memory
    using IteratorB = IteratorB_; ///< Iterates over tiles of B operand in global memory
    using ElementC = ElementC_;   ///< Data type of accumulator matrix
    using LayoutC = LayoutC_;     ///< Layout of accumulator matrix
    using Policy = Policy_;       ///< Policy describing tuning details

    using IteratorScale = IteratorScale_;
    using ElementScale = typename IteratorScale::Element;
    using LayoutScale = typename IteratorScale::Layout;

    using SmemIteratorA = SmemIteratorA_;
    using SmemIteratorB = SmemIteratorB_;
    using SmemIteratorScale = SmemIteratorScale_;

    using TransformBAfterLDG = TransformBAfterLDG_;
    using TransformBAfterLDS = TransformBAfterLDS_;

    static constexpr WeightOnlyQuantOp QuantOp = QuantOp_;

    //
    // Dependent types
    //

    /// Fragment of operand A loaded from global memory
    using Fr
