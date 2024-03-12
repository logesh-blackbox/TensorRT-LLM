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
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/mma_base.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "cutlass_extensions/weight_only_quant_op.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass
{
namespace gemm
{
namespace threadblock
{

////////////////////////////////////////////////////////////////////////////////
// SFINAE trick so I can keep the same loop code for Volta and dispatch to the
// correct warp level mma. On volta, all data is stored to shared memory as FP16.
template <typename WarpMma, int kExpansionFactor = 1>
CUTLASS_DEVICE void run_warp_mma(WarpMma& warp_mma, typename WarpMma::FragmentC& D,
    typename WarpMma::FragmentA const& A, typename WarpMma::FragmentB const& B, typename WarpMma::FragmentC const& C,
    const int warp_tileB_k_offset)
{
    warp_mma(D, A, B, C);
}

template <typename WarpMma, int kExpansionFactor = WarpMma::kExpansionFactor>
CUTLASS_DEVICE void run_warp_mma(WarpMma& warp_mma, typename WarpMma::FragmentC& D,
    typename WarpMma::TransformedFragmentA const& A, typename WarpMma::TransformedFragmentB const& B,
    typename WarpMma::FragmentC const& C, const int warp_tileB_k_offset)
{
    warp_mma(D, A, B, C, warp_tileB_k_offset);
}

////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// The type of the scales
    typename ElementScale_,
    /// Number of stages,
    int Stages,
    /// The dequantizing op to be performed.
    WeightOnlyQuantOp DequantOp,
    /// Used for partial specialization,
    typename Enable = bool>
class DqMmaBase
{
public:
    ///< Size of the Gemm problem - concept: gemm::GemmShape<>
    using Shape = Shape_;

    ///< Policy describing tuning details
    using Policy = Policy_;

    ///< Type of the scale to be loaded
    using ElementScale = ElementScale_;

    static_assert(DequantOp != WeightOnlyQuantOp::UNDEFINED, "");

    // Finegrained scales get streamed in via cp.async
    static constexpr int ScalebiasStages = isFinegrained(DequantOp) ? Stages : 1;
    // We always have scales.
    static constexpr int ScaleElementsPerStage = Shape::kN;
    // We sometimes have a bias
    static constexpr int BiasElementsPerStage = hasZero(DequantOp) ? Shape::kN : 0;

    //
    // Dependent types
    //

    /// Warp-level Mma
    using Operator = typename Policy::Operator;

    /// Shape describing the overall GEMM computed from shared memory
    /// by each warp.
    using WarpGemm = typename Policy::Operator::Shape;

    /// Shape describing the number of warps filling the CTA
    using WarpCount = GemmShape<Shape::kM / WarpGemm::kM, Shape::kN / WarpGemm::kN, Shape::kK / WarpGemm::kK
