//
// Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/numeric_conversion.h"
#include "tensorrt_llm/common/quantization.h"

namespace cutlass
{
namespace epilogue
{
namespace threadblock
{

template <typename ThreadblockShape_, int ThreadCount, typename ScaleTileIterator_, typename OutputTileIterator_,
    typename ElementAccumulator_, typename ElementCompute_, typename ElementwiseFunctor_, bool UseMasking_ = false>
class EpilogueVisitorPerRowPerCol
{
public:
    using ThreadblockShape = ThreadblockShape_;
    static int const kThreadCount = ThreadCount;

    using ScaleTileIterator = ScaleTileIterator_;
    using OutputTileIterator = OutputTileIterator_;
    using ElementwiseFunctor = ElementwiseFunctor_;

    static int const kIterations = OutputTileIterator::kIterations;
    static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

    using ElementOutput = typename OutputTileIterator::Element;
    using LayoutOutput = cutlass::layout::RowMajor;
    using ElementAccumulator = ElementAccumulator_;

    using AlphaScaleElementType = typename ScaleTileIterator::Element;

    using ElementCompute = ElementCompute_;
    using AccumulatorFragment = Array<ElementAccumulator, kElementsPerAccess>;
    using ComputeFragment = Array<ElementCompute_, kElementsPerAccess>;
    using OutputVector = Array<ElementOutput, kElementsPerAccess>;

    static int const kThreadsPerRow = OutputTileIterator::ThreadMap::Detail::kAccessWidth;
    static bool const kHasMultiStepsInRow = (OutputTileIterator::ThreadMap::Iterations::kColumn > 1);

    /// Argument structure
    struct Arguments
    {

        typename ElementwiseFunctor::Params elementwise;
        int64_t batch_stride_alpha;
        int64_t batch_stride_C;
        int64_t batch_stride_D;

        //
        // Methods
        //
        Arguments()
            : batch_stride_alpha(0)
            , batch_stride_C(0)
            , batch_stride_D(0)
        {
        }

        Arguments(typename ElementwiseFunctor::Params elementwise_)
            : elementwise(elementwise_)
            , batch_stride_alpha(0)
            , batch_stride_C(0)
            , batch_stride_D(0)
        {
        }

        Arguments(typename ElementwiseFunctor::Params elementwise_, int64_t batch_stride_alpha_,
            int64_t batch_stride_C_, int64_t batch_stride_D_)
            : elementwise(elementwise_)
            , batch_stride_alpha(batch_stride_alpha_)
            , batch_stride_C(batch_stride_C_)
            , batch_stride_D(batch_stride_D_)
        {
        }
    };

    struct Params
    {

        typename ElementwiseFunctor::Params elementwise;
        int64_t batch_stride_alpha;
        int64_t batch_stride_C;
        int64_t batch_stride_D;

        //
        // Methods
        //
        CUTLASS_HOST_DEVICE
        Params() {}

        CUTLASS_HOST_DEVICE
        Params(Arguments const& args)
            : elementwise(args.elementwise)
            , batch_stride_alpha(args.batch_stride_alpha)
            ,
