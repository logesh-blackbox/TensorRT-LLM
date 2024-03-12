/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

// The template class for the FP16-INT8 GEMM kernel.
template <typename FpA, typename IntB, typename Op>
class CutlassFpAIntBGemmRunner
{
public:
    // The constructor.
    CutlassFpAIntBGemmRunner() = default;

    // The destructor.
    ~CutlassFpAIntBGemmRunner() = default;

    // The function to run the FP16-INT8 GEMM kernel.
    void run(const FpA* fpA, const IntB* intB, FpA* fpC, int m, int n, int k, float fpA_scale, float fpB_scale, float fpC_scale);
};

// The implementation of the run function.
template <typename FpA, typename IntB, typename Op>
void CutlassFpAIntBGemmRunner<FpA, IntB, Op>::run(const FpA* fpA, const IntB* intB, FpA* fpC, int m, int n, int k, float fpA_scale, float fpB_scale, float fp
