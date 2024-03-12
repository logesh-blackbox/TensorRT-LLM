/*
 * This code includes a templated class named CutlassFpAIntBGemmRunner, which is used for performing GEMM (General Matrix Multiplication) operations with half-precision floating-point (FP16) data type (fpA) and 4-bit unsigned integer data type with 4 elements packed into one integer (uint4b_t) (intB).
 *
 * The class is implemented using Cutlass, a CUDA template library for high-performance linear algebra. The GEMM operation is performed using the WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS quantization scheme.
 *
 * The code is licensed under the Apache License, Version 2.0.
 */

#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{
// The templated class CutlassFpAIntBGemmRunner is defined here.
template class CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t,
    cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>;
} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm

