/*
 * This code is a C++ header file that includes a templated class named CutlassFpAIntBGemmRunner. 
 * The class is used for performing a general matrix multiplication (GEMM) operation with floating-point 
 * type A (half) and integer type B (uint8_t). The GEMM operation is implemented using the Cutlass library, 
 * which is a CUDA-focused linear algebra library developed by NVIDIA.
 *
 * The class is defined within the namespace tensorrt_llm::kernels::cutlass_kernels.
 *
 * The template parameters are:
 * 1. half: The floating-point type A for the GEMM operation.
 * 2. uint8_t: The integer type B for the GEMM operation.
 * 3. cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY: The quantization operation for the weights.
 *
 * The header file is protected by the Apache License, Version 2.0, which provides users with permissions 
 * and limitations regarding the use, reproduction, and distribution of the code.
 */

#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{
// The templated class definition is provided here.
template class CutlassFpAIntBGemmRunner<half, uint8_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>;
} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm

