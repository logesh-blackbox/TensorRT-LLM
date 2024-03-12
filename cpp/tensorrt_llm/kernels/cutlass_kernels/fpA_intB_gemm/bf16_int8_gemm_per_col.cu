/*
 * This code is a C++ header file that includes a templated class for a Gemm (General Matrix Multiplication) operation.
 * The Gemm operation is performed using the Cutlass library, a CUDA-based templated linear algebra library.
 *
 * The template parameters are:
 * 1. __nv_bfloat16: The data type of the first matrix (A).
 * 2. uint8_t: The data type of the second matrix (B).
 * 3. cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY: The quantization operation for the second matrix (B).
 *
 * The code is licensed under the Apache License, Version 2.0.
 *
 * The code is distributed under the "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
 * See the License for the specific language governing permissions and limitations under the License.
 */

#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{
#ifdef ENABLE_BF16
// Instantiates the CutlassFpAIntBGemmRunner class for __nv_bfloat16 and uint8_t data types.
template class CutlassFpAIntBGemmRunner<__nv_bfloat16, uint8_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>;
#endif
} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm

