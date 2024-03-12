#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_base.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/epilogue/threadblock/epilogue_with_visitor.h>

#include "cutlass_extensions/compute_occupancy.h"
#include "cutlass_extensions/epilogue/threadblock/epilogue_tensor_op_int32.h"
#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/gemm_configs.h"

#include "cutlass_extensions/gemm/kernel/default_int8_traits.h"
#include "cutlass_extensions/gemm/kernel/gemm_with_epilogue_visitor.h"

#include "tensorrt_llm/common/allocator.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.h"
#include "tensorrt_llm/kernels/cutlass_kernels/int8_gemm/int8_gemm.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttentionUtils.h"

#include <chrono>
#include <sstream>

namespace tk = tensorrt_llm::common;
namespace tkc = tensorrt_llm::cutlass_extensions;

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

template <typename T, typename arch, typename ThreadblockShape, typename WarpShape, int Stages>
void genericInt8GemmKernelLauncher(const int8_t* A, const int8_t* B, tk::QuantMode quantOption, const float* alphaCol,
    const float* alphaRow, T* C, int m, int n, int k, tkc::CutlassGemmConfig gemmConfig, char* workspace,
    size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    using ElementInput = int8_t;

    using ElementOutput_ =
        typename cutlass::platform::conditional<cutlass::platform::is_same<T, half>::value, cutlass::half_t, T>::type;
#ifdef ENABLE_BF16
    using ElementOutput =
        typename cutlass::platform::conditional<cutlass::platform::is_same<ElementOutput_, __nv_bfloat16>::value,
            cutlass::bfloat16_t, ElementOutput_>::type;
#else
    using ElementOutput = ElementOutput_;
#endif

    using ElementAccumulator = int32_t;
    using ElementCompute = float;

    using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    using OperatorClass = typename cutlass::gemm::kernel::Int8GemmArchTraits<arch>::OperatorClass;
    using InstructionShape = typename cutlass::gemm::kernel::Int8GemmArchTraits<arch>::InstructionShape;

    using DefaultGemmConf = typename cutlass::gemm::device::DefaultGemmConfiguration<OperatorClass, arch, ElementInput,
        ElementInput, ElementOutput, ElementCompute>;
    using GemmOp = typename DefaultGemmConf::Operator;
    using EpilogueOp = typename DefaultGemmConf::EpilogueOutputOp;

    // only TN is supported (s8 * s8 + s32)
    using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemm<ElementInput, cutlass::layout::RowMajor,
        DefaultGemmConf::kAlignmentA, ElementInput, cutlass::layout::ColumnMajor, DefaultGemmConf::kAlignmentB,
        ElementOutput, cutlass::layout::RowMajor, ElementAccumulator, OperatorClass, arch, ThreadblockShape, WarpShape,
        InstructionShape, EpilogueOp, ThreadblockSwizzle, Stages, true, GemmOp>::GemmKernel;

    using AlphaColTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
        cutlass::epilogue::threadblock::OutputTileOptimalThreadMap<
            typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Shape,
            typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Count,
            GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::kThreads,
            GemmKernel_::Epilogue::OutputTileIterator::kElementsPerAccess, cutlass::sizeof_bits<ElementOutput>::value>,
        ElementCompute>;

    // Epilogue visitor
    using EpilogueVisitor = typename cutlass::epilogue::threadblock::EpilogueVisitorPerRowPerCol<
