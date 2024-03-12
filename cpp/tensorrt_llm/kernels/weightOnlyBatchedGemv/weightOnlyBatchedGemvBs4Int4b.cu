#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/kernel.h"

namespace tensorrt_llm
{
namespace kernels
{

template <typename QuantType, typename WeightLayout, typename Activation, bool Transpose, bool IsBias,
          int M, int K, int N>
struct WeightOnlyBatchedGemvKernelLauncher
{
    static void launch(const QuantType* d_weight, const QuantType* d_input, QuantType* d_output, int batchSize,
                       int m, int k, int n)
    {
        // Allocate memory on the GPU using CUDA Unified Memory
        auto d_weight_managed = tensorrt_llm::cuda::makeManaged<QuantType[]>(m * k);
        auto d_input_managed = tensorrt_llm::cuda::makeManaged<QuantType[]>(batchSize * k);
        auto d_output_managed = tensorrt_llm::cuda::makeManaged<QuantType[]>(batchSize * n);

        // Copy data to the GPU using CUDA Unified Memory
        tensorrt_llm::cuda::copy(d_weight_managed.get(), d_weight, m * k * sizeof(QuantType));
        tensorrt_llm::cuda::copy(d_input_managed.get(), d_input, batchSize * k * sizeof(QuantType));

        // Launch the GPU kernel using CUDA
        weightOnlyBatchedGemvKernel<<<1, 1>>>(d_weight_managed.get(), d_input_managed.get(), d_output_managed.get(),
                                              batchSize, m, k, n, Transpose, IsBias);

        // Copy data back to the host using CUDA Unified Memory
        tensorrt_llm::cuda::copy(d_output, d_output_managed.get(), batchSize * n * sizeof(QuantType));
    }
};

} // namespace kernels
} // namespace tensorrt_llm

