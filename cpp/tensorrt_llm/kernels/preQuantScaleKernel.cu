// apply_per_channel_scale.h

#include <cuda_runtime.h>

namespace tensorrt_llm
{
namespace kernels
{

// Apply per-channel scale to the given input tensor.
template <typename T, int kProcessRows, typename AccessType>
__global__ void apply_per_channel_scale(T* smoothed_act, const T* act, const T* per_channel_scale, int rows, int cols);

// Launcher function for apply_per_channel_scale kernel.
template <typename T, int kProcessRows, typename AccessType = float4>
void apply_per_channel_scale_kernel_launcher_(
    T* smoothed_act, const T* act, const T* per_channel_scale, int rows, int cols, cudaStream_t stream = 0);

// Launcher function for apply_per_channel_scale kernel, with dynamic kProcessRows.
template <typename T>
void apply_per_channel_scale_kernel_launcher(
    T* smoothed_act, const T* act, const T* per_channel_scale, int rows, int cols, cudaStream_t stream);

// Instantiate apply_per_channel_scale_kernel_launcher for half type.
#define INSTANTIATE_PREQUANT_SCALE(T)                                                                                  \
    template void apply_per_channel_scale_kernel_launcher<T>(                                                          \
        T * smoothed_act, const T* act, const T* per_channel_scale, int rows, int cols, cudaStream_t stream)

INSTANTIATE_PREQUANT_SCALE(half);

} // namespace kernels
} // namespace tensorrt_llm



// preQuantScaleKernel.h

#include "apply_per_channel_scale.h"

namespace tensorrt_llm
{
namespace kernels
{

// Pre-quantization scale kernel.
template <typename T>
void preQuantScaleKernel(
    T* smoothed_act, const T* act, const T* per_channel_scale, int rows, int cols, cudaStream_t stream = 0);

} // namespace kernels
} // namespace tensorrt_llm



// preQuantScaleKernel.cu

#include "preQuantScaleKernel.h"

using namespace tensorrt_llm::kernels;

// Pre-quantization scale kernel implementation.
template <typename T>
void preQuantScaleKernel(
    T* smoothed_act, const T* act, const T* per_channel_scale, int rows, int cols, cudaStream_t stream)
{
    apply_per_channel_scale_kernel_launcher<T>(
        smoothed_act, act, per_channel_scale, rows, cols, stream);
}

// Explicitly instantiate preQuantScaleKernel for half type.
template void preQuantScaleKernel<half>(
    half* smoothed_act, const half* act, const half* per_channel_scale, int rows, int cols, cudaStream_t stream);



// preQuantScaleKernel.h

#include "apply_per_channel_scale.h"

namespace tensorrt_llm
{
namespace kernels
{

// Pre-quantization scale kernel.
template <typename T>
void preQuantScaleKernel(
    T* smoothed_act, const T* act, const T* per_channel_scale, int rows, int cols, cudaStream_t stream = 0);

} // namespace kernels
} // namespace tensorrt_llm



// preQuantScaleKernel.cu

#include "preQuantScaleKernel.h"

using namespace tensorrt_llm::kernels;

// Pre-quantization scale kernel implementation.
template <typename T>
void preQuantScaleKernel(
    T* smoothed_act, const T* act, const T* per_channel_scale, int rows, int cols, cudaStream_t stream)
{
    apply_per_channel_scale_kernel_launcher<T>(
        smoothed_act, act, per_channel_scale, rows, cols, stream);
}

// Explicitly instantiate preQuantScaleKernel for half type.
template void preQuantScaleKernel<half>(
    half* smoothed_act, const half* act, const half* per_channel_scale, int rows, int cols, cudaStream_t stream);



// preQuantScaleKernel.h

#include "apply_per_channel_scale.h"

namespace tensorrt_llm
{
namespace kernels
{

// Pre-quantization scale kernel.
template <typename T>
void preQuantScaleKernel(
    T* smoothed_act, const T* act, const T* per_channel_scale, int rows, int cols, cudaStream_t stream = 0);

} // namespace kernels
} // namespace tensorrt_llm



// preQuantScaleKernel.cu

#include "preQuantScaleKernel.h"

using namespace tensorrt_llm::kernels;

// Pre-quantization scale kernel implementation.
template <typename T>
void preQuantScaleKernel(
    T* smoothed_act, const T* act, const T* per_channel_scale, int rows, int cols, cudaStream_t stream)
{
    apply_per_channel_scale_kernel_launcher<T>(
        smoothed_act, act, per_channel_scale, rows, cols
