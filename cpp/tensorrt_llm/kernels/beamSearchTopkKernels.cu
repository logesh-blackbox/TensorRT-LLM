


#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/kernels/beamSearchTopkKernels.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
__device__ __forceinline__ T apply_length_penalty(T log_prob, int length, float length_penalty)
{
    // score = log(prob) / (length)^length_penalty.
    if (length_penalty == 0.0f || length == 1)
    {
        return log_prob;
    }
    return log_prob / static_cast<T>(powf((float) length, length_penalty));
}

template <typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void beam_topK_kernel(const T* log_probs, int* topk_tmp_id_buf, T* topk_tmp_val_buf, const bool* finished,
        const int* sequence_lengths, const int vocab_size, T diversity_rate, float length_penalty)
{
    typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int thread_id = threadIdx.x;
    int block_id = blockIdx.x; // batch beam index.
    TopK<T, MAX_K> partial;

    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

#pragma unroll
    for (int i = 0; i < MAX_K; ++i)
    {
        partial.p[i] = -1;
        partial.u[i] = -MAX_T_VAL;
    }

#pragma unroll
    for (int elem_id = thread_id; elem_id < vocab_size; elem_id += THREADBLOCK_SIZE)
    {
        int index = elem_id + block_id * vocab_size;
        T score = length_penalty == 0.0f
            ? log_probs[index]
            : apply_length_penalty(log_probs[index],
                finished[block_id] ? sequence_lengths[block_id] : sequence_lengths[block_id] + 1, length_penalty);
        partial.insert(score, index);
    }

    TopK<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

    if (thread_id == 0)
    {
        int index = block_id * MAX_K;

#pragma unroll
        for (int i = 0; i < MAX_K; ++i)
        {
            topk_tmp_id_buf[index + i] = total.p[i];
            topk_tmp_val_buf[index + i] = total.u[i] + diversity_rate * (T) i;
        }
    }
}

template <typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void batch_topK_kernel(int* topk_tmp_id_buf, T* topk_tmp_val_buf, int* id_buf)
{
    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
    TopK<T, MAX_K> partial;
    if (thread_id == 0)
    {
        for (int i = 0; i < MAX_K; ++i)
        {
            partial.p[i] = -1;
            partial.u[i] = -MAX_T_VAL;
        }

        int index = block_id * MAX_K * MAX_K;
        for (int i = 0; i < MAX_K * MAX_K; i++)
        {
            partial.insert((T) topk_tmp_val_buf[index + i], topk_tmp_id_buf[index + i]);
        }

        index = block_id * MAX_K;
        for (int i = 0; i < MAX_K; i++)
        {
            id_buf[index + i] = partial.p[i];
        }
    }
}

template <typename T, int MAX_K, int THREADBLOCK_SIZE
