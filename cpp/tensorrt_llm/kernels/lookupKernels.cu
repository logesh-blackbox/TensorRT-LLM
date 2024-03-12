template <typename T, typename Idx>
__global__ void lookup_kernel(T* output, const Idx* input, const T* weight, const Idx batch_size, const Idx offset,
    const Idx size, const int n_embed)
{
    __shared__ T shared_weight[BLOCK_SIZE * n_embed];
    Idx thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    Idx batch_id = thread_id / n_embed;
    Idx emb_id = thread_id % n_embed;
    if (batch_id < batch_size && emb_id < n_embed)
    {
        Idx word_index = input[batch_id] - offset;
        if (word_index >= 0 && word_index < size)
        {
            shared_weight[emb_id] = weight[word_index * n_embed + emb_id];
        }
        else
        {
            shared_weight[emb_id] = T(0.f);
        }
        __syncthreads();
        output[batch_id * n_embed + emb_id] = shared_weight[emb_id];
    }
}

template <typename T, typename Idx>
void invokeLookUp(T* out, const Idx* input, const T* weight, const Idx batch_size, const Idx offset, const Idx size,
    const int n_embed, cudaStream_t stream)
{
    dim3 grid(batch_size);
    dim3 block(n_embed);
    lookup_kernel<T, Idx><<<grid, block, BLOCK_SIZE * sizeof(T), stream>>>(out, input, weight, batch_size, offset, size, n_embed);
}

#define BLOCK_SIZE 256
INSTANTIATE_LOOK_UP(float, int);
INSTANTIATE_LOOK_UP(half, int);
#ifdef ENABLE_BF16
INSTANTIATE_LOOK_UP(__nv_bfloat16, int);
#endif
