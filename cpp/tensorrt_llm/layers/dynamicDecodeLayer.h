class DynamicDecodeLayer : public BaseLayer
{
public:
    DynamicDecodeLayer(size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream, tc::IAllocator* allocator,
        bool is_free_buffer_after_forward, cudaDeviceProp* cuda_device_prop);

    DynamicDecodeLayer(DynamicDecodeLayer const& dynamic_decode_layer) = delete;

    // ...
};
