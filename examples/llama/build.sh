# Build the model using the specified configuration.
# The model will be saved in the specified output directory.

# Parameters:
# model_dir: The directory containing the pre-trained model.
# dtype: The data type to be used for the model.
# use_gpt_attention_plugin: Whether to use the GPT attention plugin.
# enable_context_fmha: Whether to enable contextual feedforward multihead attention.
# use_gemm_plugin: Whether to use the GEMM plugin.
# rotary_base: The base for rotary embeddings.
# vocab_size: The size of the vocabulary.
# max_batch_size: The maximum batch size.
# int8_kv_cache: Whether to use int8 key-value cache.
# use_weight_only: Whether to use weight-only quantization.
# output_dir: The directory where the built model will be saved.

python build.py \
    --model_dir glaiveai/glaive-coder-7b \
    --dtype float16 \
    --use_gpt_attention_plugin float16 \
    --enable_context_fmha \
    --use_gemm_plugin float16 \
    --rotary_base 1000000 \
    --vocab_size 32016 \
    --max_batch_size 128 \
    --int8_kv_cache \
    --use_weight_only \
    --output_dir ./glaive_multitask_tensorrt
