
python build.py --model_dir glaiveai/glaive-coder-7b \
		--dtype float16 \
		--use_gpt_attention_plugin float16 \
		--enable_context_fmha \
		--use_gemm_plugin float16 \
		--rotary_base  1000000 \
		--vocab_size 32016 \
		--max_batch_size 128 \
		--int8_kv_cache \
        --use_weight_only \
		--output_dir ./glaive_multitask_tensorrt
