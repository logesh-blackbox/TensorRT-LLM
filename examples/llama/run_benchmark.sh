batch_size=4
max_output_len=40
model_folder=glaive_multitask_tensorrt
tokenizer=glaiveai/glaive-coder-7b



declare -a batch_sizes=(
                4
                5
                6
                8
                16
                32
                64
                128
                )

for bs in "${batch_sizes[@]}":
do
    echo "Running prediction for Model: ${model_folder}, Batch_size: ${bs}"
    python benchmark.py --max_output_len=$max_output_len \
                        --tokenizer_dir=$tokenizer  \
                        --engine_dir=$model_folder \
                        --batch_size=$bs
done
