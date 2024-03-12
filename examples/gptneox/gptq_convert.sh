# Clone the GPTQ-for-LLaMa repository
git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa.git GPTQ-for-LLaMa

# Install the required packages
pip install -r ./GPTQ-for-LLaMa/requirements.txt

# Set the CUDA_VISIBLE_DEVICES environment variable to specify the GPU to be used
CUDA_VISIBLE_DEVICES=0

# Run the GPTQ-for-LLaMa script to quantize the GPTNeoX model
python3 GPTQ-for-LLaMa/neox.py ./gptneox_model \
wikitext2 \
--wbits 4 \
--groupsize 128 \
--save_safetensors ./gptneox_model/gptneox-20b-4bit-gs128.safetensors

# This script quantizes the GPTNeoX model located in the ./gptneox_model directory using the GPTQ-for-LLaMa tool.
# The quantization process involves reducing the model's precision from 16-bit to 4-bit while maintaining its performance.
# The --wbits 4 flag specifies the target bit width, and the --groupsize 128 flag sets the group size for the quantization process.
# The quantized model will be saved as gptneox-20b-4bit-gs128.safetensors in the ./gptneox_model directory.
