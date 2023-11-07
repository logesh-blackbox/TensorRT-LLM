## Instructions to run the benchmark:

1. Follow the steps in this `docs/source/installation.md` document to build the tensorRT docker image (`Fetch the Sources` and `Build TensorRT-LLM in One Step` steps). Go into the bash cli of the container and execute the following steps.

2. Switch to the examples folder : `cd examples/llama`

3. Build the model for inference by executing : `bash build.sh`. Remove the `int8_kv_cache` flag if you are not building for int8 inference. Update the `model_dir` with your model.

4. Once the build is completed. Run `bash run_benchmark.sh` to run the inference benchmark. Make sure to update the `model_folder` in the script with the output of step 3.