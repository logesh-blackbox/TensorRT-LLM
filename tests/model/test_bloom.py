
This is a Python script that tests the Bloom model using TensorRT LLM. The script defines several functions to generate the Bloom model, the TensorRT LLM network, and the TensorRT LLM runtime. It then uses these functions to test the Bloom model with different configurations.

The script uses the `unittest` module to define test cases and assert that the results of the TensorRT LLM model match the results of the reference model. It also uses the `parameterized` module to run the same test case with different configurations.

The script tests the Bloom model with and without the GPT attention plugin, with different context FMHA types, and with and without input padding removal. It also tests the Bloom model with greedy search and generates output sequences of different lengths.

Overall, this script provides a comprehensive test suite for the Bloom model using TensorRT LLM.
