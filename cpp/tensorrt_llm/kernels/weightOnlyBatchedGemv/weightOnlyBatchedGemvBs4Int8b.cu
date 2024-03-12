/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * This file contains the implementation of WeightOnlyBatchedGemvKernelLauncher for various configurations of
 * WeightOnlyQuantType, WeightOnlyPerChannel, and activation functions.
 *
 * The kernel launcher is a function that prepares data and invokes the GPU kernel for the batched GEMM operation
 * with weight-only quantization. The configurations include different combinations of Int8b weight quantization,
 * per-channel or per-tensor weight layout, and activation functions such as GeluActivation, ReluActivation, and
 * IdentityActivation.
 *
 * The template parameters specify the configuration for each kernel launcher instance:
 * 1. WeightOnlyQuantType: Quantization type for weights (Int8b in this case).
 * 2. WeightOnlyPerChannel: Whether to use per-channel or per-tensor weight layout.
 * 3. ActivationFunction: The activation function to be applied after the GEMM operation.
 * 4. UseBias: Whether to use bias in the GEMM operation.
 * 5. UseFusedActivation: Whether to fuse the activation function with the GEMM operation.
 * 6. M: Number of rows in the weight matrix.
 * 7: K: Number of columns in the weight matrix.
 * 8: N: Number of columns in the input matrix.
 *
 * The code instantiates multiple kernel launcher templates for different configurations, allowing the compiler to
 * generate optimized code for each specific case.
 */
