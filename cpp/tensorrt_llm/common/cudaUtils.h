
This is a C++ header file that provides utilities for working with CUDA and cuBLAS. It includes functions for checking CUDA and cuBLAS errors, synchronizing the CUDA device, and printing debug information. It also includes type definitions for various data types used in CUDA and cuBLAS, as well as macros for checking CUDA errors and synchronizing the CUDA device.

The file begins with a license header that specifies the Apache License, Version 2.0. It also includes several include statements for various CUDA and cuBLAS headers, as well as standard C++ headers.

The `CUBLAS_WORKSPACE_SIZE` constant is defined as 32 MB, which is the size of the workspace used by cuBLAS for gemm operations.

The `half4` struct is defined as a struct with four `half` fields, aligned on a 4-byte boundary.

The `CublasDataType` enum is defined as an enumeration of various data types supported by cuBLAS, including `FLOAT_DATATYPE`, `HALF_DATATYPE`, `BFLOAT16_DATATYPE`, `INT8_DATATYPE`, and `FP8_DATATYPE`.

The `TRTLLMCudaDataType` enum is defined as an enumeration of various data types supported by TensorRT, including `FP32`, `FP16`, `BF16`, `INT8`, and `FP8`.

The `OperationType` enum is defined as an enumeration of various data types supported by TensorRT, including `FP32`, `FP16`, `BF16`, `INT8`, and `FP8`.

The `_cudaGetErrorEnum` function is defined as a function that returns a string representation of a CUDA error code.

The `_cudaGetErrorEnum` function is defined as a function that returns a string representation of a cuBLAS error code.

The `check` function is defined as a template function that throws an exception if a given result is not equal to zero. It takes a result, a function name, a file name, and a line number as arguments.

The `check_cuda_error` macro is defined as a macro that calls the `check` function with the result of a CUDA function call, the name of the function, the file name, and the line number.

The `check_cuda_error_2` macro is defined as a macro that calls the `check` function with the result of a CUDA function call, the name of the function, a file name, and a line number.

The `isCudaLaunchBlocking` function is defined as a function that returns a boolean value indicating whether CUDA launch is blocking.

The `syncAndCheck` function is defined as a function that synchronizes the CUDA device and checks for errors. It takes a file name and a line number as arguments.

The `sync_check_cuda_error` macro is defined as a macro that calls the `syncAndCheck` function with the file name and line number.

The `PRINT_FUNC_NAME_` macro is defined as a macro that prints the name of the current function to the console.

The `packed_type` struct template is defined as a struct template that provides a type that is packed on a 16-byte boundary. It is specialized for various data types, including `float`, `half`, `__nv_bfloat16`, and `__nv_fp8_e4m3`.

The `num_elems` struct template is defined as a struct template that provides the number of elements in a given data type. It is specialized for various data types, including `float`, `float2`, `float4`, `half`, `half2`, `__nv_bfloat16`, and `__nv_fp8_e4m3`.

The `packed_as` struct template is defined as a struct template that provides a type that is packed on a 16-byte boundary and has a given number of elements. It is specialized for various data types, including `float`, `half`, `__nv_bfloat16`, and `__nv_fp8_e4m3`.

The `CudaDataType` struct template is defined as a struct template that provides the CUDA data type for a given data type. It is specialized for various data types, including `float` and `half`.

The `getSMVersion` function is defined as a function that returns the SM version of the current CUDA device.

The `getDevice` function is defined as a function that returns the ID of the current CUDA device.

The `getDeviceCount` function is defined as a function that returns the number of CUDA devices in the system.

The `getDeviceMemoryInfo` function is defined as a function that returns the free and total amount of memory on the current CUDA device.

The `divUp` function is defined as a function that returns the smallest integer greater than or equal to the quotient of two integers.

The `ceilDiv` function template is defined as a function template that returns the smallest integer greater than or equal to the quotient of two integers.

The `printAbsMean` function template is defined as a function template that prints the absolute mean, sum, and maximum value of a given array of data. It takes a data array, its size, a CUDA stream, and an optional name as arguments.

The `printToStream` function template is defined as a function template that prints the contents of a given array of data to a given stream. It takes a data array, its size, and a stream as arguments.

The `printToScreen` function template is defined as a function template that prints the contents of a given array of data to the console. It takes a data array and its size as arguments.

The `print2dToStream` function template is defined as a function template that prints the contents of a given 2D array of data to a given stream. It takes a data array, its rows, columns, stride, and a stream as arguments.

The `print2dToScreen` function template is defined as a function template that prints the contents of a given 2D array of data to the console. It takes a data array, its rows, columns, stride as arguments.

The `print_float_` function is defined as a function that prints a given float value to the console.

The `print_element_` function template is defined as a function template that prints a given element of a given data type to the console. It is specialized for various data types, including `float`, `half`, `__nv_bfloat16`, `uint32
