// This code defines several logging utilities for the NVIDIA TensorRT plugin framework.
// It includes error, warning, info, and verbose log streams, as well as functions for
// reporting validation failures, catching exceptions, and handling assertions.
// The code also includes functions for throwing and logging errors related to CUBLAS
// and CUDA.

// The `PLUGIN_CUDA_CHECK` macro is used to check for errors after CUDA API calls.
// If an error is detected, the function `throwCudaError` is called to log and throw
// a CudaError exception.

// The `LogStream` class template is used to define log streams for different severity
// levels (ERROR, WARNING, INFO, and VERBOSE). The `sync` method is used to flush the
// contents of the log stream to the logger.

// The `throwPluginError` function is used to throw a PluginError exception with a
// given message and status code. The `reportValidationFailure` function is used to
// log a validation failure message with file and line information.

// The `caughtError` function is used to log any uncaught exceptions with the error
// log stream. The `reportAssertion` function is used to log an assertion failure
// message with file and line information, and then abort the program.

// The `throwCublasError` function is used to throw a CublasError exception with a
// given message and status code.

// The `nvinfer1::plugin` namespace contains all the TensorRT plugin-related code.
// The `nvinfer1` namespace contains the TensorRT core library.
