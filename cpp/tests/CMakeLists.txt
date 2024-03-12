cmake_minimum_required(VERSION 3.14)
project(tensorrt_llm)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1)
FetchContent_MakeAvailable(googletest)

add_custom_target(google-tests)

function(add_gtest test_name test_src)
  add_executable(${test_name} ${test_src})

  if(NOT WIN32)
    target_link_libraries(${test_name} PUBLIC gtest_main gmock_main nvonnxparser nvinfer_plugin_tensorrt_llm)
  else()
    target_link_libraries(${test_name} PUBLIC gtest_main gmock_main nvonnxparser nvinfer_plugin_tensorrt_llm STATIC_TARGET)
  endif()

  target_compile_features(${test_name} PRIVATE cxx_std_17)
  target_compile_definitions(${test_name}
                             PUBLIC TOP_LEVEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/..")
  gtest_discover_tests(${test_name} PROPERTIES ENVIRONMENT "CUDA_MODULE_LOADING=LAZY" DISCOVERY_TIMEOUT 30)
  add_dependencies(google-tests ${test_name})
endfunction()

add_gtest(attentionKernelTest runtime/transposeKVKernelTest.cpp)
add_gtest(gptDecoderTest runtime/gptDecoderTest.cpp)
add_gtest(gptDecoderBatchTest runtime/gptDecoderBatchTest.cpp)
add_gtest(gptSessionTest runtime/gptSessionTest.cpp)
add_gtest(allocatorTest common/allocatorTest.cpp)
add_gtest(memoryUtilsTest common/memoryUtilsTest.cu)
add_gtest(quantizationTest common/quantizationTest.cpp)
add_gtest(stringUtilsTest common/stringUtilsTest.cpp)
add_gtest(tllmExceptionTest common/tllmExceptionTest.cpp)
add_gtest(tensorTest common/tensorTest.cpp)
add_gtest(stlUtilsTest common/stlUtilsTest.cpp)
add_gtest(tllmRuntimeTest runtime/tllmRuntimeTest.cpp)
add_gtest(tllmBuffersTest runtime/tllmBuffersTest.cpp)
add_gtest(bufferManagerTest runtime/bufferManagerTest.cpp)
add_gtest(runtimeKernelTest runtime/runtimeKernelTest.cpp)
add_gtest(samplingTest runtime/samplingTest.cpp)
add_gtest(torchTest runtime/torchTest.cpp)
target_link_libraries(torchTest PUBLIC ${TORCH_LIBRARIES})

if(BUILD_BATCH_MANAGER)
  add_subdirectory(batch_manager)
endif()
