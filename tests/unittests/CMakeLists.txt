include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)

FetchContent_MakeAvailable(googletest)

add_executable(unittest 
    test_tensor.cu
    test_int8.cu
    test_attention_kernels.cu)

# automatic discovery of unit tests
target_link_libraries(unittest PUBLIC "${TORCH_LIBRARIES}" gtest_main)
target_compile_features(unittest PRIVATE cxx_std_14)

target_link_libraries(  # Libs for test_attention_kernels
  unittest PUBLIC -lcudart -lcurand gpt_kernels gtest tensor memory_utils unfused_attention_kernels logger)

target_link_libraries(  # Libs for test_int8
  unittest PUBLIC tensor memory_utils transpose_int8_kernels logger)

target_link_libraries(  # Libs for test_tensor 
  unittest PUBLIC tensor cuda_utils logger)

add_executable(test_activation test_activation.cu)
target_link_libraries(test_activation PUBLIC -lcublas -lcublasLt -lcudart
    activation_kernels memory_utils cuda_utils logger)

add_executable(test_gemm test_gemm.cu)
target_link_libraries(test_gemm PUBLIC -lcublas -lcudart -lcurand gemm cublasMMWrapper tensor cuda_utils logger)

add_executable(test_gpt_kernels test_gpt_kernels.cu)
target_link_libraries(test_gpt_kernels PUBLIC gpt_kernels memory_utils tensor cuda_utils logger)

add_executable(test_layernorm test_layernorm.cu)
target_link_libraries(test_layernorm PUBLIC -lcublasLt layernorm_kernels memory_utils cuda_utils logger)
