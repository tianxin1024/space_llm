include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)

FetchContent_MakeAvailable(googletest)

add_executable(unittest 
    test_tensor.cu
    test_int8.cu)

# automatic discovery of unit tests
target_link_libraries(unittest PUBLIC "${TORCH_LIBRARIES}" gtest_main)
target_compile_features(unittest PRIVATE cxx_std_14)

target_link_libraries(  # Libs for test_tensor and test_int8
  unittest PUBLIC tensor cuda_utils logger)

add_executable(test_activation test_activation.cu)
target_link_libraries(test_activation PUBLIC -lcublas -lcublasLt -lcudart
    activation_kernels memory_utils cuda_utils logger)


