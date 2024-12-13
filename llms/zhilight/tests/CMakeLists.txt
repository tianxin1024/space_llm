cmake_minimum_required(VERSION 3.10)
project(tests VERSION 0.1)

enable_language(C)
enable_language(CXX)
# enable_language(CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_VISIBILITY_PRESET default)
set(CMAKE_CUDA_VISIBILITY_PRESET default)
# SET ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__CUDA_FP16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__ " )
# SET ( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__CUDA_FP16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__ " )
# SET ( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -D__CUDA_FP16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__  -rdynamic -Wl,-Bsymbolic -Wl,-Bsymbolic-functions" )
# SET ( CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS} -D__CUDA_FP16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__  -rdynamic -Wl,-Bsymbolic -Wl,-Bsymbolic-functions" )

# add_executable(test_allocator test_allocator.cpp)
# target_link_libraries(test_allocator LINK_PUBLIC bmengine)

add_executable(test_template test_template.cpp)
target_link_libraries(test_template LINK_PUBLIC bmengine)

add_executable(test_element test_element.cpp)
target_link_libraries(test_element LINK_PUBLIC bmengine)

add_executable(test_gemm test_gemm.cpp)
target_link_libraries(test_gemm LINK_PUBLIC bmengine)

add_executable(test_print_tensor test_print_tensor.cpp)
target_link_libraries(test_print_tensor LINK_PUBLIC bmengine)
