cmake_minimum_required(VERSION 3.10)

project(pro LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_PATH "/usr/local/cuda")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

set(CMAKE_CUDA_ARCHITECTURES "80;86")

# CUTLASS 头文件
set(CUTLASS_HEADER_DIR ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/include)

set(CUDA_HEADER_DIRS 
    ${PROJECT_SOURCE_DIR}
    ${CUDA_PATH}/include
    ${CUTLASS_HEADER_DIR})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(COMMON_LIB_DIRS 
    ${CUDA_PATH}/lib64)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=compute_75 -gencode arch=compute_80,code=compute_80 -gencode arch=compute_86,code=compute_86")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-fPIC --expt-extended-lambda")
# message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

include_directories(${CUDA_HEADER_DIRS})
link_directories(${COMMON_LIB_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)


add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)

