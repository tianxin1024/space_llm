cmake_minimum_required(VERSION 3.10)
project(bmengine VERSION 0.1)

enable_language(C)
enable_language(CXX)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;86")
endif()
if(NOT APPLE)
  enable_language(CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUBLAS_LIBRARY cublas ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUBLASLT_LIBRARY cublasLt ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUBRAND_LIBRARY curand_static ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CULIBOS_LIBRARY culibos ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

file(GLOB_RECURSE FILES_BMENGINE "*.cpp")
file(GLOB_RECURSE FILES_BMENGINE_CUDA "*.cu")
file(GLOB_RECURSE FILES_BMENGINE_HEADER "*.h")

add_library(bmengine STATIC
    ${FILES_BMENGINE}
    ${FILES_BMENGINE_CUDA}
)

target_include_directories(bmengine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bmengine
    ${CUDART_LIBRARY}
    ${CUBLAS_LIBRARY}
    ${CUBLASLT_LIBRARY}
    ${CUBRAND_LIBRARY}
    ${CULIBOS_LIBRARY}
    "-Wl,-Bsymbolic -Wl,-Bsymbolic-functions"
    "pthread"
)

