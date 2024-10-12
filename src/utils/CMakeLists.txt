cmake_minimum_required(VERSION 3.10)

add_library(logger STATIC logger.cc)
set_property(TARGET logger PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET logger PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(logger PUBLIC -lcudart)

add_library(cuda_utils STATIC cuda_utils.cc)
set_property(TARGET cuda_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cuda_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(cuda_utils PUBLIC -lcudart)

add_library(tensor STATIC tensor.cc)
set_property(TARGET tensor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET tensor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(tensor PUBLIC cuda_utils logger)

add_library(memory_utils STATIC memory_utils.cu)
set_property(TARGET memory_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET memory_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(memory_utils PUBLIC cuda_utils logger tensor)
