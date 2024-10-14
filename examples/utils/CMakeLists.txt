cmake_minimum_required(VERSION 3.10)

add_executable(demo_tensor demo_tensor.cu)
set_property(TARGET demo_tensor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET demo_tensor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(demo_tensor PUBLIC tensor)

