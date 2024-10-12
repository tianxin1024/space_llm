cmake_minimum_required(VERSION 3.10)

add_executable(test_tensor test_tensor.cpp)
set_property(TARGET test_tensor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET test_tensor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(test_tensor PUBLIC tensor)

