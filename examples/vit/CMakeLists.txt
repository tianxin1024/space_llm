cmake_minimum_required(VERSION 3.10)

add_executable(demo_vit demo_vit.cpp)
set_property(TARGET demo_vit PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET demo_vit PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(demo_vit PUBLIC ViT cublasMMWrapper -lcublas -lcublasLt -lcudnn)
