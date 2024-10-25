cmake_minimum_required(VERSION 3.10)

add_library(ViT STATIC ViT.cc)
set_property(TARGET ViT PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ViT PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ViT PUBLIC -lcublas -lcudart cublasMMWrapper memory_utils
    UnfusedAttentionLayer ffnLayer layernorm_kernels preprocess_kernels tensor cuda_utils logger)
