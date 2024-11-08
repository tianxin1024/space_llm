cmake_minimum_required(VERSION 3.10)

add_subdirectory(attention_layers)
add_subdirectory(sampling_layers)

add_library(ffnLayer STATIC ffnLayer.cc)
set_property(TARGET ffnLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ffnLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ffnLayer PUBLIC -lcublas -lcudart cublasMMWrapper activation_kernels memory_utils)

add_library(DynamicDecodeLayer STATIC DynamicDecodeLayer.cc)
set_property(TARGET DynamicDecodeLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET DynamicDecodeLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(DynamicDecodeLayer PUBLIC -lcudart TopKSamplingLayer 
                        gpt_kernels tensor)

