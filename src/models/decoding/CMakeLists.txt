cmake_minimum_required(VERSION 3.10)

add_library(Decoder STATIC Decoder.cc)
set_property(TARGET Decoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Decoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Decoder PUBLIC -lcudart cublasMMWrapper DecoderSelfAttentionLayer memory_utils
                    DecoderCrossAttentionLayer ffnLayer layernorm_kernels add_residual_kernels tensor cuda_utils logger)

add_library(Decoding STATIC Decoding.cc)
set_property(TARGET Decoding PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Decoding PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Decoding PUBLIC -lcublas -lcudart -lcurand Decoder memory_utils 
                        BeamSearchLayer DynamicDecodeLayer tensor cuda_utils logger)

