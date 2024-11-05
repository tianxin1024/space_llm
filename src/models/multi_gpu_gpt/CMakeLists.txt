cmake_minimum_required(VERSION 3.10)

add_library(ParallelGptDecoderLayerWeight STATIC ParallelGptDecoderLayerWeight.cc)
set_property(TARGET ParallelGptDecoderLayerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelGptDecoderLayerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelGptDecoderLayerWeight PUBLIC memory_utils cuda_utils logger)

add_library(ParallelGptWeight STATIC ParallelGptWeight.cc)
set_property(TARGET ParallelGptWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelGptWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelGptWeight PUBLIC ParallelGptDecoderLayerWeight cuda_utils logger)

add_library(ParallelGptDecoder STATIC ParallelGptDecoder.cc)
set_property(TARGET ParallelGptDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelGptDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelGptDecoder PUBLIC -lcudart 
                                                layernorm_kernels TensorParallelDecoderSelfAttentionLayer  
                                                add_residual_kernels tensor cuda_utils logger)

add_library(ParallelGptContextDecoder STATIC ParallelGptContextDecoder.cc)
set_property(TARGET ParallelGptDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelGptDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelGptContextDecoder PUBLIC -lcudart TensorParallelReluFfnLayer 
                                                TensorParallelGptContextAttentionLayer layernorm_kernels
                                                add_residual_kernels preprocess_kernels gpt_kernels tensor
                                                cuda_utils logger)

add_library(ParallelGpt STATIC ParallelGpt.cc)
set_property(TARGET ParallelGpt PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelGpt PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelGpt PUBLIC -lcudart ParallelGptDecoder gpt_kernels ParallelGptContextDecoder
                      preprocess_kernels ParallelGptWeight memory_utils 
                      cuda_utils logger)

