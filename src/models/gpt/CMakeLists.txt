cmake_minimum_required(VERSION 3.10)

add_library(GptDecoderLayerWeight STATIC GptDecoderLayerWeight.cc)
set_property(TARGET GptDecoderLayerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptDecoderLayerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptDecoderLayerWeight PUBLIC memory_utils cuda_utils logger)

add_library(GptWeight STATIC GptWeight.cc)
set_property(TARGET GptWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptWeight PUBLIC GptDecoderLayerWeight cuda_utils logger)

add_library(GptDecoder STATIC GptDecoder.cc)
set_property(TARGET GptDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptDecoder PUBLIC -lcudart layernorm_kernels TensorParallelDecoderSelfAttentionLayer  
                                                add_residual_kernels tensor cuda_utils logger)

add_library(GptContextDecoder STATIC GptContextDecoder.cc)
set_property(TARGET GptDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GptDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GptContextDecoder PUBLIC -lcudart TensorParallelReluFfnLayer TensorParallelGeluFfnLayer  
                                                TensorParallelGptContextAttentionLayer layernorm_kernels
                                                add_residual_kernels preprocess_kernels gpt_kernels tensor
                                                cuda_utils logger)

add_library(Gpt STATIC Gpt.cc)
set_property(TARGET Gpt PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Gpt PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Gpt PUBLIC -lcudart GptDecoder gpt_kernels GptContextDecoder
                      preprocess_kernels GptWeight DynamicDecodeLayer memory_utils decoding_kernels
                      cuda_utils logger)


