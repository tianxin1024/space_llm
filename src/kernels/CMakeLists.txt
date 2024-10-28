
add_library(activation_kernels STATIC activation_kernels.cu)
set_property(TARGET activation_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET activation_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(transpose_int8_kernels STATIC transpose_int8_kernels.cu)
set_property(TARGET transpose_int8_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET transpose_int8_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(unfused_attention_kernels STATIC unfused_attention_kernels.cu)
set_property(TARGET unfused_attention_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET unfused_attention_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(gpt_kernels STATIC gpt_kernels.cu)
set_property(TARGET gpt_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET gpt_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(layernorm_kernels STATIC layernorm_kernels.cu)
set_property(TARGET layernorm_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET layernorm_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(preprocess_kernels STATIC preprocess_kernels.cu)
set_property(TARGET preprocess_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET preprocess_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(vit_kernels STATIC vit_kernels.cu)
set_property(TARGET vit_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET vit_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

