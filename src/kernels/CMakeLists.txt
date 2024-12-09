
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

add_library(add_residual_kernels STATIC add_residual_kernels.cu)
set_property(TARGET add_residual_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET add_residual_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)


set(decoder_masked_multihead_attention_files
    decoder_masked_multihead_attention.cu
)
file(GLOB decoder_masked_multihead_attention_files ${decoder_masked_multihead_attention_files} ./decoder_masked_multihead_attention/*.cu)
add_library(decoder_masked_multihead_attention STATIC ${decoder_masked_multihead_attention_files})
set_property(TARGET decoder_masked_multihead_attention PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET decoder_masked_multihead_attention PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

# add_library(decoder_masked_multihead_attention STATIC decoder_masked_multihead_attention.cu)
# set_property(TARGET decoder_masked_multihead_attention PROPERTY POSITION_INDEPENDENT_CODE  ON)
# set_property(TARGET decoder_masked_multihead_attention PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(decoding_kernels STATIC decoding_kernels.cu)
set_property(TARGET decoding_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET decoding_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(sampling_topk_kernels STATIC sampling_topk_kernels.cu)
set_property(TARGET sampling_topk_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET sampling_topk_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(sampling_topp_kernels STATIC sampling_topp_kernels.cu)
set_property(TARGET sampling_topp_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET sampling_topp_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(ban_bad_words STATIC ban_bad_words.cu)
set_property(TARGET ban_bad_words PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ban_bad_words PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(sampling_penalty_kernels STATIC sampling_penalty_kernels.cu)
set_property(TARGET sampling_penalty_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET sampling_penalty_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(stop_criteria_kernels STATIC stop_criteria_kernels.cu)
set_property(TARGET stop_criteria_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET stop_criteria_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
