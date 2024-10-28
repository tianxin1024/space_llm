#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace space_llm {

template <typename T>
void invokeAddBiasConcatClsTokenAddPosEmbed(const T *input,
                                            T *out,
                                            const T *bias,
                                            const T *cls_token,
                                            const T *pos_embed,
                                            const int m,
                                            const int n,
                                            const int s,
                                            cudaStream_t stream);

template <typename T>
void invokeAddBiasAddPosEmbed(
    T *out, const T *bias, const T *pos_embed, const int m, const int n, const int s, cudaStream_t stream);

} // namespace space_llm
