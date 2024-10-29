#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <assert.h>

#include "utils/cuda_utils.h"

namespace space_llm {

template <typename T>
struct LayerNormWeight {
    const T *gamma = nullptr;
    const T *beta = nullptr;
};

template <typename T>
void invokeGeneralAddBiasResidualPreLayerNorm(T *output,
                                              T *norm_output,
                                              const T *input,
                                              const T *residual1,
                                              const T *residual2,
                                              const T *gamma,
                                              const T *beta,
                                              const T *bias,
                                              const float layernorm_eps,
                                              int m,
                                              int n,
                                              const float *scale_inter,
                                              const float *scale_out,
                                              float *scale,
                                              float *dynamic_scale,
                                              const int int8_mode,
                                              cudaStream_t stream,
                                              int opt_version = 2);

template <typename T>
void invokeGeneralAddBiasResidualPreLayerNorm(T *output,
                                              T *norm_output,
                                              const T *input,
                                              const T *residual1,
                                              const T *gamma,
                                              const T *beta,
                                              const T *bias,
                                              const float layernorm_eps,
                                              int m,
                                              int n,
                                              const float *scale_inter,
                                              const float *scale_out,
                                              float *scale,
                                              float *dynamic_scale,
                                              const int int8_mode,
                                              cudaStream_t stream,
                                              int opt_version = 2) {
    invokeGeneralAddBiasResidualPreLayerNorm(output,
                                             norm_output,
                                             input,
                                             residual1,
                                             (const T *)nullptr,
                                             gamma,
                                             beta,
                                             bias,
                                             layernorm_eps,
                                             m,
                                             n,
                                             scale_inter,
                                             scale_out,
                                             scale,
                                             dynamic_scale,
                                             int8_mode,
                                             stream,
                                             opt_version);
}

template <typename T>
void invokeGeneralLayerNorm(T *out,
                            const T *input,
                            const T *gamma,
                            const T *beta,
                            const float layernorm_eps,
                            const int m,
                            const int n,
                            float *scale,
                            float *dynamic_scale,
                            const int int8_mode,
                            cudaStream_t stream,
                            int opt_version = 2);

template <typename T>
void invokeGeneralLayerNorm(T *out,
                            const T *input,
                            const T *gamma,
                            const T *beta,
                            const float layernorm_eps,
                            const int m,
                            const int n,
                            float *scale,
                            const int int8_mode,
                            cudaStream_t stream,
                            int opt_version = 2) {
    invokeGeneralLayerNorm(
        out, input, gamma, beta, layernorm_eps, m, n, scale, (float *)nullptr, int8_mode, stream, opt_version = 2);
}

} // namespace space_llm
