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
