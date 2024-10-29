#pragma once

#include "utils/cuda_utils.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace space_llm {

template <typename T>
void invokeAddBiasResidual(T *output,
                           const T *input,
                           const T *residual1,
                           const T *residual2,
                           const T *bias,
                           const float *scale_inter,
                           const float *scale_out,
                           const int m,
                           const int n,
                           cudaStream_t stream);

template <typename T>
void invokeAddBiasResidual(
    T *output, const T *residual1, const T *residual2, const T *bias, const int m, const int n, cudaStream_t stream);

template <typename T>
void invokeAddBiasResidual(T *output, const T *residual1, const T *bias, const int m, const int n, cudaStream_t stream) {
    invokeAddBiasResidual(output, residual1, (const T *)nullptr, bias, m, n, stream);
}

} // namespace space_llm
