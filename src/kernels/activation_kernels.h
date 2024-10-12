#pragma once

#include <cuda_runtime.h>

namespace space_llm {

// clang-format off
template<typename T> struct GeluActivation;
template<typename T> struct ReluActivation;
template<typename T> struct SiluActivation;
template<typename T> struct IdentityActivation;
// clang-format on

template <template <typename T> class Activation, typename T, typename BT>
void invokeGenericActivation(T *out,
                             const BT *bias,
                             const T *gated_weights,
                             const BT *gated_bias,
                             const int *ia3_tasks,
                             const T *ia3_weights,
                             const int m,
                             const int n,
                             const int int8_mode,
                             const float *activation_in,
                             const float *activation_out,
                             const int *padding_offset,
                             const int seq_len,
                             cudaStream_t stream);

template <typename T>
void invokeAddBiasGeluV2(T *out,
                         const T *bias,
                         const int *ia3_tasks,
                         const T *ia3_weights,
                         const int *padding_offset,
                         const int seq_len,
                         const int m,
                         const int n,
                         cudaStream_t stream);

template <typename T>
void invokeAddBiasGeluV2(
    T *out, const T *bias, const int *ia3_tasks, const T *ia3_weights, const int m, const int n, cudaStream_t stream) {
    invokeAddBiasGeluV2(out, bias, ia3_tasks, ia3_weights, nullptr, 0, m, n, stream);
}

} // namespace space_llm
