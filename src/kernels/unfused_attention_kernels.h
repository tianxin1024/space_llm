#pragma once

#include "utils/tensor.h"
#include <cuda_runtime.h>

namespace space_llm {

template <typename T>
void invokeAddQKVBiasIA3Transpose(T *q_buf,
                                  T *k_buf,
                                  T *v_buf,
                                  T *Q,
                                  const T *bias_Q,
                                  T *K,
                                  const T *bias_K,
                                  T *V,
                                  const T *bias_V,
                                  const int batch_size,
                                  const int seq_len,
                                  const int head_num,
                                  const int size_per_head,
                                  const int *ia3_tasks,
                                  const T *ia3_key_weights,
                                  const T *ia3_value_weights,
                                  cudaStream_t stream);

template <typename T>
void invokeAddQKVBiasIA3RebuildPadding(T *Q,
                                       const T *bias_Q,
                                       T *K,
                                       const T *bias_K,
                                       T *V,
                                       const T *bias_V,
                                       T *q_buf,
                                       T *k_buf,
                                       T *v_buf,
                                       const int batch_size,
                                       const int seq_len,
                                       const int head_num,
                                       const int size_per_head,
                                       const int valid_word_num,
                                       const int *mask_offset,
                                       const int *ia3_tasks,
                                       const T *ia3_key_weights,
                                       const T *ia3_value_weights,
                                       cudaStream_t stream);

template <typename T>
void invokeAddRelativeAttentionBias(T *qk_buf,
                                    const T *relative_attention_bias,
                                    const int batch_size,
                                    const int head_num,
                                    const int seq_len,
                                    cudaStream_t stream);

template <typename T, typename T_IN>
struct MaskedSoftmaxParam {
    // Common parameters.
    T *attention_score = nullptr;      // (batch_size, head_num, q_length, k_length)
    const T_IN *qk = nullptr;          // (batch_size, head_num, q_length, k_length)
    const T *attention_mask = nullptr; // (batch_size, q_length, k_length)
    int batch_size = 0;
    int q_length = 0;
    int k_length = 0;
    int num_heads = 0;
    T qk_scale = T(0.0f);

    // Optional parameters that depend on the type of attention.
    // The slopes of the linear position bias of ALiBi.
    const T *linear_bias_slopes = nullptr; // (head_num,), optional
};

template <typename T, typename T_IN>
void invokeMaskedSoftmax(MaskedSoftmaxParam<T, T_IN> &param, cudaStream_t stream);

template <typename T>
void invokeTransposeAttentions(Tensor &attentions_out, const Tensor &attentions_in, cudaStream_t stream = 0);

template <typename T>
void invokeTransposeQKV(T *dst,
                        T *src,
                        const int batch_size,
                        const int seq_len,
                        const int head_num,
                        const int size_per_head,
                        const float *scale,
                        const int int8_mode,
                        cudaStream_t stream);

template <typename T>
void invokeTransposeAttentionOutRemovePadding(T *src,
                                              T *dst,
                                              const int valid_word_num,
                                              const int batch_size,
                                              const int seq_len,
                                              const int head_num,
                                              const int size_per_head,
                                              const int *mask_offset,
                                              const float *scale,
                                              const int int8_mode,
                                              cudaStream_t stream);

} // namespace space_llm
