#pragma once

#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "int8_utils.cuh"
#include "utils/tensor.h"

namespace space_llm {

void invokeTransposeCOL32(int8_t *dst,
                          const int8_t *src,
                          const int batch_size,
                          const int seq_len,
                          const int head_num,
                          const int size_per_head,
                          const float *bmm2_deQFactor,
                          const float *out_scale_ptr,
                          cudaStream_t stream);

void invokeTransposeCOL32(int8_t *dst,
                          const int *src,
                          const int batch_size,
                          const int seq_len,
                          const int head_num,
                          const int size_per_head,
                          const float *v_buf_addBias_deQFactor,
                          const float *qk_afterSM_deQFactor,
                          const float *out_scale_ptr,
                          cudaStream_t stream);

void invokeTransposeCOL32RebuildPadding(int8_t *dst,
                                        const int *src,
                                        const int *sequence_id_map,
                                        const int valid_word_num,
                                        const int batch_size,
                                        const int seq_len,
                                        const int head_num,
                                        const int size_per_head,
                                        const float *v_buf_addBias_deQFactor,
                                        const float *qk_afterSM_deQFactor,
                                        const float *out_scale_ptr,
                                        cudaStream_t stream);

void invokeTransposeCOL32RebuildPadding(int8_t *dst,
                                        const int8_t *src,
                                        const int *sequence_id_map,
                                        const int valid_word_num,
                                        const int batch_size,
                                        const int seq_len,
                                        const int head_num,
                                        const int size_per_head,
                                        const float *bmm2_deQFactor,
                                        const float *out_scale_ptr,
                                        cudaStream_t stream);

void invokeTransposeCOL32ToRow(int8_t *dst,
                               const int8_t *src,
                               const int batch_size,
                               const int seq_len,
                               const int head_num,
                               const int size_per_head,
                               const float *bmm2_deQFactor,
                               const float *out_scale_ptr,
                               cudaStream_t stream);

void invokeTransposeCOL32ToRowRebuildPadding(int8_t *dst,
                                             const int8_t *src,
                                             const int *sequence_id_map,
                                             const int valid_word_num,
                                             const int batch_size,
                                             const int seq_len,
                                             const int head_num,
                                             const int size_per_head,
                                             const float *bmm2_deQFactor,
                                             const float *out_scale_ptr,
                                             cudaStream_t stream);

void invokeTransposeInt8Tensor(const Tensor &x_t, const Tensor &x, cudaStream_t stream = 0);

} // namespace space_llm
