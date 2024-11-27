#pragma once

#include <cuda_fp16.h>

#include "kernels/penalty_types.h"
#include "utils/cuda_utils.h"

namespace space_llm {

template <typename T>
void invokeApplyRepetitionPenalty(T *logits,
                                  const float penalty,
                                  const int *start_ids,
                                  int *output_ids,
                                  const int batch_size,
                                  const int local_batch_size,
                                  const int vocab_size,
                                  const int vocab_size_padd,
                                  const int *input_lengths,
                                  const int max_input_len,
                                  const int step,
                                  const RepetitionPenaltyType penalty_type,
                                  cudaStream_t stream);

template <typename T>
void invokeBatchApplyRepetitionPenalty(T *logits,
                                       const float *penalties,
                                       const int *output_ids,
                                       const int batch_size,
                                       const int local_batch_size,
                                       const int vocab_size,
                                       const int *input_lengths,
                                       const int max_input_length,
                                       const int step,
                                       const RepetitionPenaltyType penalty_type,
                                       cudaStream_t stream);

template <typename T>
void invokeApplyTemperaturePenalty(T *logits,
                                   const T *bias,
                                   const float temperature,
                                   const int batch_size,
                                   const int vocab_size,
                                   const int vocab_size_padd,
                                   cudaStream_t stream);

template <typename T>
void invokeBatchApplyTemperaturePenalty(T *logits,
                                        const T *bias,
                                        const float *temperatures,
                                        const int batch_size,
                                        const int vocab_size,
                                        const int vocab_size_padd,
                                        cudaStream_t stream);

template <typename T>
void invokeMinLengthPenalty(T *logits,
                            const int *min_lengths,
                            const int *end_ids,
                            const int *sequnece_lengths,
                            const int max_input_length,
                            const int batch_size,
                            const int vocab_size_padded,
                            cudaStream_t stream);

} // namespace space_llm
