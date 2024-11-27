#pragma once

#include "utils/logger.h"
#include <curand_kernel.h>

namespace space_llm {

template <typename T>
void invokeTopKSampling(void *workspace,
                        size_t &workspace_size,
                        const T *log_probs,
                        int *ids,
                        int *sequence_lengths,
                        bool *finished_buf,
                        float *cum_log_probs,
                        float *output_log_probs,
                        curandState_t *curandstate,
                        const int top_k,
                        const float top_p,
                        const int vocab_size_padded,
                        const int *end_ids,
                        cudaStream_t stream,
                        const int batch_size,
                        const bool *skip_decode);

template <typename T>
void invokeBatchTopKSampling(void *workspace,
                             size_t &workspace_size,
                             const T *log_probs,
                             int *ids,
                             int *sequence_length,
                             bool *finished,
                             float *cum_log_probs,
                             float *output_log_probs,
                             curandState_t *curandstate,
                             const int max_top_k,
                             const int *top_ks,
                             const float top_p,
                             const float *top_ps,
                             const int vocab_size_padded,
                             const int *end_ids,
                             cudaStream_t stream,
                             const int batch_size,
                             const bool *skip_decode);

void invokeCurandInitialize(curandState_t *state,
                            const size_t batch_size,
                            unsigned long long random_seed,
                            cudaStream_t stream);

template <typename T>
void invokeAddBiasEndMask(T *logits,
                          const T *bias,
                          const int *end_ids,
                          const bool *finished,
                          const int batch_size,
                          const int vocab_size,
                          const int vocab_size_padded,
                          cudaStream_t stream);

} // namespace space_llm
