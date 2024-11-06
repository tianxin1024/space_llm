#pragma once

#include "kernels/gpt_kernels.h"

namespace space_llm {

template <typename T>
void invokeDecodingInitialize(bool *finished,
                              int *sequence_length,
                              int *word_ids,
                              T *cum_log_probs,
                              const int *sentence_ids,
                              const int batch_size,
                              const int beam_width,
                              const int max_input_length,
                              cudaStream_t stream);

// get token from all_ids at step, then lookup from the embedding table by the token
template <typename T>
void invokeEmbeddingLookupPosEncodingPadCount(T *from_tensor,
                                              const T *embedding_table,
                                              const T *position_encoding,
                                              const int *all_ids,
                                              pPromptTuningParam<T> prompt_param,
                                              const int local_token_num,
                                              const int hidden_units,
                                              const T scale,
                                              const int step,
                                              const int token_num,
                                              const int ite,
                                              const int seq_len,
                                              cudaStream_t stream);

template <typename T>
void invokeEmbeddingLookupPosEncodingPadCount(T *from_tensor,
                                              const T *embedding_table,
                                              const T *position_encoding,
                                              const int *all_ids,
                                              const int *padding_count,
                                              const int local_token_num,
                                              const int hidden_units,
                                              const T scale,
                                              const int step,
                                              const int token_num,
                                              const int ite,
                                              cudaStream_t stream) {
    invokeEmbeddingLookupPosEncodingPadCount(from_tensor,
                                             embedding_table,
                                             position_encoding,
                                             all_ids,
                                             padding_count,
                                             {(const T **)nullptr, 0, 0, false, nullptr},
                                             local_token_num,
                                             hidden_units,
                                             scale,
                                             step,
                                             token_num,
                                             ite,
                                             0,
                                             stream);
}

} // namespace space_llm
