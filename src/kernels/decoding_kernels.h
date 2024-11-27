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
                                              const int *padding_count,
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

void invokeGatherTree(int *beams,
                      int *max_sequence_lengths,
                      const int max_time,
                      const int batch_size,
                      const int beam_width,
                      const int *step_ids,
                      const int *parent_ids,
                      const int *end_tokens,
                      cudaStream_t stream);

void invokeGatherTree(int *beams,
                      int *max_sequence_lengths,
                      const int max_time,
                      const int batch_size,
                      const int beam_width,
                      const int *step_ids,
                      const int *parent_ids,
                      const int *end_tokens,
                      const int max_input_length,
                      cudaStream_t stream);

struct gatherTreeParam {
    int *beams = nullptr;
    int *max_sequence_lengths = nullptr;
    int max_sequence_length_final_step = 0;
    const int *input_lengths = nullptr;
    // response input lengths (used to slice the ids during postprocessing)
    int *response_input_lengths = nullptr;
    int max_time = 0;
    int batch_size = 0;
    int beam_width = 0;
    const int *step_ids = nullptr;
    const int *parent_ids = nullptr;
    const int *end_tokens = nullptr;
    int max_input_length = 0;
    const int *prefix_soft_prompt_lengths = nullptr;
    // p_prompt_tuning prompt leangths, used to remove prompts during post-processing
    const int *p_prompt_tuning_prompt_lengths = nullptr;
    int max_input_without_prompt_length = 0;
    // prefix soft prompt
    int max_prefix_soft_prompt_length = 0;
    int *output_ids = nullptr;
    cudaStream_t stream;
};

void invokeGatherTree(gatherTreeParam param);

} // namespace space_llm
