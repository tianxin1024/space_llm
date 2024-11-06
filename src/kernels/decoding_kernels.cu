#include "kernels/decoding_kernels.h"
#include "kernels/reduce_kernel_utils.cuh"

namespace space_llm {

template <typename T>
__global__ void decodingInitialize(bool *finished,
                                   int *sequence_length,
                                   int *word_ids,
                                   T *cum_log_probs,
                                   const int *sentence_ids,
                                   const int batch_size,
                                   const int beam_width,
                                   const int max_input_length) {
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? (T)HALF_FLT_MAX : (T)1e20f;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * beam_width; index += blockDim.x * gridDim.x) {
        finished[index] = false;
        sequence_length[index] = max_input_length;
        if (word_ids != nullptr) {
            word_ids[index] = sentence_ids[index / beam_width];
        }
        cum_log_probs[index] = (index % beam_width == 0) ? (T)0.0f : (T)-MAX_T_VAL;
    }
}

template <typename T>
void invokeDecodingInitialize(bool *finished,
                              int *sequence_length,
                              int *word_ids,
                              T *cum_log_probs,
                              const int *sentence_ids,
                              const int batch_size,
                              const int beam_width,
                              const int max_input_length,
                              cudaStream_t stream) {
    dim3 grid((int)ceil(batch_size * beam_width * 1.0 / 256));
    dim3 block(256);

    decodingInitialize<T><<<grid, block, 0, stream>>>(
        finished, sequence_length, word_ids, cum_log_probs, sentence_ids, batch_size, beam_width, max_input_length);
}

template void invokeDecodingInitialize(bool *finished,
                                       int *sequence_length,
                                       int *word_ids,
                                       float *cum_log_probs,
                                       const int *sentence_ids,
                                       const int batch_size,
                                       const int beam_width,
                                       const int max_input_length,
                                       cudaStream_t stream);

template void invokeDecodingInitialize(bool *finished,
                                       int *sequence_length,
                                       int *word_ids,
                                       half *cum_log_probs,
                                       const int *sentence_ids,
                                       const int batch_size,
                                       const int beam_width,
                                       const int max_input_length,
                                       cudaStream_t stream);

// PROMPT_SRC: 0 --> no prompts, 1 --> from loaded prompts, 2 --> from request prompts
template <typename T>
__global__ void embeddingLookupPosEncoding(T *from_tensor,
                                           const T *embedding_table,
                                           const T *position_encoding,
                                           const int *all_ids,
                                           const int *padding_count,
                                           const int *input_lengths,
                                           const int local_token_num,
                                           const int64_t hidden_units,
                                           const int step,
                                           const int max_input_length,
                                           const int token_num,
                                           const int ite,
                                           const T scale) {
    // 1. lookup from embedding table
    // 2. multiply scale
    // 3. add the position encoding
    const int id_offset = step * token_num + ite * local_token_num;

    const bool use_padding_count = padding_count != nullptr;
    const bool use_input_len = input_lengths != nullptr;

    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < local_token_num * hidden_units; index += blockDim.x * gridDim.x) {
        const int row_index = index / hidden_units;
        const int col_index = index % hidden_units;
        int step_offset = step;
        if (use_padding_count) {
            step_offset -= padding_count[row_index];
        } else if (use_input_len) {
            step_offset -= max_input_length - input_lengths[row_index];
        }
        step_offset *= hidden_units;

        T val = embedding_table[all_ids[id_offset + row_index] * hidden_units + col_index] * scale;
        val = val + position_encoding[step_offset + col_index];

        from_tensor[index] = val;
    }
}

// No absolute position embedding
// PROMPT_SRC: 0 --> no prompts, 1 --> from loaded prompts, 2 --> from request prompts
template <typename T, int PROMPT_SRC>
__global__ void embeddingLookup(T *from_tensor,
                                const T *embedding_table,
                                const int *all_ids,
                                pPromptTuningParam<T> prompt_param,
                                const int local_token_num,
                                const int64_t hidden_units,
                                const int step,
                                const int token_num,
                                const int ite,
                                const int seq_len,
                                const T scale) {
    // 1. lookup from embedding table
    // 2. multiply scale
    const int id_offset = step * token_num + ite * local_token_num;

    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < local_token_num * hidden_units; index += blockDim.x * gridDim.x) {
        const int word_index = index / hidden_units;
        const int word_index_row = word_index / seq_len; // batch_id
        const int col_index = index % hidden_units;
        const int input_id = all_ids == nullptr ? word_index : all_ids[id_offset + word_index];
        const int prompt_id = input_id - prompt_param.p_prompt_tuning_id_start;
        T embedding = (T)0.0f;

        if (PROMPT_SRC > 0 && prompt_id >= 0) {
            if (PROMPT_SRC == 1) {
                // from loaded prompt embedding tables
                embedding =
                    prompt_param.p_prompt_tuning_batch_weights[word_index_row][prompt_id * hidden_units + col_index];
            } else {
                // from request prompt embedding
                embedding =
                    prompt_param
                        .request_prompt_embedding[word_index_row * prompt_param.request_prompt_max_length * hidden_units
                                                  + prompt_id * hidden_units + col_index];
            }
        } else {
            embedding = embedding_table[input_id * hidden_units + col_index];
        }
        from_tensor[index] = embedding * scale;
    }
}

#define EMBEDDING_LOOKUP(PROMPT_SRC)                                            \
    embeddingLookup<T, PROMPT_SRC><<<grid, block, 0, stream>>>(from_tensor,     \
                                                               embedding_table, \
                                                               all_ids,         \
                                                               prompt_param,    \
                                                               local_token_num, \
                                                               hidden_units,    \
                                                               step,            \
                                                               token_num,       \
                                                               ite,             \
                                                               seq_len,         \
                                                               scale);

/* Adapter function for invokeEmbeddingLookupPosEncoding{PadCount,InputLen} */
template <typename T>
void invokeEmbeddingLookupPosEncoding(T *from_tensor,
                                      const T *embedding_table,
                                      const T *position_encoding,
                                      const int *all_ids,
                                      const int *padding_count,
                                      const int *input_lengths,
                                      pPromptTuningParam<T> prompt_param,
                                      const int local_token_num,
                                      const int hidden_units,
                                      const T scale,
                                      const int step,
                                      const int max_input_length,
                                      const int token_num,
                                      const int ite,
                                      const int seq_len,
                                      cudaStream_t stream) {
    dim3 grid(std::min(local_token_num, 65536));
    dim3 block(std::min(hidden_units, 1024));
    if (position_encoding != nullptr) {
        QK_CHECK_WITH_INFO(prompt_param.use_request_p_prompt_embedding == false
                               && prompt_param.p_prompt_tuning_batch_weigths == nullptr,
                           fmtstr("embeddingLookupPosEncoding still not support prompt tuning"));
        embeddingLookupPosEncoding<T><<<grid, block, 0, stream>>>(from_tensor,
                                                                  embedding_table,
                                                                  position_encoding,
                                                                  all_ids,
                                                                  padding_count,
                                                                  input_lengths,
                                                                  local_token_num,
                                                                  hidden_units,
                                                                  step,
                                                                  max_input_length,
                                                                  token_num,
                                                                  ite,
                                                                  scale);
    } else {
        if (prompt_param.use_request_p_prompt_embedding) {
            EMBEDDING_LOOKUP(2);
        } else if (prompt_param.p_prompt_tuning_batch_weights != nullptr) {
            EMBEDDING_LOOKUP(1);
        } else {
            EMBEDDING_LOOKUP(0);
        }
    }
}

#undef EMBEDDING_LOOKUP

} // namespace space_llm
