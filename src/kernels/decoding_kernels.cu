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

} // namespace space_llm
