#pragma once

#include <cuda_runtime.h>

namespace space_llm {

void invokeGetPaddingOffsetAndCuSeqLens(size_t *h_pinned_token_num,
                                        size_t *h_token_num,
                                        int *tmp_mask_offset,
                                        int *cu_seqlens,
                                        const int *sequence_length,
                                        const int batch_size,
                                        const int max_seq_len,
                                        cudaStream_t stream);

inline void invokeGetPaddingOffset(size_t *h_pinned_token_num,
                                   size_t *h_token_num,
                                   int *tmp_mask_offset,
                                   const int *sequence_length,
                                   const int batch_size,
                                   const int max_seq_len,
                                   cudaStream_t stream) {
    invokeGetPaddingOffsetAndCuSeqLens(
        h_pinned_token_num, h_token_num, tmp_mask_offset, nullptr, sequence_length, batch_size, max_seq_len, stream);
}

void invokeGetTrtPaddingOffset(int *trt_mha_padding_offset,
                               const int *requence_length,
                               const int request_batch_size,
                               cudaStream_t stream);

void invokeGetTrtPaddingOffset(int *trt_mha_padding_offset,
                               const int *sequence_length,
                               const int request_batch_size,
                               const int request_seq_len,
                               cudaStream_t stream);

} // namespace space_llm
