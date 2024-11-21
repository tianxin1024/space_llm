#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace space_llm {

template <typename T>
void invokeBanBadWords(T *logits,
                       const int *output_ids_buf,
                       const int *parent_ids_buf,
                       int batch_size,
                       int local_batch_size,
                       int beam_width,
                       const int *bad_words,
                       bool share_words,
                       size_t bad_words_len,
                       int id_offset,
                       int vocab_size_padded,
                       size_t step,
                       cudaStream_t stream);

} // namespace space_llm
