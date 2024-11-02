#pragma once

#include "layers/attention_layers/GptContextAttentionLayer.h"

namespace space_llm {

template <typename T>
class TensorParallelGptContextAttentionLayer : public GptContextAttentionLayer<T> {
public:
    TensorParallelGptContextAttentionLayer(size_t max_batch_size,
                                           size_t max_seq_len,
                                           size_t head_num,
                                           size_t size_per_head,
                                           cudaStream_t stream,
                                           cublasMMWrapper *cublas_wrapper,
                                           IAllocator *allocator,
                                           bool do_all_reduce,
                                           bool is_free_buffer_after_forward,
                                           bool is_qk_buf_float,
                                           bool sparse = false,
                                           int int8_mode = 0);

    TensorParallelGptContextAttentionLayer(size_t max_batch_size,
                                           size_t max_seq_len,
                                           size_t head_num,
                                           size_t size_per_head,
                                           size_t rotary_embedding_dim,
                                           bool neox_rotary_style,
                                           cudaStream_t stream,
                                           cublasMMWrapper *cublas_wrapper,
                                           IAllocator *allocator,
                                           bool do_all_reduce,
                                           bool is_free_buffer_after_forward,
                                           bool is_qk_buf_float,
                                           bool sparse = false,
                                           int int8_mode = 0);

    TensorParallelGptContextAttentionLayer(TensorParallelGptContextAttentionLayer<T> const &attention_layer);

    void forward(TensorMap *output_tensors, TensorMap *input_tensors, const AttentionWeight<T> *attention_weights) override;
};

} // namespace space_llm
