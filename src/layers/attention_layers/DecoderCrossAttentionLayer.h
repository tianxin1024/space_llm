#pragma once

#include "layers/attention_layers/BaseAttentionLayer.h"

namespace space_llm {

template <typename T>
class DecoderCrossAttentionLayer : public BaseAttentionLayer<T> {
private:
    // metadate
    const size_t head_num_;
    const size_t size_per_head_;
    const size_t d_model_;
    bool is_batch_major_cache_ = true;

    // calculated params
    const size_t hidden_units_;
    const float q_scaling_;

    // buffer handling
    size_t max_batch_size_ = 0;
    size_t max_mem_seq_len_ = 0;

    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size, size_t max_mem_seq_len);
    void freeBuffer() override;

protected:
    using BaseAttentionLayer<T>::stream_;
    using BaseAttentionLayer<T>::is_free_buffer_after_forward_;
    using BaseAttentionLayer<T>::is_allocate_buffer_;
    using BaseAttentionLayer<T>::cublas_wrapper_;
    using BaseAttentionLayer<T>::allocator_;

    T *q_buf_ = nullptr;
    T *context_buf_ = nullptr;
    T *mem_cache_buf_ = nullptr;

public:
    DecoderCrossAttentionLayer(size_t max_batch_size,
                               size_t head_num,
                               size_t size_per_head,
                               cudaStream_t stream,
                               cublasMMWrapper *cublas_wrapper,
                               IAllocator *allocator,
                               bool is_free_buffer_after_forward);

    DecoderCrossAttentionLayer(size_t max_batch_size,
                               size_t head_num,
                               size_t size_per_head,
                               const float q_scaling,
                               cudaStream_t stream,
                               cublasMMWrapper *cublas_wrapper,
                               IAllocator *allocator,
                               bool is_free_buffer_after_forward);

    DecoderCrossAttentionLayer(size_t max_batch_size,
                               size_t head_num,
                               size_t size_per_head,
                               size_t d_model,
                               const float q_scaling,
                               cudaStream_t stream,
                               cublasMMWrapper *cublas_wrapper,
                               IAllocator *allocator,
                               bool is_free_buffer_after_forward);

    DecoderCrossAttentionLayer(DecoderCrossAttentionLayer<T> const &attention_layer);

    ~DecoderCrossAttentionLayer();

    void forward(TensorMap *output_tensors, TensorMap *input_tensors, const AttentionWeight<T> *attention_weights) override;
};

} // namespace space_llm
