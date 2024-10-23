#pragma once

#include "layers/attention_layers/BaseAttentionLayer.h"

namespace space_llm {

template <typename T>
class UnfusedAttentionLayer : public BaseAttentionLayer<T> {
private:
    // metadata
    size_t head_num_;
    size_t size_per_head_;
    size_t hidden_units_; // d_model
    size_t d_model_;
    bool sparse_;
    float q_scaling_;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len);

protected:
    using BaseAttentionLayer<T>::stream_;
    using BaseAttentionLayer<T>::is_free_buffer_after_forward_;
    using BaseAttentionLayer<T>::is_allocate_buffer_;
    using BaseAttentionLayer<T>::cublas_wrapper_;
    using BaseAttentionLayer<T>::allocator_;

    T *q_buf_ = nullptr;
    T *k_buf_ = nullptr;
    T *v_buf_ = nullptr;
    T *q_buf_2_ = nullptr;
    T *k_buf_2_ = nullptr;
    T *v_buf_2_ = nullptr;
    T *qk_buf_ = nullptr;
    T *qkv_buf_ = nullptr;
    T *qkv_buf_2_ = nullptr;

    T **batch_qkv_kernel_ptr_ = nullptr;
    T **batch_qkv_input_ptr_ = nullptr;
    T **batch_qkv_buf_ptr_ = nullptr;

public:
    UnfusedAttentionLayer(size_t max_batch_size,
                          size_t max_seq_len,
                          size_t head_num,
                          size_t size_per_head,
                          float q_scaling,
                          cudaStream_t stream,
                          cublasMMWrapper *cublas_wrapper,
                          IAllocator *allocator,
                          bool is_free_buffer_after_forward,
                          bool sparse = false);

    UnfusedAttentionLayer(size_t max_batch_size,
                          size_t max_seq_len,
                          size_t head_num,
                          size_t size_per_head,
                          size_t d_model,
                          float q_scaling,
                          cudaStream_t stream,
                          cublasMMWrapper *cublas_wrapper,
                          IAllocator *allocator,
                          bool is_free_buffer_after_forward,
                          bool sparse = false);

    UnfusedAttentionLayer(UnfusedAttentionLayer<T> const &attention_layer);

    ~UnfusedAttentionLayer();

    void forward(TensorMap *output_tensors, TensorMap *input_tensors, const AttentionWeight<T> *attention_weights) override;
};

} // namespace space_llm
