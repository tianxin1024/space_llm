#include "layers/attention_layers/DecoderCrossAttentionLayer.h"

namespace space_llm {

const int WARP_SIZE = 32;
const bool ATTENTION_OPT = true;
const int ATTENTION_BLOCK_SIZE = 256;

template <typename T>
DecoderCrossAttentionLayer<T>::DecoderCrossAttentionLayer(size_t max_batch_size,
                                                          size_t head_num,
                                                          size_t size_per_head,
                                                          size_t d_model,
                                                          const float q_scaling,
                                                          cudaStream_t stream,
                                                          cublasMMWrapper *cublas_wrapper,
                                                          IAllocator *allocator,
                                                          bool is_free_buffer_after_forward) :
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num_ * size_per_head_),
    d_model_(d_model),
    q_scaling_(q_scaling) {
    QK_CHECK(size_per_head_ == 32 || size_per_head_ == 48 || size_per_head_ == 64 || size_per_head_ == 80
             || size_per_head_ == 96 || size_per_head_ == 112 || size_per_head_ == 128 || size_per_head_ == 144
             || size_per_head_ == 160 || size_per_head_ == 192 || size_per_head_ == 224 || size_per_head_ == 256);
}

template <typename T>
DecoderCrossAttentionLayer<T>::DecoderCrossAttentionLayer(size_t max_batch_size,
                                                          size_t head_num,
                                                          size_t size_per_head,
                                                          cudaStream_t stream,
                                                          cublasMMWrapper *cublas_wrapper,
                                                          IAllocator *allocator,
                                                          bool is_free_buffer_after_forward) :
    DecoderCrossAttentionLayer<T>(max_batch_size,
                                  head_num,
                                  size_per_head,
                                  head_num * size_per_head,
                                  1.0f,
                                  stream,
                                  cublas_wrapper,
                                  allocator,
                                  is_free_buffer_after_forward) {
}

template <typename T>
DecoderCrossAttentionLayer<T>::DecoderCrossAttentionLayer(size_t max_batch_size,
                                                          size_t head_num,
                                                          size_t size_per_head,
                                                          const float q_scaling,
                                                          cudaStream_t stream,
                                                          cublasMMWrapper *cublas_wrapper,
                                                          IAllocator *allocator,
                                                          bool is_free_buffer_after_forward) :
    DecoderCrossAttentionLayer<T>(max_batch_size,
                                  head_num,
                                  size_per_head,
                                  head_num * size_per_head,
                                  q_scaling,
                                  stream,
                                  cublas_wrapper,
                                  allocator,
                                  is_free_buffer_after_forward) {
}

template <typename T>
DecoderCrossAttentionLayer<T>::DecoderCrossAttentionLayer(DecoderCrossAttentionLayer<T> const &attention_layer) :
    DecoderCrossAttentionLayer(attention_layer.max_batch_size_,
                               attention_layer.head_num_,
                               attention_layer.size_per_head_,
                               attention_layer.d_model_,
                               attention_layer.q_scaling_,
                               attention_layer.stream_,
                               attention_layer.cublas_wrapper_,
                               attention_layer.allocator_,
                               attention_layer.is_free_buffer_after_forward_) {
}

template <typename T>
DecoderCrossAttentionLayer<T>::~DecoderCrossAttentionLayer() {
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template class DecoderCrossAttentionLayer<float>;
template class DecoderCrossAttentionLayer<half>;

} // namespace space_llm
