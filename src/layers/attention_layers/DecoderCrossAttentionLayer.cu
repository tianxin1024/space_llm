#include "layers/attention_layers/DecoderCrossAttentionLayer.h"

namespace space_llm {

const int WARP_SIZE = 32;
const bool ATTENTION_OPT = true;
const int ATTENTION_BLOCK_SIZE = 256;

template <typename T>
void DecoderCrossAttentionLayer<T>::allocateBuffer() {
    QK_CHECK(false);
    if (is_allocate_buffer_ == false) {
        q_buf_ = reinterpret_cast<T *>(allocator_->reMalloc(q_buf_, sizeof(T) * max_batch_size_ * hidden_units_, false));
        context_buf_ = reinterpret_cast<T *>(
            allocator_->reMalloc(context_buf_, sizeof(T) * max_batch_size_ * hidden_units_, false));

        if (is_batch_major_cache_) {
            mem_cache_buf_ = reinterpret_cast<T *>(allocator_->reMalloc(
                mem_cache_buf_, sizeof(T) * max_batch_size_ * max_mem_seq_len_ * hidden_units_, false));
        }

        is_allocate_buffer_ = true;
    }
}

template <typename T>
void DecoderCrossAttentionLayer<T>::allocateBuffer(size_t batch_size, size_t max_mem_seq_len) {
    q_buf_ = reinterpret_cast<T *>(allocator_->reMalloc(q_buf_, sizeof(T) * batch_size * hidden_units_, false));
    context_buf_ = reinterpret_cast<T *>(
        allocator_->reMalloc(context_buf_, sizeof(T) * batch_size * hidden_units_, false));

    if (is_batch_major_cache_) {
        mem_cache_buf_ = reinterpret_cast<T *>(allocator_->reMalloc(
            mem_cache_buf_, sizeof(T) * batch_size * max_mem_seq_len * hidden_units_, false));
    }

    is_allocate_buffer_ = true;
}

template <typename T>
void DecoderCrossAttentionLayer<T>::freeBuffer() {
    if (is_allocate_buffer_) {
        allocator_->free((void **)(&q_buf_));
        allocator_->free((void **)(&context_buf_));
        if (is_batch_major_cache_) {
            allocator_->free((void **)(&mem_cache_buf_));
        }
        is_allocate_buffer_ = false;
    }
}

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

template <typename T>
void DecoderCrossAttentionLayer<T>::forward(TensorMap *output_tensors,
                                            TensorMap *input_tensors,
                                            const AttentionWeight<T> *attention_weights) {
    // input tensors:
    //      attention_input [batch_size, d_model],
    //      encoder_output [batch_size, mem_max_seq_len, memory_d_model],
    //      encoder_sequence_length [batch_size],
    //      step [1] on cpu
    //      finished [batch_size] (optional)
    //      ia3_tasks [batch_size] (optional)

    // output tensors:
    //      decoder_layer_output [batch_size, d_model],
    //      key_mem_cache [batch_size, head_num, size_per_head // x, mem_max_seq_len, x], where x = 16 / sizeof(T)
    //      value_mem_cache [batch_size, head_num, mem_max_seq_len, size_per_head]
    //      cross_attentions [batch_size, head_num, max_decoder_seq_len, mem_max_seq_len] optional float*
    allocateBuffer(input_tensors->at("input_query").shape[0], input_tensors->at("encoder_output").shape[1]);
}

template class DecoderCrossAttentionLayer<float>;
template class DecoderCrossAttentionLayer<half>;

} // namespace space_llm
