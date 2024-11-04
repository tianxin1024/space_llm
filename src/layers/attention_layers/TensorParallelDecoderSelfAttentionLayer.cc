#include "layers/attention_layers/TensorParallelDecoderSelfAttentionLayer.h"

namespace space_llm {
template <typename T>
TensorParallelDecoderSelfAttentionLayer<T>::TensorParallelDecoderSelfAttentionLayer(
    size_t max_batch_size,
    size_t head_num,
    size_t size_per_head,
    size_t rotary_embedding_dim,
    bool neox_rotary_style,
    size_t d_model,
    float q_scaling,
    cudaStream_t stream,
    cublasMMWrapper *cublas_wrapper,
    IAllocator *allocator,
    bool do_all_reduce,
    bool is_free_buffer_after_forward,
    bool is_sparse,
    int int8_mode) :
    DecoderSelfAttentionLayer<T>(max_batch_size,
                                 head_num,
                                 size_per_head,
                                 head_num,
                                 rotary_embedding_dim,
                                 neox_rotary_style,
                                 d_model,
                                 q_scaling, // NOTE
                                 stream,
                                 cublas_wrapper,
                                 allocator,
                                 is_free_buffer_after_forward,
                                 is_sparse,
                                 int8_mode) {
}

template <typename T>
TensorParallelDecoderSelfAttentionLayer<T>::TensorParallelDecoderSelfAttentionLayer(
    size_t max_batch_size,
    size_t head_num,
    size_t size_per_head,
    cudaStream_t stream,
    cublasMMWrapper *cublas_wrapper,
    IAllocator *allocator,
    bool do_all_reduce,
    bool is_free_buffer_after_forward,
    bool is_sparse,
    int int8_mode) :
    TensorParallelDecoderSelfAttentionLayer(max_batch_size,
                                            head_num,
                                            size_per_head,
                                            0,
                                            false,
                                            head_num * size_per_head,
                                            1.0f,
                                            stream,
                                            cublas_wrapper,
                                            allocator,
                                            do_all_reduce,
                                            is_free_buffer_after_forward,
                                            is_sparse,
                                            int8_mode) {
}

template <typename T>
TensorParallelDecoderSelfAttentionLayer<T>::TensorParallelDecoderSelfAttentionLayer(
    size_t max_batch_size,
    size_t head_num,
    size_t size_per_head,
    size_t d_model,
    float q_scaling,
    cudaStream_t stream,
    cublasMMWrapper *cublas_wrapper,
    IAllocator *allocator,
    bool do_all_reduce,
    bool is_free_buffer_after_forward,
    bool is_sparse,
    int int8_mode) :
    TensorParallelDecoderSelfAttentionLayer(max_batch_size,
                                            head_num,
                                            size_per_head,
                                            0,
                                            false,
                                            d_model,
                                            q_scaling,
                                            stream,
                                            cublas_wrapper,
                                            allocator,
                                            do_all_reduce,
                                            is_free_buffer_after_forward,
                                            is_sparse,
                                            int8_mode) {
}

template <typename T>
TensorParallelDecoderSelfAttentionLayer<T>::TensorParallelDecoderSelfAttentionLayer(
    size_t max_batch_size,
    size_t head_num,
    size_t size_per_head,
    size_t rotary_embedding_dim,
    bool neox_rotary_style,
    cudaStream_t stream,
    cublasMMWrapper *cublas_wrapper,
    IAllocator *allocator,
    bool do_all_reduce,
    bool is_free_buffer_after_forward,
    bool is_sparse,
    int int8_mode) :
    TensorParallelDecoderSelfAttentionLayer(max_batch_size,
                                            head_num,
                                            size_per_head,
                                            rotary_embedding_dim,
                                            neox_rotary_style,
                                            head_num * size_per_head,
                                            1.0f,
                                            stream,
                                            cublas_wrapper,
                                            allocator,
                                            do_all_reduce,
                                            is_free_buffer_after_forward,
                                            is_sparse,
                                            int8_mode) {
}

template <typename T>
TensorParallelDecoderSelfAttentionLayer<T>::TensorParallelDecoderSelfAttentionLayer(
    TensorParallelDecoderSelfAttentionLayer<T> const &attention_layer) :
    DecoderSelfAttentionLayer<T>(attention_layer) {
}

template <typename T>
void TensorParallelDecoderSelfAttentionLayer<T>::forward(TensorMap *output_tensors,
                                                         TensorMap *input_tensors,
                                                         const AttentionWeight<T> *attention_weights) {
    // input tensors:
    //      attention_input [batch_size, hidden_dimension],
    //      finished [batch_size],
    //      sequence_lengths [batch_size]
    //      input_lengths [batch_size]
    //      max_input_length [1] on cpu
    //      step [1] on cpu

    // output tensors:
    //      attention_output [batch_size, hidden_dimension],
    //      key_cache [batch, head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [batch, head_num, max_seq_len, size_per_head]

    const size_t size = output_tensors->at("hidden_features").size();
    std::vector<Tensor> reduce_tensor{output_tensors->at("hidden_features")};

    bool use_custom_all_reduce_kernel = false;
    DecoderSelfAttentionLayer<T>::forward(output_tensors, input_tensors, attention_weights);

    T *attention_out = output_tensors->getPtr<T>("hidden_features");
}

template class TensorParallelDecoderSelfAttentionLayer<float>;
template class TensorParallelDecoderSelfAttentionLayer<half>;

} // namespace space_llm
