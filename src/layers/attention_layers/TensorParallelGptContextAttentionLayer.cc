#include "layers/attention_layers/TensorParallelGptContextAttentionLayer.h"

namespace space_llm {

template <typename T>
void TensorParallelGptContextAttentionLayer<T>::forward(TensorMap *output_tensors,
                                                        TensorMap *input_tensors,
                                                        const AttentionWeight<T> *attention_weights) {
    // input_tensors:
    //      input_query [batch_size * seq_len, hidden_dimendion]
    //      attention_mask [batch_size, 1, seq_len, seq_len]
    //      is_final_layer [1], bool on cpu
    //
    // output_tensors:
    //      hidden_features [batch_size * seq_len, hidden_dimension]
    //      key_cache [batch, local_head_num, size_per_head // x, max_seqlen, x]
    //      value_cache [batch, local_head_num, max_seq_len, size_per_head]
    const size_t size = output_tensors->at("hidden_features").size();
    std::vector<Tensor> reduce_tensor{output_tensors->at("hidden_features")};

    GptContextAttentionLayer<T>::forward(output_tensors, input_tensors, attention_weights);
}

template <typename T>
TensorParallelGptContextAttentionLayer<T>::TensorParallelGptContextAttentionLayer(
    size_t max_batch_size,
    size_t max_seq_len,
    size_t head_num,
    size_t size_per_head,
    cudaStream_t stream,
    cublasMMWrapper *cublas_wrapper,
    IAllocator *allocator,
    bool do_all_reduce,
    bool is_free_buffer_after_forward,
    bool is_qk_buf_float,
    bool sparse,
    int int8_mode) :
    GptContextAttentionLayer<T>(max_batch_size,
                                max_seq_len,
                                head_num,
                                size_per_head,
                                head_num,
                                stream,
                                cublas_wrapper,
                                allocator,
                                is_free_buffer_after_forward,
                                is_qk_buf_float,
                                sparse,
                                int8_mode) {
}

template <typename T>
TensorParallelGptContextAttentionLayer<T>::TensorParallelGptContextAttentionLayer(
    size_t max_batch_size,
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
    bool sparse,
    int int8_mode) :
    GptContextAttentionLayer<T>(max_batch_size,
                                max_seq_len,
                                head_num,
                                size_per_head,
                                head_num,
                                rotary_embedding_dim,
                                neox_rotary_style,
                                stream,
                                cublas_wrapper,
                                allocator,
                                is_free_buffer_after_forward,
                                is_qk_buf_float,
                                sparse,
                                int8_mode) {
}

template <typename T>
TensorParallelGptContextAttentionLayer<T>::TensorParallelGptContextAttentionLayer(
    TensorParallelGptContextAttentionLayer<T> const &attention_layer) :
    GptContextAttentionLayer<T>(attention_layer) {
}

template class TensorParallelGptContextAttentionLayer<float>;
template class TensorParallelGptContextAttentionLayer<half>;

} // namespace space_llm
