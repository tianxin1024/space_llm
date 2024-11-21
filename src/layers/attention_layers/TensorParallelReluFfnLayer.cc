#include "layers/attention_layers/TensorParallelReluFfnLayer.h"
#include "layers/ffnLayer.h"

namespace space_llm {

template <typename T>
void TensorParallelReluFfnLayer<T>::forward(std::vector<Tensor> *output_tensors,
                                            const std::vector<Tensor> *input_tensors,
                                            const ffnWeight<T> *ffn_weights) {
    TensorMap input_tensor({{"ffn_input", input_tensors->at(0)}});
    TensorMap output_tensor({{"ffn_output", output_tensors->at(0)}});
    forward(&output_tensor, &input_tensor, ffn_weights);
}

template <typename T>
void TensorParallelReluFfnLayer<T>::forward(TensorMap *output_tensors,
                                            TensorMap *input_tensors,
                                            const ffnWeight<T> *ffn_weights) {
    Tensor out_tensor = output_tensors->at("ffn_output");
    const size_t token_num = out_tensor.shape[0];
    const size_t hidden_units = out_tensor.shape[1];

    std::vector<Tensor> swap_tensors = {out_tensor};

    bool use_custom_all_reduce_kernel = false;

    ReluffnLayer<T>::forward(output_tensors, input_tensors, ffn_weights);

    T *ffn_out = out_tensor.getPtr<T>();
    QK_LOG_INFO("FFN all reduce sum");
}

template <typename T>
TensorParallelReluFfnLayer<T>::TensorParallelReluFfnLayer(size_t max_batch_size,
                                                          size_t max_seq_len,
                                                          size_t head_num,
                                                          size_t size_per_head,
                                                          size_t expert_num,
                                                          size_t inter_size,
                                                          cudaStream_t stream,
                                                          cublasMMWrapper *cublas_wrapper,
                                                          IAllocator *allocator,
                                                          bool do_all_reduce,
                                                          bool is_free_buffer_after_forward,
                                                          bool is_sparse,
                                                          int int8_mode,
                                                          bool use_gated_activation) :
    ReluffnLayer<T>(max_batch_size,
                    max_seq_len,
                    head_num,
                    size_per_head,
                    expert_num,
                    inter_size,
                    stream,
                    cublas_wrapper,
                    allocator,
                    is_free_buffer_after_forward,
                    is_sparse,
                    int8_mode,
                    use_gated_activation) {
}

template <typename T>
TensorParallelReluFfnLayer<T>::TensorParallelReluFfnLayer(TensorParallelReluFfnLayer<T> const &ffn_layer) :
    ReluffnLayer<T>(ffn_layer) {
}

template class TensorParallelReluFfnLayer<float>;
template class TensorParallelReluFfnLayer<half>;

} // namespace space_llm
