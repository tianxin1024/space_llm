#pragma once

#include "layers/ffnLayer.h"

namespace space_llm {

template <typename T>
class TensorParallelGeluFfnLayer : public GeluffnLayer<T> {
protected:
public:
    TensorParallelGeluFfnLayer(size_t max_batch_size,
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
                               bool use_gated_activation = false);

    TensorParallelGeluFfnLayer(TensorParallelGeluFfnLayer<T> const &ffn_layer);

    void forward(std::vector<Tensor> *output_tensors,
                 const std::vector<Tensor> *input_tensors,
                 const ffnWeight<T> *ffn_weights) override;
    void forward(TensorMap *output_tensors, TensorMap *input_tensors, const ffnWeight<T> *ffn_weights) override;
};

} // namespace space_llm
