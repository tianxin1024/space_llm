#pragma once

#include "utils/cuda_utils.h"
#include "utils/memory_utils.h"
#include "utils/activation_types.h"
#include "layers/BaseLayer.h"
#include "layers/ffnWeight.h"

namespace space_llm {

template <typename T>
class ffnLayer : public BaseLayer {
private:
    // buffer handling
    size_t max_token_num_ = 0;

    // meta data
    size_t head_num_;
    size_t size_per_head_;
    size_t expert_num_;

    // calculated data
    size_t hidden_units_;

    // gated activation
    bool use_gated_activation_;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(int moe_k = 0, bool use_moe = false);
    void allocateBuffer(size_t token_num, int moe_k = 0, bool use_moe = false);

protected:
    T *inter_buf_ = nullptr;
    T *inter_buf_2_ = nullptr;

    char *mixed_gemm_workspace_ = nullptr;
    size_t mixed_gemm_ws_bytes_ = 0;
    char *int8_gemm_workspace_ = nullptr;
    size_t int8_gemm_ws_bytes_ = 0;

    size_t inter_size_;
    size_t max_inter_size_;

    // int8_mode_ == 0 means we don't use any mechanism related to INT8.
    // int8_mode_ == 1 for weight quantized only gemm for GPT
    // int8_mode_ == 2 for SmoothQuant O3 (per tensor scales)
    int int8_mode_ = 0;

    virtual ActivationType getActivationType() const {
        return ActivationType::InvalidType;
    }

public:
    ffnLayer(size_t max_batch_size,
             size_t max_seq_len,
             size_t head_num,
             size_t size_per_head,
             size_t expert_num,
             size_t inter_size,
             cudaStream_t stream,
             cublasMMWrapper *cublas_wrapper,
             IAllocator *allocator,
             bool is_free_buffer_after_forward,
             bool sparse = false,
             int int8_mode = 0,
             bool use_gated_activation = false);

    ffnLayer(ffnLayer<T> const &ffn_layer);

    virtual ~ffnLayer();

    void resetInterSize(size_t runtime_inter_size) {
        inter_size_ = runtime_inter_size;
    }

    virtual void forward(std::vector<Tensor> *output_tensors,
                         const std::vector<Tensor> *input_tensors,
                         const ffnWeight<T> *ffn_weights);

    virtual void forward(TensorMap *output_tensors, TensorMap *input_tensors, const ffnWeight<T> *ffn_weights);
};

} // namespace space_llm
