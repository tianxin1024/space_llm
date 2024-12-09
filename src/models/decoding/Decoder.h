#pragma once

#include "kernels/add_residual_kernels.h"
#include "layers/BaseLayer.h"
#include "layers/attention_layers/DecoderSelfAttentionLayer.h"
#include "layers/attention_layers/DecoderCrossAttentionLayer.h"
#include "layers/ffnLayer.h"
#include "models/decoding/DecoderLayerWeight.h"

namespace space_llm {

template <typename T>
class Decoder : public BaseLayer {
private:
    // buffer handling
    size_t max_batch_size_ = 0;
    // meta data
    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t num_layer_;
    size_t hidden_units_;
    static constexpr float layernorm_eps_ = 1e-6f;

    BaseAttentionLayer<T> *self_attention_layer_;
    BaseAttentionLayer<T> *cross_attention_layer_;
    ffnLayer<T> *ffn_layer_;

    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size);
    void freeBuffer() override;
    bool isValidBatchSize(size_t batch_size);

    void initialize();

protected:
    T *decoder_normed_input_ = nullptr;
    T *self_attn_output_ = nullptr;
    T *normed_self_attn_output_ = nullptr;
    T *cross_attn_output_ = nullptr;
    T *normed_cross_attn_output_ = nullptr;
    T *decoder_layer_output_ = nullptr;

public:
    Decoder(size_t max_batch_size,
            size_t head_num,
            size_t size_per_head,
            size_t inter_size,
            size_t num_layer,
            cudaStream_t stream,
            cublasMMWrapper *cublas_wrapper,
            IAllocator *allocator,
            bool is_free_buffer_after_forward);

    Decoder(Decoder<T> const &decoder);

    ~Decoder();

    void forward(std::vector<Tensor> *output_tensors,
                 const std::vector<Tensor> *input_tensors,
                 const std::vector<DecoderLayerWeight<T>> *decoder_layer_weights);
};
} // namespace space_llm
