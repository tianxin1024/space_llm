#pragma once

#include "layers/BaseLayer.h"
#include "kernels/layernorm_kernels.h"
#include "kernels/add_residual_kernels.h"
#include "utils/activation_types.h"
#include "layers/attention_layers/BaseAttentionLayer.h"
#include "layers/ffnLayer.h"
#include "models/multi_gpu_gpt/ParallelGptDecoderLayerWeight.h"
#include "layers/attention_layers/TensorParallelDecoderSelfAttentionLayer.h"

namespace space_llm {

template <typename T>
class ParallelGptDecoder : public BaseLayer {
private:
    // buffer handling
    size_t max_batch_size_ = 0;
    // meta data
    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t num_layer_;
    size_t expert_num_;
    size_t moe_k_;
    std::vector<int64_t> moe_layer_index_;
    float layernorm_eps_;
    LayerNormType layernorm_type_;
    ActivationType activation_type_;

    // adapter
    bool has_adapters_;
    size_t adapter_inter_size_;
    T *after_adapter_attn_output_;

    // calcuated data
    size_t hidden_units_;

    // buffers
    T *decoder_normed_input_ = nullptr;
    T *self_attn_output_ = nullptr;
    T *normed_self_attn_output_ = nullptr;
    T *decoder_layer_output_ = nullptr;

    T *expert_scales_ = nullptr;
    int *expanded_source_row_to_expanded_dest_row_ = nullptr;
    int *expert_for_source_row_ = nullptr;
    T *fc2_result_ = nullptr;
    T *adapter_fc2_result_ = nullptr;

    BaseAttentionLayer<T> *self_attention_layer_;
    ffnLayer<T> *ffn_layer_;

    void initialize();
    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size);
    void freeBuffer() override;

protected:
    int int8_mode_ = 0;

public:
    ParallelGptDecoder(size_t max_batch_size,
                       size_t head_num,
                       size_t size_per_head,
                       size_t inter_size,
                       size_t num_layer,
                       size_t expert_num,
                       size_t moe_k,
                       std::vector<int64_t> moe_layer_index,
                       float layernorm_eps,
                       gptVariantParams gpt_variant_params,
                       cudaStream_t stream,
                       cublasMMWrapper *cublas_wrapper,
                       IAllocator *allocator,
                       bool is_free_buffer_affer_forward,
                       bool sparse = false,
                       int int8_mode = 0);

    ParallelGptDecoder(ParallelGptDecoder<T> const &decoder);

    ~ParallelGptDecoder();

    void forward(std::unordered_map<std::string, Tensor> *output_tensors,
                 const std::unordered_map<std::string, Tensor> *input_tensors,
                 const std::vector<ParallelGptDecoderLayerWeight<T> *> *decoder_layer_weights);
};

} // namespace space_llm
