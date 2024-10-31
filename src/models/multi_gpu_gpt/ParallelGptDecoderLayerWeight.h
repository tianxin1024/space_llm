#pragma once

#include "kernels/layernorm_kernels.h"
#include "utils/activation_types.h"
#include "layers/attention_layers/AttentionWeight.h"
#include "layers/ffnWeight.h"
#include "utils/memory_utils.h"

namespace space_llm {

struct gptVariantParams {
    // GPT default params
    float layernorm_eps = 1e-6f;
    LayerNormType layernorm_type = LayerNormType::pre_layernorm;
    ActivationType activation_type = ActivationType::Gelu;
    // Whether to have a learnt positional encoding.
    bool has_positional_encoding = true;
    // A layernorm just after the word embedding and before the decoder.
    bool has_pre_decoder_layernorm = false;
    // A layernorm after the decoder.
    bool has_post_decoder_layernorm = true;
    // detoxification adapters. refer to
    bool has_adapters = false;
    size_t adapter_inter_size = 0;
    // Whether to use the attention linear positional bias
    bool use_attention_linear_positional_bias = false;
};

template <typename T>
struct ParallelGptDecoderLayerWeight {
public:
    ParallelGptDecoderLayerWeight() = default;
    ParallelGptDecoderLayerWeight(const int int8_mode);
    ParallelGptDecoderLayerWeight(const int hidden_units,
                                  const int inter_size,
                                  const int tensor_para_size,
                                  const int tensor_para_rank,
                                  const int int8_mode = 0,
                                  gptVariantParams gpt_variant_params = {});
    ~ParallelGptDecoderLayerWeight();
    ParallelGptDecoderLayerWeight(const ParallelGptDecoderLayerWeight &other);
    ParallelGptDecoderLayerWeight &operator=(const ParallelGptDecoderLayerWeight &other);
    void loadModel(std::string dir_path, QKCudaDataType model_file_type);

    LayerNormWeight<T> pre_layernorm_weights;
    AttentionWeight<T> self_attention_weights;
    LayerNormWeight<T> self_attn_layernorm_weights;
    ffnWeight<T> ffn_weights;
    ffnWeight<T> after_attention_adapter_weights;
    ffnWeight<T> after_ffn_adapter_weights;

protected:
    void setWeightPtr();
    void mallocWeights();
    bool isValidLayerParallelId(int l);

    size_t hidden_units_;
    size_t inter_size_;
    size_t tensor_para_size_ = 1;
    size_t tensor_para_rank_ = 0;
    bool is_maintain_buffer = false;
    int int8_mode_ = 0;

    // gpt varians params. e.g. detoxification adapters
    gptVariantParams gpt_variant_params_;

    std::vector<T *> weights_ptr = std::vector<T *>(20, nullptr);

    std::vector<int8_t *> int8_weights_ptr = std::vector<int8_t *>(8, nullptr);
    std::vector<T *> weight_only_scale_ptr = std::vector<T *>(8, nullptr);

    std::vector<float *> scale_ptr = std::vector<float *>(8, nullptr);
    std::vector<float *> scale_out_ptr = std::vector<float *>(8, nullptr);
    std::vector<float *> scale_inter_ptr = std::vector<float *>(8, nullptr);
    cudaStream_t stream_ = 0;
};

} // namespace space_llm
