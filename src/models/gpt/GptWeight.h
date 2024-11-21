#pragma once

#include "kernels/layernorm_kernels.h"
#include "utils/memory_utils.h"
#include "utils/prompt_learning.h"
#include "models/gpt/GptDecoderLayerWeight.h"

namespace space_llm {

template <typename T>
struct GptWeight {
    GptWeight() = default;
    GptWeight(const int hidden_units,
              const int inter_size,
              const int vocab_size,
              const int num_layer,
              const int max_seq_len,
              const int tensor_para_size,
              const int tensor_para_rank,
              const int layer_para_size,
              const int layer_para_rank,
              const int int8_mode = 0,
              PromptLearningType prompt_learning_type = PromptLearningType::no_prompt,
              std::map<std::string, std::pair<int, int>> prompt_learning_pair = {},
              gptVariantParams gpt_variant_params = {});

    ~GptWeight();
    GptWeight(const GptWeight &other);
    GptWeight &operator=(const GptWeight &other);
    void loadModel(std::string dir_path);
    void resizeLayer(const int num_layer, const int int8_mode = 0);

    std::vector<GptDecoderLayerWeight<T> *> decoder_layer_weights;
    const T *position_encoding_table = nullptr;
    const T *pre_decoder_embedding_table = nullptr;
    LayerNormWeight<T> pre_decoder_layernorm;
    LayerNormWeight<T> post_decoder_layernorm;
    DenseWeight<T> post_decoder_embedding;

    /*
       prompt_learning_pair = vectors of [weight ptr, prompt length] pair
       prompt_length is stored here for compatible prompt learning table
       prefix_prompt weights store as shape [num_layers, 2, num_heads, perfix_seq_len, size_per_head]
       p/prompt tuning weights store as shape [prompt_len, hidden_units]
       idx is the task_name_id of the prompt tables
    */
    std::vector<std::pair<const T *, int>> prompt_learning_table = {};

    inline size_t getMaxSeqLen() const {
        return max_seq_len_;
    }
    inline size_t setMaxSeqLen(size_t max_seq_len) {
        max_seq_len_ = max_seq_len;
    }

private:
    void setWeightPtr();
    void mallocWeights();
    bool isValidLayerParallelId(int l);

    size_t hidden_units_;
    size_t inter_size_;
    size_t vocab_size_;
    size_t num_layer_;
    size_t max_seq_len_;
    size_t tensor_para_size_;
    size_t tensor_para_rank_;
    size_t layer_para_size_;
    size_t layer_para_rank_;
    size_t int8_mode_ = 0;
    bool shared_embed_ = false;

    // gpt variants: e.g. meta opt
    gptVariantParams gpt_variant_params_;

    // prompt learning pair (task_name, (task_name_id, prompt_len))
    PromptLearningType prompt_learning_type_;
    std::map<std::string, std::pair<int, int>> prompt_learning_pair_;
    bool malloc_load_prompt_weights_ = false;

    // each prompt token's weight size
    size_t prompt_token_weight_size_ = 0;

    bool is_maintain_buffer = false;

    // The number of base weights of the GPT model: According to the variant params,
    // positional encoding or pre decoder layernorm can be nullptr.
    //  - 1 for the positional encoding. (optional)
    //  - 1 for the pre word embedding tables.
    //  - 2 for the pre decoder layernorm. (optional)
    //  - 2 for the post decoder layernorm. (optional)
    //  - 2 for the post word embedding tables.
    size_t num_base_weights = 7;
    // weight pointers of length num_base_weights.
    std::vector<T *> weights_ptr = std::vector<T *>(num_base_weights);
};

} // namespace space_llm
