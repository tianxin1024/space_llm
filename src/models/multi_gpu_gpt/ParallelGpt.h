#pragma once

#include "utils/prompt_learning.h"
#include "layers/BaseLayer.h"
#include "layers/attention_layers/BaseAttentionLayer.h"
#include "models/multi_gpu_gpt/ParallelGptDecoderLayerWeight.h"

namespace space_llm {

template <typename T>
class ParallelGpt : public BaseLayer {
private:
    // meta data
    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t num_layer_;
    size_t vocab_size_;
    size_t expert_num_;
    size_t moe_k_;
    std::vector<int64_t> moe_layer_index_;

    int start_id_;
    int end_id_;
    float beam_search_diversity_rate_;
    size_t hidden_units_;

    const float layernorm_eps_;
    float shared_contexts_ratio_;

    size_t top_k_;
    float top_p_;
    unsigned long long random_seed_;

    float temperature_;
    float len_penalty_;
    float repetition_penalty_;

    size_t local_head_num_;

    int enable_custom_all_reduce_;

    const bool is_context_qk_buf_float_ =
        (std::getenv("CONTEXT_ATTENTION_BMM1_HALF_ACCUM") == nullptr || std::string(std::getenv("CONTEXT_ATTENTION_BMM1_HALF_ACCUM")) != "ON");
    size_t vocab_size_padded_;
    const int int8_mode_ = 0;
    AttentionType attention_type_ = AttentionType::UNFUSED_MHA;

    // Prompt Learning Parameters
    PromptLearningType prompt_learning_type_;
    int prompt_learning_start_id_; // start_id for prompt_learning (only needed by prefix prompts)
    bool has_p_prompt_tuning_;
    bool has_prefix_prompt_;
    bool has_prefix_soft_prompt_;

    // GPT Variants parameters: e.g. Meta OPT
    gptVariantParams gpt_variant_params_;

public:
    ParallelGpt(size_t max_batch_size,
                size_t max_seq_len,
                size_t max_input_len,
                size_t beam_width,
                size_t head_num,
                size_t size_per_head,
                size_t inter_size,
                size_t num_layer,
                size_t expert_num,
                size_t moe_k,
                std::vector<int64_t> moe_layer_index,
                size_t vocab_size,
                int start_id,
                int end_id,
                int prompt_learning_start_id, // only needed by p/prompt-tuning
                PromptLearningType prompt_learning_type,
                gptVariantParams gpt_variant_params,
                float beam_search_diversity_rate,
                size_t top_k,
                float top_p,
                unsigned long long random_seed,
                float temperature,
                float len_penalty,
                float repetition_penalty,
                cudaStream_t stream,
                cublasMMWrapper *cublas_wrapper,
                IAllocator *allocator,
                bool is_free_buffer_after_forward,
                cudaDeviceProp *cuda_device_prop = nullptr,
                AttentionType attention_type = AttentionType::UNFUSED_MHA,
                bool sparse = false,
                int int8_mode = 0,
                // std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm = nullptr,
                int enable_custom_all_reduce = 0,
                float shared_contexts_ratio = 1.0f);

    ParallelGpt(ParallelGpt<T> const &gpt);
};

} // namespace space_llm
