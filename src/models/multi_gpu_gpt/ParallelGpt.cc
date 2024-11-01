#include "models/multi_gpu_gpt/ParallelGpt.h"

namespace space_llm {

template <typename T>
void ParallelGpt<T>::initialize() {
    gpt_context_decoder_ = new ParalelGptContextDecoder<T>(0,
                                                           0,
                                                           head_num_,
                                                           size_per_head_,
                                                           inter_size_,
                                                           num_layer_,
                                                           expert_num_,
                                                           moe_k_,
                                                           moe_layer_index_,
                                                           layernorm_eps_,
                                                           gpt_variant_params_,
                                                           stream_,
                                                           cublas_wrapper_,
                                                           allocator_,
                                                           if_free_buffer_after_forward_,
                                                           is_context_qk_buf_float_,
                                                           attention_type_,
                                                           sparse_,
                                                           int8_mode_,
                                                           enable_custom_all_reduce_);
}

template <typename T>
ParallelGpt<T>::ParallelGpt(size_t max_batch_size,
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
                            int prompt_learning_start_id,
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
                            cudaDeviceProp *cuda_device_prop,
                            AttentionType attention_type,
                            bool sparse,
                            int int8_mode,
                            int enable_custom_all_reduce,
                            float shared_contexts_ratio) :
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop, sparse),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    expert_num_(expert_num),
    moe_k_(moe_k),
    moe_layer_index_(moe_layer_index),
    vocab_size_(vocab_size),
    start_id_(start_id),
    end_id_(end_id),
    prompt_learning_start_id_(prompt_learning_start_id),
    prompt_learning_type_(prompt_learning_type),
    layernorm_eps_(gpt_variant_params.layernorm_eps),
    gpt_variant_params_(gpt_variant_params),
    beam_search_diversity_rate_(beam_search_diversity_rate),
    hidden_units_(head_num_ * size_per_head),
    top_k_(top_k),
    top_p_(top_p),
    random_seed_(random_seed),
    temperature_(temperature),
    len_penalty_(len_penalty),
    repetition_penalty_(repetition_penalty),
    local_head_num_(head_num),
    attention_type_(attention_type),
    int8_mode_(int8_mode),
    enable_custom_all_reduce_(enable_custom_all_reduce),
    shared_contexts_ratio_(shared_contexts_ratio) {
    int local_vacab_size = ceil(vocab_size_ / 1.f);

    if (std::is_same<half, T>::value) {
        local_vacab_size = ceil(local_vacab_size / 8.f) * 8;
    }
    vocab_size_padded_ = (size_t)local_vacab_size;
    initialize();
}

template <typename T>
ParallelGpt<T>::ParallelGpt(ParallelGpt<T> const &gpt) :
    BaseLayer(gpt),
    head_num_(gpt.head_num_),
    size_per_head_(gpt.size_per_head_),
    inter_size_(gpt.inter_size_),
    num_layer_(gpt.num_layer_),
    expert_num_(gpt.expert_num_),
    moe_k_(gpt.moe_k_),
    moe_layer_index_(gpt.moe_layer_index_),
    vocab_size_(gpt.vocab_size_),
    start_id_(gpt.start_id_),
    end_id_(gpt.end_id_),
    prompt_learning_start_id_(gpt.prompt_learning_start_id_),
    prompt_learning_type_(gpt.prompt_learning_type_),
    beam_search_diversity_rate_(gpt.beam_search_diversity_rate_),
    layernorm_eps_(gpt.gpt_variant_params_.layernorm_eps),
    gpt_variant_params_(gpt.gpt_variant_params_),
    hidden_units_(gpt.hidden_units_),
    top_k_(gpt.top_k_),
    top_p_(gpt.top_p_),
    random_seed_(gpt.random_seed_),
    temperature_(gpt.temperature_),
    len_penalty_(gpt.len_penalty_),
    repetition_penalty_(gpt.repetition_penalty_),
    local_head_num_(gpt.local_head_num_),
    vocab_size_padded_(gpt.vocab_size_padded_),
    attention_type_(gpt.attention_type_),
    int8_mode_(gpt.int8_mode_),
    enable_custom_all_reduce_(gpt.enable_custom_all_reduce_),
    shared_contexts_ratio_(gpt.shared_contexts_ratio_) {
    initialize();
}

template <typename T>
ParallelGpt<T>::~ParallelGpt() {
    delete gpt_decoder_;
    delete gpt_context_decoder_;
    delete dynamic_decode_layer_;
    freeBuffer();
}

} // namespace space_llm
