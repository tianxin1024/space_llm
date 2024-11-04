#include "models/multi_gpu_gpt/ParallelGpt.h"

namespace space_llm {

template <typename T>
void ParallelGpt<T>::initialize() {
    gpt_context_decoder_ = new ParallelGptContextDecoder<T>(0,
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
                                                            is_free_buffer_after_forward_,
                                                            is_context_qk_buf_float_,
                                                            attention_type_,
                                                            sparse_,
                                                            int8_mode_);

    gpt_decoder_ = new ParallelGptDecoder<T>(0,
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
                                             is_free_buffer_after_forward_,
                                             sparse_,
                                             int8_mode_);

    // dynamic_decode_layer_ = new DynamicDecodeLayer<float>(vocab_size_,
    //                                                       vocab_size_padded_,
    //                                                       0, // end_id, deprecated
    //                                                       stream_,
    //                                                       cublas_wrapper_,
    //                                                       allocator_,
    //                                                       is_free_buffer_after_forward_,
    //                                                       cuda_device_prop_);
}

template <typename T>
void ParallelGpt<T>::allocateBuffer() {
    QK_CHECK(false);
}

template <typename T>
void ParallelGpt<T>::allocateBuffer(size_t batch_size,
                                    size_t beam_width,
                                    size_t max_session_len,
                                    size_t memory_len,
                                    size_t max_input_len,
                                    bool is_return_context_cum_log_probs) {
    QK_LOG_DEBUG(__PRETTY_FUNCTION__);
    const size_t batchxbeam = batch_size * beam_width;
    const size_t local_batch_size = batch_size;
    QK_CHECK(batch_size % local_batch_size == 0);
    const size_t num_microbatches = batch_size / local_batch_size;

    const size_t self_cache_size =
        (num_layer_)*batchxbeam * memory_len * hidden_units_;

    if (vocab_size_ != vocab_size_padded_) {
        padded_embedding_kernel_ =
            (T *)(allocator_->reMalloc(padded_embedding_kernel_, sizeof(T) * hidden_units_ * vocab_size_padded_, true));
        padded_embedding_kernel_ptr_ = padded_embedding_kernel_;
    }

    tiled_input_attention_mask_ = (T *)(allocator_->reMalloc(
        tiled_input_attention_mask_, sizeof(T) * batchxbeam * max_input_len * max_input_len, false));
    decoder_input_buf_ = (T *)(allocator_->reMalloc(decoder_input_buf_, sizeof(T) * batchxbeam * hidden_units_, false));
    decoder_normed_input_buf_ =
        (T *)(allocator_->reMalloc(decoder_normed_input_buf_, sizeof(T) * batchxbeam * hidden_units_, false));
    decoder_output_buf_ =
        (T *)(allocator_->reMalloc(decoder_output_buf_, sizeof(T) * batchxbeam * hidden_units_, false));
    normed_decoder_output_buf_ =
        (T *)(allocator_->reMalloc(normed_decoder_output_buf_, sizeof(T) * batchxbeam * hidden_units_, false));
    logits_buf_ = (float *)(allocator_->reMalloc(logits_buf_, sizeof(float) * batchxbeam * vocab_size_padded_, false));
    nccl_logits_buf_ =
        (float *)(allocator_->reMalloc(nccl_logits_buf_, sizeof(float) * batchxbeam * vocab_size_padded_, false));
    cum_log_probs_ = (float *)(allocator_->reMalloc(cum_log_probs_, sizeof(float) * batchxbeam, false));
    finished_buf_ = (bool *)(allocator_->reMalloc(finished_buf_, sizeof(bool) * batchxbeam, false));
    sequence_lengths_ = (int *)(allocator_->reMalloc(sequence_lengths_, sizeof(int) * batchxbeam, false));

    key_cache_ = (T *)(allocator_->reMalloc(key_cache_, sizeof(T) * self_cache_size * 2, true));
    value_cache_ = key_cache_ + self_cache_size;
    if (beam_width > 1) {
        cache_indirections_[0] =
            (int *)(allocator_->reMalloc(cache_indirections_[0], sizeof(int) * batchxbeam * memory_len * 2, true));
        cache_indirections_[1] = cache_indirections_[0] + batchxbeam * memory_len;
    }

    tiled_input_ids_buf_ =
        (int *)(allocator_->reMalloc(tiled_input_ids_buf_, sizeof(int) * batchxbeam * max_session_len, true));
    tiled_input_lengths_buf_ = (int *)(allocator_->reMalloc(tiled_input_lengths_buf_, sizeof(int) * batchxbeam, true));

    // prompt_learning weight batch ptrs
    prompt_learning_weight_batch_ =
        (const T **)(allocator_->reMalloc(prompt_learning_weight_batch_, sizeof(T *) * batchxbeam, false));
    tiled_prompt_lengths_buf_ =
        (int *)(allocator_->reMalloc(tiled_prompt_lengths_buf_, sizeof(int) * batchxbeam, false));

    start_ids_buf_ = (int *)(allocator_->reMalloc(start_ids_buf_, sizeof(int) * batch_size, false));
    end_ids_buf_ = (int *)(allocator_->reMalloc(end_ids_buf_, sizeof(int) * batch_size, false));

    transposed_output_ids_buf_ =
        (int *)(allocator_->reMalloc(transposed_output_ids_buf_, sizeof(int) * batchxbeam * max_session_len, true));
    output_ids_buf_ = (int *)(allocator_->reMalloc(output_ids_buf_, sizeof(int) * batchxbeam * max_session_len, true));
    parent_ids_buf_ = (int *)(allocator_->reMalloc(parent_ids_buf_, sizeof(int) * batchxbeam * max_session_len, true));
    seq_limit_len_ = (uint32_t *)(allocator_->reMalloc(seq_limit_len_, sizeof(uint32_t) * batch_size, false));
    tiled_masked_tokens_ =
        (bool *)(allocator_->reMalloc(tiled_masked_tokens_, sizeof(bool) * batchxbeam * memory_len, true));

    context_decoder_input_buf_ = (T *)(allocator_->reMalloc(
        context_decoder_input_buf_, sizeof(T) * batchxbeam * max_input_len * hidden_units_, false));
    context_decoder_output_buf_ = (T *)(allocator_->reMalloc(
        context_decoder_output_buf_, sizeof(T) * batchxbeam * max_input_len * hidden_units_, false));
    output_log_probs_buf_ =
        (float *)(allocator_->reMalloc(output_log_probs_buf_, sizeof(float) * batchxbeam * max_session_len, false));

    if (gpt_variant_params_.has_pre_decoder_layernorm) {
        context_decoder_normed_input_buf_ = (T *)allocator_->reMalloc(
            context_decoder_normed_input_buf_, sizeof(T) * batchxbeam * max_input_len * hidden_units_, false);
        decoder_normed_input_buf_ =
            (T *)allocator_->reMalloc(decoder_normed_input_buf_, sizeof(T) * batchxbeam * hidden_units_, false);
    }

    if (gpt_variant_params_.use_attention_linear_bias) {
        linear_bias_slopes_ = (T *)(allocator_->reMalloc(linear_bias_slopes_, sizeof(T) * head_num_, false));
    }

    if (is_return_context_cum_log_probs) {
        lp_normed_decoder_output_buf_ = (T *)allocator_->reMalloc(
            lp_normed_decoder_output_buf_, sizeof(T) * batchxbeam * max_input_len * hidden_units_);
        lp_logits_buf_ = (float *)allocator_->reMalloc(lp_logits_buf_,
                                                       sizeof(float) * batchxbeam * max_input_len * vocab_size_padded_);
        lp_nccl_logits_buf_ = (float *)allocator_->reMalloc(
            lp_nccl_logits_buf_, sizeof(float) * batchxbeam * max_input_len * vocab_size_padded_);
        lp_logprob_buf_ = (float *)allocator_->reMalloc(lp_logprob_buf_, sizeof(float) * batchxbeam * max_input_len);
    }
    if (shared_contexts_ratio_ > 0.0f) {
        shared_contexts_idx_ = (int *)allocator_->reMalloc(shared_contexts_idx_, batch_size * sizeof(int), false);
        batch_to_compact_idx_ = (int *)allocator_->reMalloc(batch_to_compact_idx_, batchxbeam * sizeof(int), false);
        compact_idx_ = (int *)allocator_->reMalloc(compact_idx_, batch_size * sizeof(int), false);
        compact_size_ = (int *)allocator_->reMalloc(compact_size_, sizeof(int), false);
    }
    microbatch_should_stop_ =
        (bool *)allocator_->reMalloc(microbatch_should_stop_, sizeof(bool) * num_microbatches, true, true);
    tiled_total_padding_count_ =
        (int *)allocator_->reMalloc(tiled_total_padding_count_, batchxbeam * sizeof(int), false);

    is_allocate_buffer_ = true;
}

template <typename T>
void ParallelGpt<T>::freeBuffer() {
    if (is_allocate_buffer_) {
        if (vocab_size_ != vocab_size_padded_) {
            padded_embedding_kernel_ptr_ = nullptr;
            allocator_->free((void **)(&padded_embedding_kernel_));
        }

        allocator_->free((void **)(&tiled_input_attention_mask_));
        allocator_->free((void **)(&decoder_input_buf_));
        allocator_->free((void **)(&decoder_output_buf_));
        allocator_->free((void **)(&normed_decoder_output_buf_));
        allocator_->free((void **)(&logits_buf_));
        allocator_->free((void **)(&nccl_logits_buf_));
        allocator_->free((void **)(&cum_log_probs_));
        allocator_->free((void **)(&finished_buf_));
        allocator_->free((void **)(&sequence_lengths_));

        allocator_->free((void **)(&key_cache_));
        if (cache_indirections_[0] != nullptr) {
            allocator_->free((void **)(&cache_indirections_)[0]);
        }

        allocator_->free((void **)(&tiled_input_ids_buf_));
        allocator_->free((void **)(&tiled_input_lengths_buf_));

        allocator_->free((void **)(&prompt_learning_weight_batch_));
        allocator_->free((void **)(&tiled_prompt_lengths_buf_));

        allocator_->free((void **)(&transposed_output_ids_buf_));
        allocator_->free((void **)(&output_ids_buf_));
        allocator_->free((void **)(&parent_ids_buf_));
        allocator_->free((void **)(&tiled_masked_tokens_));

        allocator_->free((void **)(&seq_limit_len_));

        allocator_->free((void **)(&start_ids_buf_));
        allocator_->free((void **)(&end_ids_buf_));

        allocator_->free((void **)(&context_decoder_input_buf_));
        allocator_->free((void **)(&context_decoder_output_buf_));
        allocator_->free((void **)(&output_log_probs_buf_));

        if (gpt_variant_params_.has_pre_decoder_layernorm) {
            allocator_->free((void **)(&context_decoder_normed_input_buf_));
            allocator_->free((void **)(&decoder_normed_input_buf_));
        }
        if (gpt_variant_params_.use_attention_linear_bias) {
            allocator_->free((void **)(&linear_bias_slopes_));
        }

        allocator_->free((void **)(&lp_normed_decoder_output_buf_));
        allocator_->free((void **)(&lp_logits_buf_));
        allocator_->free((void **)(&lp_nccl_logits_buf_));
        allocator_->free((void **)(&lp_logprob_buf_));

        allocator_->free((void **)(&microbatch_should_stop_), true);

        if (shared_contexts_ratio_ > 0.0f) {
            allocator_->free((void **)(&shared_contexts_idx_));
            allocator_->free((void **)(&compact_size_));
        }
        allocator_->free((void **)(&tiled_total_padding_count_));

        is_allocate_buffer_ = false;
    }
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
    // delete dynamic_decode_layer_;
    freeBuffer();
}

template class ParallelGpt<float>;
template class ParallelGpt<half>;

} // namespace space_llm
