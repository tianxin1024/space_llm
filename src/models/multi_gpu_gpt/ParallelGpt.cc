#include "models/multi_gpu_gpt/ParallelGpt.h"
#include "kernels/gpt_kernels.h"

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

template <typename T>
void ParallelGpt<T>::forward(std::vector<Tensor> *output_tensors,
                             const std::vector<Tensor> *input_tensors,
                             const ParallelGptWeight<T> *gpt_weights) {
    // input_tensors:
    //      input_ids [batch_size, max_input_length]
    //      input_lengths [batch_size]
    //      max_output_seq_len [1] on cpu

    // output_tensors:
    //      output_ids [batch_size, beam, max_output_seq_len]
    //      sequence_length [batch_size, beam]
    //      output_log_probs [batch_size, beam, request_output_seq_len], must be float*.
    //          It leads to additional computing cost. If we don't need this result, please put nullptr
    //      cum_log_probs [batch_size, beam], must be float*, optional
    //          The cumulative log probability of generated sequences. It leads additional computing cost.

    // Step is from max_input_length ~ max_output_seq_len,
    // When step = k,  we put output ids and caches at step k, and the sequence_length would be k - 1 before
    // complete this step.
    // When there is no input_ids, put the start token at step 0 of output_ids_buf_. After forward, only copy
    // the step 1 ~ max_output_seq_len of output_ids_buf_ to output_tensors->at(0).data

    std::unordered_map<std::string, Tensor> input_tensors_map{{"input_ids", input_tensors->at(0)},
                                                              {"input_lengths", input_tensors->at(1)},
                                                              {"max_output_seq_len", input_tensors->at(2)}};
    input_tensors_map.insert({"random_seed", {MEMORY_CPU, TYPE_INT32, {1}, &random_seed_}});
    input_tensors_map.insert({"runtime_top_k", {MEMORY_CPU, TYPE_UINT32, {1}, &top_k_}});
    input_tensors_map.insert({"runtime_top_p", {MEMORY_CPU, TYPE_FP32, {1}, &top_p_}});

    std::unordered_map<std::string, Tensor> output_tensors_map{{"output_ids", output_tensors->at(0)},
                                                               {"sequence_length", output_tensors->at(1)},
                                                               {"output_log_probs", output_tensors->at(2)}};

    if (output_tensors->size() > 3) {
        output_tensors_map.insert({"cum_log_probs", output_tensors->at(4)});
    }

    forward(&output_tensors_map, &input_tensors_map, gpt_weights);
}

template <typename T>
void ParallelGpt<T>::forward(std::unordered_map<std::string, Tensor> *output_tensors,
                             const std::unordered_map<std::string, Tensor> *input_tensors,
                             const ParallelGptWeight<T> *gpt_weights) {
    // input_tensors:
    //      input_ids [batch_size, max_input_length]
    //      input_lengths [batch_size]
    //      input_lengths_h [batch_size] on cpu, optional
    //      prompt_learning_task_name_ids [batch_size] on cpu
    //      output_seq_len [batch_size] on cpu
    //      stop_words_list [batch_size, 2, stop_words_length], optional
    //      bad_words_list [2, bad_words_length] or [batch_size, 2, bad_words_length], optional
    //      start_id [batch_size] on cpu, optional
    //      end_id [batch_size] on cpu, optional
    //      runtime_top_k [1] or [batch_size] on cpu, optional, uint.
    //      runtime_top_p [1] or [batch_size] on cpu, optional, float.
    //      beam_search_diversity_rate [1] or [batch_size] on cpu, optional, float.
    //      temperature [1] or [batch_size] on cpu, optional, float.
    //      len_penalty [1] or [batch_size] on cpu, optional, float.
    //      repetition_penalty [1] or [batch_size] on cpu, optional, float.
    //      presence_penalty [1] or [batch_size] on cpu, optional, float.
    //          Only one of repetition and presence penalties is allowed.
    //      min_length [1] or [batch_size] on cpu, optional, int
    //      random_seed [1] or [batch_size] on cpu, optional, unsigned long long int.
    //      request_prompt_lengths [batch_size], optional
    //      request_prompt_lengths_h [batch_size], cpu, optional
    //      request_prompt_embedding [batch_size, max_prompt_length, hidden_units], float, optional
    //      request_prompt_type [batch_size], int, optional
    //      is_return_context_cum_log_probs [1] on cpu, bool, optional
    //      session_len [1] on cpu, uint32, optional
    //      memory_len [1] on cpu, uint32, optional
    //      continue_gen [1] on cpu, bool, optional
    //      is_return_context_embeddings [1] on cpu, bool, optional
    //      top_p_decay [batch_size] on gpu, float, optional
    //      top_p_min [batch_size] on gpu, float, optional
    //      top_p_reset_ids [batch_size] on gpu, uint32, optional

    // output_tensors:
    //      output_ids [batch_size, beam_width, max_output_seq_len]
    //      sequence_length [batch_size, beam_width]
    //      response_input_lengths [batch_size, beam_width], optional
    //      output_log_probs [batch_size, beam_width, request_output_seq_len], must be float*.
    //          optional. It leads to additional computing cost. If we don't need this result, don't put it.
    //      cum_log_probs [batch_size, beam_width], must be float*. optional.
    //          The cumulative log probability of generated sequences. It may lead to additional computing cost.
    //      context_embeddings [batch_size, hidden_units], must be float*, optional

    // Step is from max_input_length ~ max_output_seq_len,
    // When step = k,  we put output ids and caches at step k, and the sequence_length would be k - 1 before
    // complete this step.
    // When there is no input_ids, put the start token at step 0 of output_ids_buf_. After forward, only copy
    // the step 1 ~ max_output_seq_len of output_ids_buf_ to output_tensors->at(0).data

    QK_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    QK_CHECK_WITH_INFO(input_tensors->size() >= 3, "input_tensors->size() >= 3");
    QK_CHECK_WITH_INFO(output_tensors->size() >= 2, "output_tensors->size() >= 2");
    QK_CHECK(input_tensors->at("input_ids").shape.size() == 2);
    QK_CHECK(input_tensors->at("input_lengths").shape.size() == 1);
    QK_CHECK(input_tensors->find("output_seq_len") != input_tensors->end()
             && input_tensors->at("output_seq_len").shape.size() == 1);
    QK_CHECK(output_tensors->at("output_ids").shape.size() == 3);
    QK_CHECK(output_tensors->at("sequence_length").shape.size() == 2);
    QK_CHECK_WITH_INFO(input_tensors->at("input_ids").shape[0] == output_tensors->at("output_ids").shape[0],
                       "input_tensors->at(\"input_ids\").shape[0] == output_tensors->at(\"output_ids\").shape[0]");

    // Used when inputs do not contain random_seed
    const size_t batch_size = output_tensors->at("output_ids").shape[0];
    const size_t beam_width = output_tensors->at("output_ids").shape[1];

    QK_CHECK_WITH_INFO(output_tensors->count("cum_log_probs") == 0
                           || output_tensors->at("cum_log_probs").size() == batch_size * beam_width,
                       "The shape of cum_log_probs should match with batch_size x beam_width if provided.");

    int max_input_length = input_tensors->at("input_ids").shape[1];
    bool continue_gen = input_tensors->find("continue_gen") != input_tensors->end() ?
                            input_tensors->at("continue_gen").getVal<bool>() :
                            false;

    const bool is_return_context_embeddings =
        input_tensors->find("is_return_context_embeddings") != input_tensors->end() && input_tensors->at("is_return_context_embeddings").getVal<bool>();
    if (is_return_context_embeddings) {
        QK_CHECK_WITH_INFO(output_tensors->find("context_embeddings") != output_tensors->end(),
                           "When requesting context embeddings, a context embeddings output tensors must be provided");
    }

    const int initial_step = continue_gen ? step_ : 0;
    int max_context_len = max_input_length + initial_step;

    // NOTE: the input already contains the p/prompt-tunning tokens ids for p/prompt tuning task
    // prompt_learning_task_name_ids are used by both p/prompt-tunning and prefix_prompt task
    const int *prompt_learning_task_name_ids =
        input_tensors->count("prompt_learning_task_name_ids") ?
            input_tensors->at("prompt_learning_task_name_ids").getPtr<const int>() :
            nullptr;

    QK_CHECK_WITH_INFO(
        !(prompt_learning_task_name_ids != nullptr
          && (prompt_learning_type_ == PromptLearningType::no_prompt
              || prompt_learning_type_ == PromptLearningType::soft_prompt)),
        "prompt_learning_type is prefix_prompt either p_prompt_tuning when prompt_learning_task_name_ids are provided.");

    PromptLearningType request_prompt_type = PromptLearningType::no_prompt;
    int valid_prompt_inputs = input_tensors->count("request_prompt_type")
                              + input_tensors->count("request_prompt_lengths")
                              + input_tensors->count("request_prompt_embedding");

    if (valid_prompt_inputs == 3) {
        request_prompt_type = static_cast<PromptLearningType>(input_tensors->at("request_prompt_type").getVal<int>());
        if (prompt_learning_task_name_ids != nullptr) {
            QK_LOG_INFO("Apply prompt embedding from input, will ignore task name ids");
        }
    } else if (valid_prompt_inputs > 0) {
        QK_LOG_WARNING(
            "Prompts not applied: request_prompt_embedding, request_prompt_lengths, request_prompt_type are all needed!");
    }
    if (request_prompt_type == PromptLearningType::prefix_prompt) {
        QK_LOG_WARNING("Request prompt doesn't support prefix prompt currently!");
    }

    // whether or not use prompt embeddings from the request.
    // If true, staticlly loaded prompts weights during model loading and task name ids will be ignored
    bool use_request_p_prompt_embedding = request_prompt_type == PromptLearningType::p_prompt_tuning;
    int max_request_p_prompt_length =
        use_request_p_prompt_embedding ? input_tensors->at("request_prompt_embedding").shape[1] : 0;
    // p_prompt tuning: input and prompt are concatnenated (not separate),
    const uint32_t *input_lengths_h = input_tensors->count("input_lengths_h") ?
                                          input_tensors->at("input_lengths_h").getPtr<const uint32_t>() :
                                          nullptr;

    size_t max_input_without_prompt_length = max_context_len;
    if (use_request_p_prompt_embedding && input_lengths_h != nullptr
        && input_tensors->count("request_prompt_lengths_h")) {
        const uint32_t *request_prompt_lengths_h =
            input_tensors->at("request_prompt_lengths_h").getPtr<const uint32_t>();
        max_input_without_prompt_length = input_lengths_h[0] - request_prompt_lengths_h[0];
        for (int bs_id = 1; bs_id < batch_size; ++bs_id) {
            max_input_without_prompt_length = std::max(size_t(input_lengths_h[bs_id] - request_prompt_lengths_h[bs_id]),
                                                       max_input_without_prompt_length);
        }
    }

    has_prefix_prompt_ =
        (prompt_learning_task_name_ids != nullptr && prompt_learning_type_ == PromptLearningType::prefix_prompt);
    has_p_prompt_tuning_ =
        prompt_learning_task_name_ids != nullptr && prompt_learning_type_ == PromptLearningType::p_prompt_tuning
        || use_request_p_prompt_embedding;
    bool use_loaded_p_prompt_embedding = has_p_prompt_tuning_ && !use_request_p_prompt_embedding;
    has_prefix_soft_prompt_ = request_prompt_type == PromptLearningType::soft_prompt;

    // NOTE: soft prompt
    QK_CHECK_WITH_INFO(!(has_prefix_soft_prompt_ && continue_gen),
                       "Interactive Generations cannot work with prefix_soft_prompt !");
    const size_t max_prefix_soft_prompt_length =
        has_prefix_soft_prompt_ ? input_tensors->at("request_prompt_embedding").shape[1] : 0;
    const size_t limit_len_offset = max_prefix_soft_prompt_length + (max_input_length == 0 ? 1 : 0);
    const size_t gen_len = input_tensors->at("output_seq_len").max<uint32_t>() + limit_len_offset;

    size_t session_len = 0;
    if (continue_gen) {
        session_len = session_len_; // Record the size of allocated buffer in previous round.
    } else if (input_tensors->find("session_len") != input_tensors->end()) {
        session_len = input_tensors->at("session_len").getVal<uint32_t>(); // Use for allocate buffer in first round.
    } else {
        session_len = gen_len; // When the interactive generation mode is disabled.
    }
    session_len_ = session_len;
    QK_CHECK_WITH_INFO(
        gen_len + initial_step <= session_len,
        fmtstr("Session size too low (%d) vs. total output size (%d)", session_len, gen_len + initial_step));
    size_t memory_len = 0;
    if (continue_gen) {
        memory_len = memory_len_; // Record the size of allocated buffer in previous round.
    } else if (input_tensors->find("memory_len") != input_tensors->end()) {
        memory_len = input_tensors->at("memory_len").getVal<uint32_t>(); // Use for allocate buffer in first round.
    } else {
        memory_len = session_len; // When the interactive generation mode is disabled.
    }
    memory_len_ = memory_len;
    /* TODO: could remove this constraint by changing how context decoder operates */
    QK_CHECK_WITH_INFO(max_input_length <= memory_len,
                       fmtstr("Memory size too low (%d) vs. input length (%d)", memory_len, max_input_length));

    if (memory_len < session_len) {
        QK_LOG_WARNING("memory_len (%d) is less than session_len (%d). "
                       "Note that this reduces the memory cost of k/v cache, but may hurt the accuracy.",
                       memory_len,
                       session_len);
    } else if (memory_len > session_len) {
        QK_LOG_WARNING("memory_len (%d) is larger than session_len (%d). "
                       "This may lead to additional memory cost. Suggest to use smaller memory_len.",
                       memory_len,
                       session_len);
    }

    if (gpt_variant_params_.has_positional_encoding && session_len_ > gpt_weights->getMaxSeqLen()) {
        QK_LOG_ERROR("The session_len_ (%d) of request is longer than max_seq_len (%d) of embedding table."
                     " This is a invalid input. Setting the session_len_ to %d.",
                     session_len_,
                     gpt_weights->getMaxSeqLen(),
                     gpt_weights->getMaxSeqLen());
        session_len_ = gpt_weights->getMaxSeqLen();
    }

    const bool is_return_context_cum_log_probs = input_tensors->count("is_return_context_cum_log_probs") > 0
                                                 && input_tensors->at("is_return_context_cum_log_probs").getVal<bool>();
    if (is_return_context_cum_log_probs) {
        QK_CHECK_WITH_INFO(output_tensors->count("cum_log_probs")
                               && output_tensors->at("cum_log_probs").data != nullptr,
                           "`cum_log_probs` must be provided in `output_tensors` in order to enable "
                           "the cumulative log probability computation of input contexts.");
    }

    QK_LOG_INFO("buffer allocation");
    if (!continue_gen) {
        allocateBuffer(batch_size,
                       beam_width,
                       session_len,
                       memory_len,
                       max_input_length + max_prefix_soft_prompt_length,
                       is_return_context_cum_log_probs);
        sync_check_cuda_error();
    }
    setSeqLimitLen(seq_limit_len_, input_tensors->at("output_seq_len"), limit_len_offset, batch_size);

    const DataType data_type = getTensorType<T>();
    const cudaDataType_t gemm_data_type = getCudaDataType<T>();

    const std::vector<size_t> self_k_cache_shape = {num_layer_,
                                                    batch_size * beam_width,
                                                    local_head_num_,
                                                    size_per_head_ / (16 / sizeof(T)),
                                                    memory_len,
                                                    16 / sizeof(T)};
    const std::vector<size_t> self_v_cache_shape = {
        num_layer_, batch_size * beam_width, local_head_num_, memory_len, size_per_head_};

    // {
    //     TensorMap input_map(*input_tensors);

    //     QK_LOG_INFO("dynamic decode setup");
    //     dynamic_decode_layer_->setup(batch_size, beam_width, &input_map);
    //     handleOptArg(&input_map, "start_id", start_ids_buf_, start_id_, batch_size);
    //     handleOptArg(&input_map, "end_id", end_ids_buf_, end_id_, batch_size);
    // }

    // if (gpt_variant_params_.use_attention_linear_bias) {
    //     QK_LOG_INFO("build alibi slopes");
    //     invokeBuildAlibiSlopes(linear_bias_slopes_, head_num_, stream_);
    // }
}

template class ParallelGpt<float>;
template class ParallelGpt<half>;

} // namespace space_llm
