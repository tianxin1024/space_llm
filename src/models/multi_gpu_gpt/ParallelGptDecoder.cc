#include "models/multi_gpu_gpt/ParallelGptDecoder.h"
#include "layers/attention_layers/TensorParallelReluFfnLayer.h"

namespace space_llm {

template <typename T>
void ParallelGptDecoder<T>::initialize() {
    QK_LOG_DEBUG(__PRETTY_FUNCTION__);

    self_attention_layer_ = new TensorParallelDecoderSelfAttentionLayer<T>(max_batch_size_,
                                                                           head_num_,
                                                                           size_per_head_,
                                                                           stream_,
                                                                           cublas_wrapper_,
                                                                           allocator_,
                                                                           true,
                                                                           is_free_buffer_after_forward_,
                                                                           sparse_,
                                                                           int8_mode_);

    bool use_gated_activation = activation_type_ == ActivationType::GeGLU || activation_type_ == ActivationType::ReGLU;
    size_t max_inter_size = has_adapters_ ? std::max(inter_size_, adapter_inter_size_) : inter_size_;
    if (activation_type_ == ActivationType::Gelu || activation_type_ == ActivationType::GeGLU) {
        // TODO ...
    } else if (activation_type_ == ActivationType::Relu || activation_type_ == ActivationType::ReGLU) {
        ffn_layer_ = new TensorParallelReluFfnLayer<T>(max_batch_size_,
                                                       1,
                                                       head_num_,
                                                       size_per_head_,
                                                       expert_num_, // expert_num
                                                       max_inter_size,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       true,
                                                       is_free_buffer_after_forward_,
                                                       sparse_,
                                                       int8_mode_,
                                                       use_gated_activation);
    }
}

template <typename T>
ParallelGptDecoder<T>::ParallelGptDecoder(size_t max_batch_size,
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
                                          bool is_free_buffer_after_forward,
                                          bool sparse,
                                          int int8_mode) :

    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    max_batch_size_(max_batch_size),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    expert_num_(expert_num),
    moe_k_(moe_k),
    moe_layer_index_(moe_layer_index),
    layernorm_eps_(layernorm_eps),
    layernorm_type_(gpt_variant_params.layernorm_type),
    activation_type_(gpt_variant_params.activation_type),
    adapter_inter_size_(gpt_variant_params.adapter_inter_size),
    has_adapters_(gpt_variant_params.has_adapters),
    hidden_units_(head_num_ * size_per_head_),
    int8_mode_(int8_mode) {
    initialize();
}

template <typename T>
ParallelGptDecoder<T>::ParallelGptDecoder(ParallelGptDecoder<T> const &decoder) :
    BaseLayer(decoder.stream_,
              decoder.cublas_wrapper_,
              decoder.allocator_,
              decoder.is_free_buffer_after_forward_,
              decoder.cuda_device_prop_,
              decoder.sparse_),
    max_batch_size_(decoder.max_batch_size_),
    head_num_(decoder.head_num_),
    size_per_head_(decoder.size_per_head_),
    inter_size_(decoder.inter_size_),
    num_layer_(decoder.num_layer_),
    expert_num_(decoder.expert_num_),
    moe_k_(decoder.moe_k_),
    moe_layer_index_(decoder.moe_layer_index_),
    layernorm_eps_(decoder.layernorm_eps_),
    layernorm_type_(decoder.layernorm_type_),
    activation_type_(decoder.activation_type_),
    adapter_inter_size_(decoder.adapter_inter_size_),
    has_adapters_(decoder.has_adapters_),
    hidden_units_(decoder.hidden_units_),
    int8_mode_(decoder.int8_mode_) {
    initialize();
}

template <typename T>
ParallelGptDecoder<T>::~ParallelGptDecoder() {
    delete self_attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template <typename T>
void ParallelGptDecoder<T>::allocateBuffer() {
    QK_CHECK(false);
}

template <typename T>
void ParallelGptDecoder<T>::allocateBuffer(size_t batch_size) {
    QK_LOG_DEBUG(__PRETTY_FUNCTION__);
    decoder_layer_output_ = reinterpret_cast<T *>(
        allocator_->reMalloc(decoder_layer_output_, sizeof(T) * batch_size * hidden_units_, false));
    decoder_normed_input_ = reinterpret_cast<T *>(
        allocator_->reMalloc(decoder_normed_input_, sizeof(T) * batch_size * hidden_units_, false));
    self_attn_output_ =
        reinterpret_cast<T *>(allocator_->reMalloc(self_attn_output_, sizeof(T) * batch_size * hidden_units_, false));
    normed_self_attn_output_ = reinterpret_cast<T *>(
        allocator_->reMalloc(normed_self_attn_output_, sizeof(T) * batch_size * hidden_units_, false));
    // only allocate additionl buffers when has adapters
    after_adapter_attn_output_ = has_adapters_ ? reinterpret_cast<T *>(allocator_->reMalloc(
                                     after_adapter_attn_output_, sizeof(T) * batch_size * hidden_units_, false)) :
                                                 self_attn_output_;
    // for moe
    // expert_scales_ = reinterpret_cast<T *>(
    //     allocator_->reMalloc(expert_scales_, sizeof(T) * pad_to_multiple_of_16(moe_k_ * batch_size), false));
    // expanded_source_row_to_expanded_dest_row_ = reinterpret_cast<int *>(allocator_->reMalloc(
    //     expanded_source_row_to_expanded_dest_row_, sizeof(int) * pad_to_multiple_of_16(moe_k_ * batch_size), false));
    // expert_for_source_row_ = reinterpret_cast<int *>(
    //     allocator_->reMalloc(expert_for_source_row_, sizeof(int) * pad_to_multiple_of_16(moe_k_ * batch_size), false));
    // fc2_result_ = reinterpret_cast<T *>(allocator_->reMalloc(
    //     fc2_result_, sizeof(T) * pad_to_multiple_of_16(moe_k_ * batch_size * hidden_units_), false));
    // adapter_fc2_result_ =
    //     has_adapters_ ? reinterpret_cast<T *>(allocator_->reMalloc(
    //         adapter_fc2_result_, sizeof(T) * pad_to_multiple_of_16(moe_k_ * batch_size * hidden_units_), false)) :
    //                     nullptr;
    is_allocate_buffer_ = true;
}

template <typename T>
void ParallelGptDecoder<T>::forward(std::unordered_map<std::string, Tensor> *output_tensors,
                                    const std::unordered_map<std::string, Tensor> *input_tensors,
                                    const std::vector<ParallelGptDecoderLayerWeight<T> *> *gpt_decoder_layer_weight) {
    // input tensors:
    //      decoder_input [local_batch_size, hidden_dimension],
    //      finished [local_batch_size],
    //      input_lengths [local_batch_size],
    //      total_padding_tokens [local_batch_size]
    //      max_input_length [1] on cpu
    //      step [1] on cpu
    //      ite [1] on cpu
    //      cache_indirection [local_batch_size / beam_width, beam_width, memory_len]
    //          Here, local_batch_size contains the beam_width, so local_batch_size / beam_width
    //          is real local_batch_size. (optional.)
    //      masked_tokens [local_batch_size, memory_len]
    //      linear_bias_slopes [head_num], optional

    // output tensors:
    //      decoder_output [local_batch_size, hidden_dimension],
    //      key_cache [num_layer, batch_size, head_num, size_per_head // x, memory_len, x]
    //      value_cache [num_layer, batch_size, head_num, memory_len, size_per_head]

    QK_LOG_DEBUG(__PRETTY_FUNCTION__);

    QK_CHECK(input_tensors->count("decoder_input"));
    QK_CHECK(input_tensors->count("finished"));
    QK_CHECK(input_tensors->count("input_lengths"));
    QK_CHECK(input_tensors->count("total_padding_tokens"));
    QK_CHECK(input_tensors->count("max_input_length"));
    QK_CHECK(input_tensors->count("step"));
    QK_CHECK(input_tensors->count("ite"));
    QK_CHECK(input_tensors->count("masked_tokens"));
    QK_CHECK(output_tensors->count("decoder_output"));
    QK_CHECK(output_tensors->count("key_cache"));
    QK_CHECK(output_tensors->count("value_cache"));

    const size_t local_batch_size = input_tensors->at("decoder_input").shape[0];
    allocateBuffer(local_batch_size);

    const DataType data_type = getTensorType<T>();

    const int ite = input_tensors->at("ite").getVal<int>();

    Tensor k_cache = output_tensors->at("key_cache");
    Tensor v_cache = output_tensors->at("value_cache");

    // The resize of the key cache buffer by
    //   (local_batch_size, local_head_num, size_per_head // x, max_seq_len, x) where x is constant.
    std::vector<size_t> self_k_cache_size(k_cache.shape.begin() + 2, k_cache.shape.end());
    self_k_cache_size.insert(self_k_cache_size.begin(), local_batch_size);

    // The resize of the value cache buffer by (local_batch_size, local_head_num, max_seq_len, size_per_head).
    std::vector<size_t> self_v_cache_size(v_cache.shape.begin() + 2, v_cache.shape.end());
    self_v_cache_size.insert(self_v_cache_size.begin(), local_batch_size);

    const auto activation_in_type = int8_mode_ == 2 ? TYPE_INT8 : data_type;
    const auto activation_out_type = data_type;

    for (uint l = 0; l < num_layer_; l++) {
        bool use_moe = std::find(moe_layer_index_.begin(), moe_layer_index_.end(), l) != moe_layer_index_.end();
        T *decoder_input = (l == 0) ? input_tensors->at("decoder_input").getPtr<T>() : decoder_layer_output_;
        T *decoder_output = (l == num_layer_ - 1) ? output_tensors->at("decoder_output").getPtr<T>() : decoder_layer_output_;

        ParallelGptDecoderLayerWeight<T> *layer_weight = gpt_decoder_layer_weight->at(l);

        if (layernorm_type_ == LayerNormType::pre_layernorm) {
            invokeGeneralLayerNorm(decoder_normed_input_,
                                   decoder_input,
                                   layer_weight->pre_layernorm_weights.gamma,
                                   layer_weight->pre_layernorm_weights.beta,
                                   layernorm_eps_,
                                   local_batch_size,
                                   hidden_units_,
                                   const_cast<float *>(layer_weight->self_attention_weights.query_weight.scale),
                                   int8_mode_,
                                   stream_);
        }
        sync_check_cuda_error();

        TensorMap self_attention_input_tensors{
            {"input_query",
             Tensor{MEMORY_GPU,
                    activation_in_type,
                    {local_batch_size, hidden_units_},
                    layernorm_type_ == LayerNormType::pre_layernorm ? decoder_normed_input_ : decoder_input}},
            {"finished", input_tensors->at("finished")},
            {"sequence_lengths", input_tensors->at("input_lengths")},
            {"total_padding_tokens", input_tensors->at("total_padding_tokens")},
            {"max_input_length", input_tensors->at("max_input_length")},
            {"step", input_tensors->at("step")},
            {"masked_tokens", input_tensors->at("masked_tokens")}};
        if (input_tensors->count("cache_indirection")) {
            self_attention_input_tensors.insert("cache_indirection", input_tensors->at("cache_indirection"));
        }
        if (input_tensors->count("linear_bias_slopes")) {
            self_attention_input_tensors.insert("linear_bias_slopes", input_tensors->at("linear_bias_slopes"));
        }

        // maybe have a bug
        size_t cache_offset = l;
        for (auto t = k_cache.shape.begin() + 1; t != k_cache.shape.end(); ++t) {
            cache_offset *= *t;
        };
        size_t ite_cache_offset = ite * local_batch_size;
        for (auto t = k_cache.shape.begin() + 2; t != k_cache.shape.end(); ++t) {
            ite_cache_offset *= *t;
        }
        cache_offset += ite_cache_offset;

        TensorMap self_attention_output_tensors{
            {"hidden_features",
             Tensor(MEMORY_GPU, activation_out_type, {local_batch_size, hidden_units_}, self_attn_output_)},
            {"key_cache", Tensor(MEMORY_GPU, data_type, self_k_cache_size, k_cache.getPtrWithOffset<T>(cache_offset))},
            {"value_cache",
             Tensor(MEMORY_GPU, data_type, self_v_cache_size, v_cache.getPtrWithOffset<T>(cache_offset))}};

        self_attention_layer_->forward(
            &self_attention_output_tensors, &self_attention_input_tensors, &layer_weight->self_attention_weights);

        // the adapter after attention
        if (has_adapters_) {
            invokeGenericActivation<IdentityActivation, T, T>(
                self_attn_output_,
                layer_weight->self_attention_weights.attention_output_weight.bias,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                local_batch_size,
                hidden_units_,
                0,
                nullptr,
                nullptr,
                stream_);

            TensorMap ffn_input_tensors(
                {{"ffn_input", Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, self_attn_output_}}});
            TensorMap ffn_output_tensors(
                {{"ffn_output",
                  Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, after_adapter_attn_output_}}});

            ffn_layer_->resetInterSize(adapter_inter_size_);
            ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &layer_weight->after_attention_adapter_weights);
        }

        if (layernorm_type_ == LayerNormType::pre_layernorm) {
            invokeGeneralAddBiasResidualPreLayerNorm(
                // in case of has_adaptor false isn't it self_attn_output_? i.e.
                //   has_adapters_ ? after_adapter_attn_outpu_ : self_attn_output_,
                has_adapters_ ? after_adapter_attn_output_ : self_attn_output_,
                normed_self_attn_output_,
                has_adapters_ ? after_adapter_attn_output_ : self_attn_output_,
                decoder_input,
                has_adapters_ ? self_attn_output_ : nullptr,
                layer_weight->self_attn_layernorm_weights.gamma,
                layer_weight->self_attn_layernorm_weights.beta,
                has_adapters_ ? layer_weight->after_attention_adapter_weights.output_weight.bias :
                                layer_weight->self_attention_weights.attention_output_weight.bias,
                layernorm_eps_,
                local_batch_size,
                hidden_units_,
                nullptr,
                nullptr,
                const_cast<float *>(layer_weight->ffn_weights.intermediate_weight.scale),
                (float *)nullptr,
                int8_mode_,
                stream_);
        } else if (layernorm_type_ == LayerNormType::post_layernorm) {
            // invokeAddBiasResidualLayerNorm(
            //     // check correctness.
            //     after_adapter_attn_output_,
            //     decoder_input,
            //     has_adapters_ ? layer_weight->after_attention_adapter_weights.output_weight.bias :
            //                     layer_weight->self_attention_weights.attention_output_weight.bias,
            //     layer_weight->pre_layernorm_weights.gamma,
            //     layer_weight->pre_layernorm_weights.beta,
            //     layernorm_eps_,
            //     local_batch_size,
            //     hidden_units_,
            //     stream_);
        }

        sync_check_cuda_error();

        T *ffn_output_ptr = has_adapters_ ? self_attn_output_ : decoder_output;

        TensorMap ffn_input_tensors(
            {{"ffn_input",
              Tensor{MEMORY_GPU,
                     activation_in_type,
                     {local_batch_size, hidden_units_},
                     layernorm_type_ == LayerNormType::pre_layernorm ? normed_self_attn_output_ :
                                                                       after_adapter_attn_output_}}});
        TensorMap ffn_output_tensors;
        if (!use_moe) {
            ffn_output_tensors.insert(
                "ffn_output",
                Tensor{MEMORY_GPU, activation_out_type, {local_batch_size, hidden_units_}, ffn_output_ptr});
        } else {
            // ffn_input_tensors.insert("moe_k", Tensor{MEMORY_CPU, TYPE_UINT64, {1}, &moe_k_});

            // ffn_output_tensors.insert("ffn_output",
            //                           Tensor{MEMORY_GPU,
            //                                  activation_out_type,
            //                                  {moe_k_ * local_batch_size, hidden_units_},
            //                                  has_adapters_ ? adapter_fc2_result_ : fc2_result_});
            // ffn_output_tensors.insert(
            //     "expert_scales", Tensor{MEMORY_GPU, activation_out_type, {local_batch_size, moe_k_}, expert_scales_});
            // ffn_output_tensors.insert(
            //     "expanded_source_row_to_expanded_dest_row",
            //     Tensor{MEMORY_GPU, TYPE_INT32, {local_batch_size, moe_k_}, expanded_source_row_to_expanded_dest_row_});
            // ffn_output_tensors.insert(
            //     "expert_for_source_row",
            //     Tensor{MEMORY_GPU, TYPE_INT32, {local_batch_size, moe_k_}, expert_for_source_row_});
        }

        ffn_layer_->resetInterSize(inter_size_);
        ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &layer_weight->ffn_weights);

        // the adapter after ffn
        if (has_adapters_) {
            TensorMap ffn_input_tensors;
            TensorMap ffn_output_tensors;
            if (!use_moe) {
                invokeGenericActivation<IdentityActivation, T, T>(ffn_output_ptr,
                                                                  layer_weight->ffn_weights.output_weight.bias,
                                                                  nullptr,
                                                                  nullptr,
                                                                  nullptr,
                                                                  nullptr,
                                                                  local_batch_size,
                                                                  hidden_units_,
                                                                  0,
                                                                  nullptr,
                                                                  nullptr,
                                                                  stream_);

                ffn_input_tensors.insert(
                    "ffn_input", Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, ffn_output_ptr});
                ffn_output_tensors.insert(
                    "ffn_output", Tensor{MEMORY_GPU, data_type, {local_batch_size, hidden_units_}, decoder_output});
            } else {
                invokeGenericActivation<IdentityActivation, T, T>(adapter_fc2_result_,
                                                                  layer_weight->ffn_weights.output_weight.bias,
                                                                  nullptr,
                                                                  nullptr,
                                                                  nullptr,
                                                                  nullptr,
                                                                  moe_k_ * local_batch_size,
                                                                  hidden_units_,
                                                                  0,
                                                                  nullptr,
                                                                  nullptr,
                                                                  stream_);

                ffn_input_tensors.insert(
                    "ffn_input",
                    Tensor{MEMORY_GPU, data_type, {moe_k_ * local_batch_size, hidden_units_}, adapter_fc2_result_});
                ffn_output_tensors.insert(
                    "ffn_output",
                    Tensor{MEMORY_GPU, data_type, {moe_k_ * local_batch_size, hidden_units_}, fc2_result_});
            }

            ffn_layer_->resetInterSize(adapter_inter_size_);
            ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &layer_weight->after_ffn_adapter_weights);
        }

        if (!use_moe) {
            if (layernorm_type_ == LayerNormType::pre_layernorm) {
                invokeAddBiasResidual(decoder_output,
                                      decoder_output,
                                      after_adapter_attn_output_,
                                      has_adapters_ ? ffn_output_ptr : nullptr,
                                      has_adapters_ ? layer_weight->after_ffn_adapter_weights.output_weight.bias :
                                                      layer_weight->ffn_weights.output_weight.bias,
                                      nullptr,
                                      nullptr,
                                      local_batch_size,
                                      hidden_units_,
                                      stream_);
            } else if (layernorm_type_ == LayerNormType::post_layernorm) {
                // invokeAddBiasResidualLayerNorm(decoder_output,
                //                                after_adapter_attn_output_,
                //                                has_adapters_ ?
                //                                    layer_weight->after_ffn_adapter_weights.output_weight.bias :
                //                                    layer_weight->ffn_weights.output_weight.bias,
                //                                layer_weight->self_attn_layernorm_weights.gamma,
                //                                layer_weight->self_attn_layernorm_weights.beta,
                //                                layernorm_eps_,
                //                                local_batch_size,
                //                                hidden_units_,
                //                                stream_);
            }
        } else {
            // TODO ...
        }
        sync_check_cuda_error();
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template <typename T>
void ParallelGptDecoder<T>::freeBuffer() {
    if (is_allocate_buffer_) {
        QK_LOG_DEBUG(__PRETTY_FUNCTION__);
        allocator_->free((void **)(&decoder_layer_output_));
        allocator_->free((void **)(&decoder_normed_input_));
        allocator_->free((void **)(&self_attn_output_));
        allocator_->free((void **)(&normed_self_attn_output_));
        if (has_adapters_) {
            allocator_->free((void **)(&after_adapter_attn_output_));
            allocator_->free((void **)(&adapter_fc2_result_));
        }
        is_allocate_buffer_ = false;

        allocator_->free((void **)(&expert_scales_));
        allocator_->free((void **)(&expanded_source_row_to_expanded_dest_row_));
        allocator_->free((void **)(&expert_for_source_row_));
        allocator_->free((void **)(&fc2_result_));
    }
}

template class ParallelGptDecoder<float>;
template class ParallelGptDecoder<half>;

} // namespace space_llm
