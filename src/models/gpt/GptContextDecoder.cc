#include "kernels/preprocess_kernels.h"
#include "kernels/gpt_kernels.h"
#include "models/gpt/GptContextDecoder.h"
#include "layers/attention_layers/TensorParallelGptContextAttentionLayer.h"

namespace space_llm {

template <typename T>
void GptContextDecoder<T>::initialize() {
    QK_LOG_DEBUG(__PRETTY_FUNCTION__);
    self_attention_layer_ = new TensorParallelGptContextAttentionLayer<T>(max_batch_size_,
                                                                          max_seq_len_,
                                                                          head_num_,
                                                                          size_per_head_,
                                                                          stream_,
                                                                          cublas_wrapper_,
                                                                          allocator_,
                                                                          true,
                                                                          is_free_buffer_after_forward_,
                                                                          is_qk_buf_float_,
                                                                          sparse_,
                                                                          int8_mode_);

    bool use_gated_activation = activation_type_ == ActivationType::GeGLU || activation_type_ == ActivationType::ReGLU;
    size_t max_inter_size = has_adapters_ ? std::max(inter_size_, adapter_inter_size_) : inter_size_;
    if (activation_type_ == ActivationType::Gelu || activation_type_ == ActivationType::GeGLU) {
        // ffn_layer_ = new TensorParallelGeluFfnLayer<T>(max_batch_size_,
        //                                                max_seq_len_,
        //                                                head_num_,
        //                                                size_per_head_,
        //                                                expert_num_, // expert_num
        //                                                max_inter_size,
        //                                                tensor_para_,
        //                                                stream_,
        //                                                cublas_wrapper_,
        //                                                allocator_,
        //                                                true,
        //                                                is_free_buffer_after_forward_,
        //                                                sparse_,
        //                                                int8_mode_,
        //                                                use_gated_activation,
        //                                                custom_all_reduce_comm_,
        //                                                enable_custom_all_reduce_);
    } else if (activation_type_ == ActivationType::Relu || activation_type_ == ActivationType::ReGLU) {
        ffn_layer_ = new TensorParallelReluFfnLayer<T>(max_batch_size_,
                                                       max_seq_len_,
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
void GptContextDecoder<T>::allocateBuffer() {
    QK_CHECK(false);
}

template <typename T>
void GptContextDecoder<T>::allocateBuffer(size_t batch_size, size_t seq_len, bool use_shared_contexts) {
    QK_LOG_DEBUG(__PRETTY_FUNCTION__);
    decoder_normed_input_ = reinterpret_cast<T *>(
        allocator_->reMalloc(decoder_normed_input_, sizeof(T) * batch_size * seq_len * hidden_units_, false));
    self_attn_output_ = reinterpret_cast<T *>(
        allocator_->reMalloc(self_attn_output_, sizeof(T) * batch_size * seq_len * hidden_units_, false));
    normed_self_attn_output_ = decoder_normed_input_; // reuse the buffer
    // only allocate additionl buffers when has adapters
    after_adapter_attn_output_ =
        has_adapters_ ? reinterpret_cast<T *>(
            allocator_->reMalloc(after_adapter_attn_output_, sizeof(T) * batch_size * seq_len * hidden_units_, false)) :
                        self_attn_output_;
    decoder_layer_output_ = reinterpret_cast<T *>(
        allocator_->reMalloc(decoder_layer_output_, sizeof(T) * batch_size * seq_len * hidden_units_, false));
    if (int8_mode_ == 2) {
        // dynamic scale
        attention_query_dynamic_scale_ = reinterpret_cast<float *>(
            allocator_->reMalloc(attention_query_dynamic_scale_, sizeof(float) * batch_size * seq_len, true));
        ffn_intermediate_dynamic_scale_ = reinterpret_cast<float *>(
            allocator_->reMalloc(ffn_intermediate_dynamic_scale_, sizeof(float) * batch_size * seq_len, true));
    }
    h_pinned_token_num_ptr_ = (size_t *)allocator_->reMalloc(h_pinned_token_num_ptr_, sizeof(size_t), true, true);
    padding_offset_ =
        reinterpret_cast<int *>(allocator_->reMalloc(padding_offset_, sizeof(int) * batch_size * seq_len, false));
    cu_seqlens_ = reinterpret_cast<int *>(allocator_->reMalloc(cu_seqlens_, sizeof(int) * (batch_size + 1), false));
    // for moe
    // expert_scales_ = reinterpret_cast<T *>(
    //     allocator_->malloc(sizeof(T) * pad_to_multiple_of_16(moe_k_ * batch_size * seq_len), false));
    // expanded_source_row_to_expanded_dest_row_ = reinterpret_cast<int *>(
    //     allocator_->malloc(sizeof(int) * pad_to_multiple_of_16(moe_k_ * batch_size * seq_len), false));
    // expert_for_source_row_ = reinterpret_cast<int *>(
    //     allocator_->malloc(sizeof(int) * pad_to_multiple_of_16(moe_k_ * batch_size * seq_len), false));
    // fc2_result_ = reinterpret_cast<T *>(
    //     allocator_->malloc(sizeof(T) * pad_to_multiple_of_16(moe_k_ * batch_size * seq_len * hidden_units_), false));
    // adapter_fc2_result_ =
    //     has_adapters_ ? reinterpret_cast<T *>(allocator_->malloc(
    //         sizeof(T) * pad_to_multiple_of_16(moe_k_ * batch_size * seq_len * hidden_units_), false)) :
    //                     nullptr;
    is_allocate_buffer_ = true;

    if (use_shared_contexts) {
        compact_decoder_features_ = reinterpret_cast<T *>(
            allocator_->reMalloc(compact_decoder_features_, sizeof(T) * batch_size * seq_len * hidden_units_, false));
        compact_attention_mask_ = reinterpret_cast<T *>(
            allocator_->reMalloc(compact_attention_mask_, sizeof(T) * batch_size * seq_len * seq_len, false));
        compact_input_lengths_ =
            reinterpret_cast<int *>(allocator_->reMalloc(compact_input_lengths_, sizeof(int) * batch_size, false));
        k_cache_layer_ = reinterpret_cast<T *>(
            allocator_->reMalloc(k_cache_layer_, sizeof(T) * batch_size * seq_len * hidden_units_, false));
        v_cache_layer_ = reinterpret_cast<T *>(
            allocator_->reMalloc(v_cache_layer_, sizeof(T) * batch_size * seq_len * hidden_units_, false));
    }
}

template <typename T>
void GptContextDecoder<T>::freeBuffer() {
    if (is_allocate_buffer_) {
        QK_LOG_DEBUG(__PRETTY_FUNCTION__);
        allocator_->free((void **)(&decoder_normed_input_));
        allocator_->free((void **)(&self_attn_output_));
        if (has_adapters_) {
            allocator_->free((void **)(&after_adapter_attn_output_));
            allocator_->free((void **)(&adapter_fc2_result_));
        }
        allocator_->free((void **)(&decoder_layer_output_));
        allocator_->free((void **)(&h_pinned_token_num_ptr_), true);
        allocator_->free((void **)(&padding_offset_));
        allocator_->free((void **)(&cu_seqlens_));

        allocator_->free((void **)(&expert_scales_));
        allocator_->free((void **)(&expanded_source_row_to_expanded_dest_row_));
        allocator_->free((void **)(&expert_for_source_row_));
        allocator_->free((void **)(&fc2_result_));

        if (compact_attention_mask_ != nullptr) {
            allocator_->free((void **)(&compact_decoder_features_));
            allocator_->free((void **)(&compact_attention_mask_));
            allocator_->free((void **)(&compact_input_lengths_));
            allocator_->free((void **)(&k_cache_layer_));
            allocator_->free((void **)(&v_cache_layer_));
        }
        if (int8_mode_ == 2) {
            allocator_->free((void **)(&attention_query_dynamic_scale_));
            allocator_->free((void **)(&ffn_intermediate_dynamic_scale_));
        }
        is_allocate_buffer_ = false;
    }
}

template <typename T>
GptContextDecoder<T>::GptContextDecoder(size_t max_batch_size,
                                        size_t max_seq_len,
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
                                        bool is_qk_buf_float,
                                        AttentionType attention_type,
                                        bool sparse,
                                        int int8_mode) :
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
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
    hidden_units_(head_num_ * size_per_head),
    is_qk_buf_float_(is_qk_buf_float),
    attention_type_(attention_type),
    int8_mode_(int8_mode) {
    initialize();
}

template <typename T>
GptContextDecoder<T>::GptContextDecoder(GptContextDecoder<T> const &decoder) :
    BaseLayer(decoder.stream_, decoder.cublas_wrapper_, decoder.allocator_, decoder.is_free_buffer_after_forward_),
    max_batch_size_(decoder.max_batch_size_),
    max_seq_len_(decoder.max_seq_len_),
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
    is_qk_buf_float_(decoder.is_qk_buf_float_),
    attention_type_(decoder.attention_type_),
    int8_mode_(decoder.int8_mode_) {
    initialize();
}

template <typename T>
GptContextDecoder<T>::~GptContextDecoder() {
    delete self_attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template <typename T>
void GptContextDecoder<T>::forward(
    TensorMap *output_tensors,
    const TensorMap *input_tensors,
    const std::vector<GptDecoderLayerWeight<T> *> *gpt_decoder_layer_weight) {
    // input tensors:
    //      decoder_input [batch_size, seq_len, hidden_dimension],
    //      attention_mask [batch_size, 1, seq_len, seq_len]
    //      input_lengths [batch_size]
    //      compact_idx [compact_size], optional
    //      batch_to_compact_idx [batch_size], optional
    //      linear_bias_slopes [head_num], optional

    // output tensors:
    //      decoder_output [batch_size, seq_len, hidden_dimension],
    //      key_cache [num_layer, batch, local_head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [num_layer, batch, local_head_num, max_seq_len, size_per_head]
    //      last_token_hidden_units [batch_size, hidden_dimension]

    // To use layer/pipeline parallelism, we view the shape of 'batch_size' to 'ite * local_batch_size'.
    // For example, the shape of decoder_input becomes [ite, batch_size, seq_len, hidden_dimension] during
    // computing.

    QK_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    QK_CHECK(input_tensors->isExist("decoder_input"));
    QK_CHECK(input_tensors->isExist("attention_mask"));
    QK_CHECK(input_tensors->isExist("input_lengths"));
    QK_CHECK(output_tensors->isExist("decoder_output"));
    QK_CHECK(output_tensors->isExist("key_cache"));
    QK_CHECK(output_tensors->isExist("value_cache"));
    QK_CHECK(output_tensors->isExist("last_token_hidden_units"));

    const bool use_shared_contexts = input_tensors->isExist("compact_idx");
    QK_CHECK(!use_shared_contexts || input_tensors->isExist("batch_to_compact_idx"));

    Tensor decoder_input_tensor = input_tensors->at("decoder_input");
    QK_CHECK(decoder_input_tensor.shape[2] == hidden_units_);

    // Request batch size
    const size_t request_batch_size = decoder_input_tensor.shape[0];
    // Maybe compacted batch size.
    const size_t batch_size =
        use_shared_contexts ? input_tensors->at("compact_idx").shape[0] : decoder_input_tensor.shape[0];
    // Request input length
    const size_t seq_len = decoder_input_tensor.shape[1];
    // The maximum length of generation.
    const size_t max_seq_len = output_tensors->at("value_cache").shape[3];

    const DataType data_type = getTensorType<T>();

    allocateBuffer(batch_size, seq_len, use_shared_contexts);

    if (use_shared_contexts) {
        invokeCompactInputs(compact_decoder_features_,
                            compact_attention_mask_,
                            compact_input_lengths_,
                            decoder_input_tensor.getPtr<T>(),
                            input_tensors->at("attention_mask").getPtr<T>(),
                            input_tensors->at("input_lengths").getPtr<int>(),
                            input_tensors->at("compact_idx").getPtr<int>(),
                            batch_size,
                            seq_len,
                            hidden_units_,
                            stream_);
    }

    const size_t local_batch_size = 1;
    QK_CHECK(batch_size % local_batch_size == 0);
    const size_t iteration_num = batch_size / local_batch_size;

    Tensor k_cache = output_tensors->at("key_cache");
    Tensor v_cache = output_tensors->at("value_cache");

    const auto activation_in_type = int8_mode_ == 2 ? TYPE_INT8 : data_type;
    const auto activation_out_type = data_type;

    // The resize of the key cache buffer by
    //   (local_batch_size, local_head_num, size_per_head // x, max_seq_len, x) where x is constant.
    std::vector<size_t> self_k_cache_size(k_cache.shape.begin() + 2, k_cache.shape.end());
    self_k_cache_size.insert(self_k_cache_size.begin(), local_batch_size);

    // The resize of the value cache buffer by
    //   (local_batch_size, local_head_num, max_seq_len, size_per_head).
    std::vector<size_t> self_v_cache_size(v_cache.shape.begin() + 2, v_cache.shape.end());
    self_v_cache_size.insert(self_v_cache_size.begin(), local_batch_size);

    if (use_shared_contexts) {
        // we use k_cache_layer_ and v_cache_layer_
        self_k_cache_size[3] = seq_len;
        self_v_cache_size[2] = seq_len;
    }

    AttentionType attention_type = attention_type_;
    const bool is_unpadded_mha = isUnPaddedMHA(attention_type);

    for (uint ite = 0; ite < iteration_num; ite++) {
        size_t h_token_num = local_batch_size * seq_len;
        if (is_unpadded_mha) {
            const int *base_input_lengths =
                (use_shared_contexts ? compact_input_lengths_ : input_tensors->at("input_lengths").getPtr<int>());
            invokeGetPaddingOffsetAndCuSeqLens(h_pinned_token_num_ptr_,
                                               &h_token_num,
                                               padding_offset_,
                                               cu_seqlens_,
                                               base_input_lengths + ite * local_batch_size,
                                               local_batch_size,
                                               seq_len,
                                               stream_);
        }

        for (uint l = 0; l < num_layer_; l++) {
            bool use_moe = std::find(moe_layer_index_.begin(), moe_layer_index_.end(), l) != moe_layer_index_.end();
            if (isValidLayerParallelId(l) == false) {
                continue;
            }

            if (l == 0 && is_unpadded_mha) {
                const T *base_input =
                    (use_shared_contexts ? compact_decoder_features_ : decoder_input_tensor.getPtr<T>());
                invokeRemovePadding(decoder_layer_output_,
                                    base_input + ite * local_batch_size * seq_len * hidden_units_,
                                    padding_offset_,
                                    h_token_num,
                                    hidden_units_,
                                    stream_);
            }

            GptDecoderLayerWeight<T> *layer_weight = gpt_decoder_layer_weight->at(l);

            T *decoder_input = decoder_layer_output_;
            T *decoder_output = decoder_layer_output_;
            if (!is_unpadded_mha) {
                if (l == 0) {
                    decoder_input = use_shared_contexts ? compact_decoder_features_ : decoder_input_tensor.getPtr<T>();
                    decoder_input += ite * local_batch_size * seq_len * hidden_units_;
                }
                if (l == num_layer_ - 1) {
                    decoder_output = use_shared_contexts ? compact_decoder_features_ :
                                                           output_tensors->at("decoder_output").getPtr<T>();
                    decoder_output += ite * local_batch_size * seq_len * hidden_units_;
                }
            }

            if (layernorm_type_ == LayerNormType::pre_layernorm) {
                // TODO tianxin debug
                invokeGeneralLayerNorm(decoder_normed_input_,
                                       decoder_input,
                                       layer_weight->pre_layernorm_weights.gamma,
                                       layer_weight->pre_layernorm_weights.beta,
                                       layernorm_eps_,
                                       h_token_num,
                                       hidden_units_,
                                       const_cast<float *>(layer_weight->self_attention_weights.query_weight.scale),
                                       nullptr,
                                       /* dynamic_quant_ ? attention_query_dynamic_scale_ : nullptr, */
                                       int8_mode_,
                                       stream_);
            }
            sync_check_cuda_error();

            const bool is_final = false; // TODO(bhsueh) remove this flag

            const T *attention_ptr =
                use_shared_contexts ? compact_attention_mask_ : input_tensors->at("attention_mask").getPtr<T>();

            TensorMap self_attention_input_tensors{
                {"input_query",
                 Tensor{MEMORY_GPU,
                        activation_in_type,
                        {h_token_num, hidden_units_},
                        layernorm_type_ == LayerNormType::pre_layernorm ? decoder_normed_input_ : decoder_input}},
                {"attention_mask",
                 Tensor{MEMORY_GPU,
                        data_type,
                        {local_batch_size, 1, seq_len, seq_len},
                        attention_ptr + local_batch_size * ite * seq_len * seq_len}},
                {"attention_type", Tensor{MEMORY_CPU, TYPE_VOID, {1}, &attention_type}},
                {"is_final_layer", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &is_final}},
                {"layer_id", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &l}}};

            if (is_unpadded_mha) {
                self_attention_input_tensors.insert("padding_offset",
                                                    Tensor{MEMORY_GPU, TYPE_INT32, {h_token_num}, padding_offset_});
                self_attention_input_tensors.insert(
                    "cu_seqlens", Tensor{MEMORY_GPU, TYPE_INT32, {size_t(local_batch_size + 1)}, cu_seqlens_});
            }

            /* if (dynamic_quant_) { */
            /*     self_attention_input_tensors.insert("attention_query_dynamic_scale", */
            /*         Tensor{MEMORY_GPU, TYPE_FP32, {h_token_num}, attention_query_dynamic_scale_}); */
            /* } */

            if (input_tensors->isExist("linear_bias_slopes")) {
                self_attention_input_tensors.insert("linear_bias_slopes", input_tensors->at("linear_bias_slopes"));
            }

            // The key/value cache stride per batch.
            const size_t cache_stride_per_batch = hidden_units_ / 1 * max_seq_len;
            // The key/value cache offset of the layer.
            const size_t cache_layer_offset =
                (l - getFirstLayerParallelId()) * request_batch_size * cache_stride_per_batch;
            // The key/value cache offset of the current local batch iteration.
            const size_t ite_cache_offset = ite * local_batch_size * cache_stride_per_batch;
            const size_t cache_offset = cache_layer_offset + ite_cache_offset;

            T *k_cache_ptr = use_shared_contexts ? k_cache_layer_ : k_cache.getPtrWithOffset<T>(cache_offset);
            T *v_cache_ptr = use_shared_contexts ? v_cache_layer_ : v_cache.getPtrWithOffset<T>(cache_offset);

            TensorMap self_attention_output_tensors{
                {"hidden_features",
                 Tensor{MEMORY_GPU, activation_out_type, {h_token_num, hidden_units_}, self_attn_output_}},
                {"key_cache", Tensor{MEMORY_GPU, data_type, self_k_cache_size, k_cache_ptr}},
                {"value_cache", Tensor{MEMORY_GPU, data_type, self_v_cache_size, v_cache_ptr}}};

            self_attention_layer_->forward(
                &self_attention_output_tensors, &self_attention_input_tensors, &layer_weight->self_attention_weights);

            if (use_shared_contexts) {
                // Even with local batches, we must process the whole K/V caches as any
                // element in batch_idx_to_compact_idx may reference the local batch
                // we're processing. We also need to discard references that aren't in
                // that particular local batch.
                invokeUnCompactCaches(k_cache.getPtrWithOffset<T>(cache_layer_offset),
                                      v_cache.getPtrWithOffset<T>(cache_layer_offset),
                                      k_cache_layer_,
                                      v_cache_layer_,
                                      input_tensors->at("batch_to_compact_idx").getPtr<int>(),
                                      request_batch_size, // batch_size (uncompact)
                                      v_cache.shape[2],   // local_head_num
                                      max_seq_len,
                                      seq_len,
                                      size_per_head_,
                                      local_batch_size,
                                      ite,
                                      stream_);
                sync_check_cuda_error();
            }

            // the adapter after attention (only pre layernorm currently)
            if (has_adapters_) {
                invokeGenericActivation<IdentityActivation, T, T>(
                    self_attn_output_,
                    layer_weight->self_attention_weights.attention_output_weight.bias,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    h_token_num,
                    hidden_units_,
                    0,
                    nullptr,
                    nullptr,
                    stream_);

                TensorMap ffn_input_tensors(
                    {{"ffn_input", Tensor{MEMORY_GPU, data_type, {h_token_num, hidden_units_}, self_attn_output_}}});
                TensorMap ffn_output_tensors(
                    {{"ffn_output",
                      Tensor{MEMORY_GPU, data_type, {h_token_num, hidden_units_}, after_adapter_attn_output_}}});

                ffn_layer_->resetInterSize(adapter_inter_size_);
                ffn_layer_->forward(&ffn_output_tensors,
                                    &ffn_input_tensors,
                                    &gpt_decoder_layer_weight->at(l)->after_attention_adapter_weights);
            }

            if (layernorm_type_ == LayerNormType::pre_layernorm) {
                invokeGeneralAddBiasResidualPreLayerNorm(
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
                    h_token_num,
                    hidden_units_,
                    nullptr,
                    nullptr,
                    const_cast<float *>(layer_weight->ffn_weights.intermediate_weight.scale),
                    nullptr, // NOTE (perkzz): dynamic_quant_ ? ffn_intermediate_dynamic_scale_ : nullptr,
                    int8_mode_,
                    stream_);
            } else if (layernorm_type_ == LayerNormType::post_layernorm) {
                // TODO tianxin
                // invokeAddBiasResidualLayerNorm(after_adapter_attn_output_,
                //                                decoder_input,
                //                                has_adapters_ ?
                //                                    layer_weight->after_attention_adapter_weights.output_weight.bias :
                //                                    layer_weight->self_attention_weights.attention_output_weight.bias,
                //                                layer_weight->pre_layernorm_weights.gamma,
                //                                layer_weight->pre_layernorm_weights.beta,
                //                                layernorm_eps_,
                //                                h_token_num,
                //                                hidden_units_,
                //                                stream_);
            }
            sync_check_cuda_error();

            T *ffn_output_ptr = has_adapters_ ? self_attn_output_ : decoder_output;

            TensorMap ffn_input_tensors(
                {{"ffn_input",
                  Tensor{MEMORY_GPU,
                         activation_in_type,
                         {h_token_num, hidden_units_},
                         layernorm_type_ == LayerNormType::pre_layernorm ? normed_self_attn_output_ :
                                                                           after_adapter_attn_output_}}});
            TensorMap ffn_output_tensors;
            if (!use_moe) {
                ffn_output_tensors.insert(
                    "ffn_output",
                    Tensor{MEMORY_GPU, activation_out_type, {h_token_num, hidden_units_}, ffn_output_ptr});
            } else {
                ffn_input_tensors.insert("moe_k", Tensor{MEMORY_CPU, TYPE_UINT64, {1}, &moe_k_});

                ffn_output_tensors.insert("ffn_output",
                                          Tensor{MEMORY_GPU,
                                                 activation_out_type,
                                                 {moe_k_ * h_token_num, hidden_units_},
                                                 has_adapters_ ? adapter_fc2_result_ : fc2_result_});
                ffn_output_tensors.insert(
                    "expert_scales", Tensor{MEMORY_GPU, activation_out_type, {h_token_num, moe_k_}, expert_scales_});
                ffn_output_tensors.insert(
                    "expanded_source_row_to_expanded_dest_row",
                    Tensor{MEMORY_GPU, TYPE_INT32, {h_token_num, moe_k_}, expanded_source_row_to_expanded_dest_row_});
                ffn_output_tensors.insert(
                    "expert_for_source_row",
                    Tensor{MEMORY_GPU, TYPE_INT32, {h_token_num, moe_k_}, expert_for_source_row_});
            }

            ffn_layer_->resetInterSize(inter_size_);
            ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &layer_weight->ffn_weights);

            // the adapter after ffn (only pre layernorm currently)
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
                                                                      h_token_num,
                                                                      hidden_units_,
                                                                      0,
                                                                      nullptr,
                                                                      nullptr,
                                                                      stream_);

                    ffn_input_tensors.insert(
                        "ffn_input", Tensor{MEMORY_GPU, data_type, {h_token_num, hidden_units_}, ffn_output_ptr});
                    ffn_output_tensors.insert(
                        "ffn_output", Tensor{MEMORY_GPU, data_type, {h_token_num, hidden_units_}, decoder_output});
                } else {
                    invokeGenericActivation<IdentityActivation, T, T>(adapter_fc2_result_,
                                                                      layer_weight->ffn_weights.output_weight.bias,
                                                                      nullptr,
                                                                      nullptr,
                                                                      nullptr,
                                                                      nullptr,
                                                                      moe_k_ * h_token_num,
                                                                      hidden_units_,
                                                                      0,
                                                                      nullptr,
                                                                      nullptr,
                                                                      stream_);

                    ffn_input_tensors.insert(
                        "ffn_input",
                        Tensor{MEMORY_GPU, data_type, {moe_k_ * h_token_num, hidden_units_}, adapter_fc2_result_});
                    ffn_output_tensors.insert(
                        "ffn_output",
                        Tensor{MEMORY_GPU, data_type, {moe_k_ * h_token_num, hidden_units_}, fc2_result_});
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
                                          h_token_num,
                                          hidden_units_,
                                          stream_);
                } else if (layernorm_type_ == LayerNormType::post_layernorm) {
                    // TODO tianxin
                    // invokeAddBiasResidualLayerNorm(decoder_output,
                    //                                after_adapter_attn_output_,
                    //                                has_adapters_ ?
                    //                                    layer_weight->after_ffn_adapter_weights.output_weight.bias :
                    //                                    layer_weight->ffn_weights.output_weight.bias,
                    //                                layer_weight->self_attn_layernorm_weights.gamma,
                    //                                layer_weight->self_attn_layernorm_weights.beta,
                    //                                layernorm_eps_,
                    //                                h_token_num,
                    //                                hidden_units_,
                    //                                stream_);
                }
            } else {
                // TODO ...
            }
            sync_check_cuda_error();

            if ((l == num_layer_ - 1) && is_unpadded_mha) {
                T *base_ptr =
                    use_shared_contexts ? compact_decoder_features_ : output_tensors->at("decoder_output").getPtr<T>();
                invokeRebuildPadding(base_ptr + ite * local_batch_size * seq_len * hidden_units_,
                                     decoder_layer_output_,
                                     padding_offset_,
                                     h_token_num,
                                     head_num_ * size_per_head_,
                                     stream_);
            }
        }
    }

    if (use_shared_contexts) {
        invokeUnCompactOutputs(output_tensors->at("decoder_output").getPtr<T>(),
                               compact_decoder_features_,
                               input_tensors->at("batch_to_compact_idx").getPtr<int>(),
                               request_batch_size, // batch
                               seq_len * hidden_units_,
                               stream_);
    }

    // TODO(bhsueh) We could optimize this point by only computing the last token for the last layer
    invokeLookupHiddenStateOfLastToken(output_tensors->at("last_token_hidden_units").getPtr<T>(),
                                       output_tensors->at("decoder_output").getPtr<T>(),
                                       input_tensors->at("input_lengths").getPtr<int>(),
                                       seq_len,
                                       request_batch_size,
                                       hidden_units_,
                                       stream_);
    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    QK_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
bool GptContextDecoder<T>::isValidLayerParallelId(uint l) {
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f));
    return l < num_layer_ && (l >= local_num_layer)
           && (l < local_num_layer * 1);
}

template <typename T>
bool GptContextDecoder<T>::isFirstLayerParallelId(uint l) {
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f));
    return l < num_layer_ && (l == local_num_layer);
}

template <typename T>
bool GptContextDecoder<T>::isLastLayerParallelId(uint l) {
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f));
    return l < num_layer_ && (l == local_num_layer - 1);
}

template <typename T>
int GptContextDecoder<T>::getFirstLayerParallelId() {
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f));
    return local_num_layer;
}

template class GptContextDecoder<float>;
template class GptContextDecoder<half>;

} // namespace space_llm
