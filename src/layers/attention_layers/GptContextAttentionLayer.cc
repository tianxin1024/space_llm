#include "layers/attention_layers/GptContextAttentionLayer.h"
#include "kernels/unfused_attention_kernels.h"

namespace space_llm {

template <typename T>
void GptContextAttentionLayer<T>::forward(TensorMap *output_tensors,
                                          TensorMap *input_tensors,
                                          const AttentionWeight<T> *attention_weights) {
    // input_tensors:
    //      input_query [token_num, hidden_dimension]
    //      attention_mask [batch_size, 1, seq_len, seq_len + max_prompt_length]
    //      attention_type [1]
    //      is_final_layer [1], bool on cpu
    //      layer_id [1], int on cpu
    //      padding_offset, int, [token_num] (optional)
    //      cu_seqlens, int, [batch_size] (optional)
    //      d_prefix_prompt_batch [global_batch_size], (optional)
    //          each element contains ptr with buffer shape[2, local_head_num_, prompt_length, size_per_head]
    //      d_prefix_prompt_lengths [batch_size], int (optional)
    //      linear_bias_slopes [head_num] (optional)

    // output_tensors:
    //      hidden_features [token_num, hidden_dimension]
    //      key_cache [batch, local_head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [batch, local_head_num, max_seq_len, size_per_head]
    QK_CHECK(output_tensors->at("key_cache").shape.size() == 5);
    QK_CHECK(output_tensors->at("value_cache").shape.size() == 4
             || output_tensors->at("value_cache").shape.size() == 3);
    const int request_batch_size = input_tensors->at("attention_mask").shape[0];
    const int request_seq_len = input_tensors->at("attention_mask").shape[2];
    const int max_prompt_length =
        input_tensors->at("attention_mask").shape[3] - input_tensors->at("attention_mask").shape[2];
    const int layer_id = input_tensors->getVal<int>("layer_id");
    const T **d_prefix_prompt_batch = input_tensors->getPtr<const T *>("d_prefix_prompt_batch", nullptr);
    const int *d_prefix_prompt_lengths = input_tensors->getPtr<int>("d_prefix_prompt_lengths", nullptr);
    const int *padding_offset = input_tensors->getPtr<int>("padding_offset", nullptr);
    int *cu_seqlens = input_tensors->getPtr<int>("cu_seqlens", nullptr);
    T *linear_bias_slopes = input_tensors->getPtr<T>("linear_bias_slopes", nullptr);
    /* float*     attention_query_dynamic_scale = input_tensors->getPtr<float>("attention_query_dynamic_scale",
     * nullptr); */

    T *attention_out = output_tensors->at("hidden_features").getPtr<T>();
    T *attention_input = input_tensors->at("input_query").getPtr<T>();
    T *attention_mask = input_tensors->at("attention_mask").getPtr<T>();

    const AttentionType attention_type = input_tensors->getVal<AttentionType>("attention_type");
    QK_CHECK_WITH_INFO(attention_type != AttentionType::FUSED_PADDED_MHA,
                       "Gpt Context FUSED_PADDED_MHA is not supported !");

    QK_LOG_INFO("attention buffer alloc");
    allocateBuffer(request_batch_size, request_seq_len + max_prompt_length, attention_type != AttentionType::FUSED_MHA);
    sync_check_cuda_error();

    const bool is_final = input_tensors->at("is_final_layer").getVal<bool>();

    const int m = input_tensors->at("input_query").shape[0];

    QK_LOG_INFO("qkv_gemm");

    constexpr bool use_sparse = false;

    if (use_sparse) {
        // sparse
    } else if (int8_mode_ == 1) {
        // TODO ...
    } else if (int8_mode_ == 2) {
        // TODO ...
    } else {
        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              3 * local_hidden_units_, // n
                              m,
                              hidden_units_, // k
                              attention_weights->query_weight.kernel,
                              3 * local_hidden_units_, // n
                              attention_input,
                              hidden_units_, // k
                              qkv_buf_,
                              3 * local_hidden_units_ /* n */);
    }

    sync_check_cuda_error();

    // IDEA: append prefix prompt key value here
    PrefixPromptBatchWeightsParam<T> param{d_prefix_prompt_batch,
                                           d_prefix_prompt_lengths,
                                           max_prompt_length,
                                           (size_t)layer_id * 2 * local_head_num_ * size_per_head_};

    if (padding_offset != nullptr) {
        // q_buf_2_, k_buf_2_ and v_buf_2_ are continuous
        cudaMemsetAsync(
            q_buf_2_, 0, request_batch_size * request_seq_len * 3 * local_hidden_units_ * sizeof(T), stream_);
    }
    invokeAddFusedQKVBiasTranspose(q_buf_2_,
                                   k_buf_2_,
                                   v_buf_2_,
                                   param, // prefix prompt
                                   qkv_buf_,
                                   attention_weights->query_weight.bias,
                                   padding_offset,
                                   request_batch_size,
                                   request_seq_len,
                                   m,
                                   local_head_num_,
                                   size_per_head_,
                                   rotary_embedding_dim_,
                                   neox_rotary_style_,
                                   attention_weights->query_weight.scale_out,
                                   int8_mode_,
                                   stream_);
    sync_check_cuda_error();

    const int max_seq_len = (int)(output_tensors->at("key_cache").shape[3]); // max output seq length
    // Use batch major
    // put k/v_buf from shape [B, H, PL + L, Dh]
    // to cache [B, H, Dh/x, PL + L, x]  and [B, H, PL + L, Dh/x, x], PL denotes prompt length
    invokeTranspose4dBatchMajor(output_tensors->getPtr<T>("key_cache"),
                                output_tensors->getPtr<T>("value_cache"),
                                k_buf_2_,
                                v_buf_2_,
                                request_batch_size,
                                max_prompt_length + request_seq_len, // max input length + prefix prompt length
                                max_seq_len,
                                size_per_head_,
                                local_head_num_,
                                stream_);
    // IDEA : after this, k_cache = (batch_size, num_heads, Dh/x, prefix_prompt_len + L, x)
    // k_cache = (batch_size, num_heads, prefix_prompt_len + L, Dh)
    sync_check_cuda_error();

    // TODO: fmha kernels doesn't support different seq lengths of q and kv
    if (attention_type == AttentionType::FUSED_MHA) {
        // TODO ...
        // dispatcher_fp16->setup_causal_masked_fmha(request_seq_len, request_batch_size);
        // dispatcher_fp16->run_causal_masked_fmha(qkv_buf_, cu_seqlens, qkv_buf_3_, true, stream_);
    }
    // NOTE: qkv buffer shape (batch_size, num_heads,L or prompt_len + L, Dh)

    if (is_final == false) {
        const cudaDataType_t gemm_data_type = getCudaDataType<T>();
        const int attention_seq_len_1 = request_seq_len;                     // q length
        const int attention_seq_len_2 = max_prompt_length + request_seq_len; // kv length
        const T qk_scale = static_cast<T>(1.0f / sqrtf(size_per_head_ * 1.0f));
        if (attention_type != AttentionType::FUSED_MHA) {
            if (is_qk_buf_float_ == true && gemm_data_type != CUDA_R_32F) {
                QK_LOG_INFO("Q*K batch gemm");
                cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    attention_seq_len_2, // n
                                                    attention_seq_len_1, // m
                                                    size_per_head_,      // k
                                                    1.0f,
                                                    k_buf_2_,
                                                    gemm_data_type,
                                                    size_per_head_,                       // k
                                                    attention_seq_len_2 * size_per_head_, // n * k
                                                    q_buf_2_,
                                                    gemm_data_type,
                                                    size_per_head_,                       // k
                                                    attention_seq_len_1 * size_per_head_, // m * k
                                                    0.0f,
                                                    qk_buf_float_,
                                                    CUDA_R_32F,
                                                    attention_seq_len_2, // n
                                                    attention_seq_len_2 * attention_seq_len_1,
                                                    request_batch_size * local_head_num_, // global batch size
                                                    CUDA_R_32F);

                sync_check_cuda_error();

                QK_LOG_INFO("softmax");
                MaskedSoftmaxParam<T, float> param;
                param.attention_score = qk_buf_;       // (batch_size, head_num, q_length, k_length)
                param.qk = qk_buf_float_;              // (batch_size, head_num, q_length, k_length)
                param.attention_mask = attention_mask; // (batch_size, q_length, k_length)
                param.batch_size = request_batch_size;
                param.q_length = attention_seq_len_1;
                param.k_length = attention_seq_len_2;
                param.num_heads = local_head_num_;
                param.qk_scale = qk_scale;
                param.linear_bias_slopes = const_cast<T *>(linear_bias_slopes); // (head_num,), optional
                invokeMaskedSoftmax(param, stream_);
                sync_check_cuda_error();
            } else {
                QK_LOG_INFO("Q*K batch gemm");
                cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    attention_seq_len_2,
                                                    attention_seq_len_1,
                                                    size_per_head_,
                                                    k_buf_2_,
                                                    size_per_head_,
                                                    attention_seq_len_2 * size_per_head_,
                                                    q_buf_2_,
                                                    size_per_head_,
                                                    attention_seq_len_1 * size_per_head_,
                                                    qk_buf_,
                                                    attention_seq_len_2,
                                                    attention_seq_len_2 * attention_seq_len_1,
                                                    request_batch_size * local_head_num_);

                QK_LOG_INFO("softmax");
                MaskedSoftmaxParam<T, T> param;
                param.attention_score = qk_buf_;       // (batch_size, head_num, q_length, k_length)
                param.qk = qk_buf_;                    // (batch_size, head_num, q_length, k_length)
                param.attention_mask = attention_mask; // (batch_size, q_length, k_length)
                param.batch_size = request_batch_size;
                param.q_length = attention_seq_len_1;
                param.k_length = attention_seq_len_2;
                param.num_heads = local_head_num_;
                param.qk_scale = qk_scale;
                param.linear_bias_slopes = const_cast<T *>(linear_bias_slopes); // (head_num,), optional
                invokeMaskedSoftmax(param, stream_);
                sync_check_cuda_error();
            }

            QK_LOG_INFO("QK*V batch gemm");

            cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                                CUBLAS_OP_N,
                                                size_per_head_,
                                                attention_seq_len_1,
                                                attention_seq_len_2,
                                                v_buf_2_,
                                                size_per_head_,
                                                attention_seq_len_2 * size_per_head_,
                                                qk_buf_,
                                                attention_seq_len_2,
                                                attention_seq_len_1 * attention_seq_len_2,
                                                qkv_buf_2_,
                                                size_per_head_,
                                                attention_seq_len_1 * size_per_head_,
                                                request_batch_size * local_head_num_);

            // transpose (batch_size, num_heads, L, Dh) to (batch_size, L, num_heads * Dh)
            if (padding_offset == nullptr) {
                invokeTransposeQKV(qkv_buf_3_,
                                   qkv_buf_2_,
                                   request_batch_size,
                                   attention_seq_len_1,
                                   local_head_num_,
                                   size_per_head_,
                                   attention_weights->attention_output_weight.scale,
                                   int8_mode_,
                                   stream_);
                sync_check_cuda_error();
            } else {
                invokeTransposeAttentionOutRemovePadding(qkv_buf_2_,
                                                         qkv_buf_3_,
                                                         m,
                                                         request_batch_size,
                                                         attention_seq_len_1,
                                                         local_head_num_,
                                                         size_per_head_,
                                                         padding_offset,
                                                         attention_weights->attention_output_weight.scale,
                                                         int8_mode_,
                                                         stream_);
            }
        }
        sync_check_cuda_error();

        QK_LOG_INFO("proj gemm");

        if (use_sparse) {
        } else {
            if (int8_mode_ == 1) {
                // TODO ...
            } else if (int8_mode_ == 2) {
                // TODO ...
            } else {
                cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                      CUBLAS_OP_N,
                                      hidden_units_,
                                      m,
                                      local_hidden_units_,
                                      attention_weights->attention_output_weight.kernel,
                                      hidden_units_,
                                      qkv_buf_3_,
                                      local_hidden_units_,
                                      attention_out,
                                      hidden_units_);
            }
        }
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template <typename T>
GptContextAttentionLayer<T>::GptContextAttentionLayer(size_t max_batch_size,
                                                      size_t max_seq_len,
                                                      size_t head_num,
                                                      size_t size_per_head,
                                                      cudaStream_t stream,
                                                      cublasMMWrapper *cublas_wrapper,
                                                      IAllocator *allocator,
                                                      bool is_free_buffer_after_forward,
                                                      bool is_qk_buf_float,
                                                      bool sparse,
                                                      int int8_mode) :
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    local_head_num_(head_num),
    local_hidden_units_(local_head_num_ * size_per_head),
    rotary_embedding_dim_(0),
    neox_rotary_style_(false),
    is_qk_buf_float_(is_qk_buf_float || int8_mode == 2),
    int8_mode_(int8_mode) {
}

template <typename T>
GptContextAttentionLayer<T>::GptContextAttentionLayer(size_t max_batch_size,
                                                      size_t max_seq_len,
                                                      size_t head_num,
                                                      size_t size_per_head,
                                                      size_t local_head_num,
                                                      cudaStream_t stream,
                                                      cublasMMWrapper *cublas_wrapper,
                                                      IAllocator *allocator,
                                                      bool is_free_buffer_after_forward,
                                                      bool is_qk_buf_float,
                                                      bool sparse,
                                                      int int8_mode) :
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    local_head_num_(local_head_num),
    local_hidden_units_(local_head_num_ * size_per_head),
    rotary_embedding_dim_(0),
    neox_rotary_style_(false),
    is_qk_buf_float_(is_qk_buf_float || int8_mode == 2),
    int8_mode_(int8_mode) {
    // dispatcher_fp16.reset(new FusedMHARunnerFP16v2(local_head_num_, size_per_head_, sm_, 1.0f));
}

template <typename T>
GptContextAttentionLayer<T>::GptContextAttentionLayer(size_t max_batch_size,
                                                      size_t max_seq_len,
                                                      size_t head_num,
                                                      size_t size_per_head,
                                                      size_t local_head_num,
                                                      size_t rotary_embedding_dim,
                                                      bool neox_rotary_style,
                                                      cudaStream_t stream,
                                                      cublasMMWrapper *cublas_wrapper,
                                                      IAllocator *allocator,
                                                      bool is_free_buffer_after_forward,
                                                      bool is_qk_buf_float,
                                                      bool sparse,
                                                      int int8_mode) :
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    local_head_num_(local_head_num),
    local_hidden_units_(local_head_num_ * size_per_head),
    rotary_embedding_dim_(rotary_embedding_dim),
    neox_rotary_style_(neox_rotary_style),
    is_qk_buf_float_(is_qk_buf_float),
    int8_mode_(int8_mode) {
    // dispatcher_fp16.reset(new FusedMHARunnerFP16v2(local_head_num_, size_per_head_, sm_, 1.0f));
}

template <typename T>
GptContextAttentionLayer<T>::GptContextAttentionLayer(GptContextAttentionLayer<T> const &attention_layer) :
    BaseAttentionLayer<T>(attention_layer.stream_,
                          attention_layer.cublas_wrapper_,
                          attention_layer.allocator_,
                          attention_layer.is_free_buffer_after_forward_,
                          attention_layer.sparse_),
    max_batch_size_(attention_layer.max_batch_size_),
    max_seq_len_(attention_layer.max_seq_len_),
    head_num_(attention_layer.head_num_),
    size_per_head_(attention_layer.size_per_head_),
    hidden_units_(attention_layer.hidden_units_),
    local_head_num_(attention_layer.local_head_num_),
    local_hidden_units_(attention_layer.local_hidden_units_),
    rotary_embedding_dim_(attention_layer.rotary_embedding_dim_),
    neox_rotary_style_(attention_layer.neox_rotary_style_),
    is_qk_buf_float_(attention_layer.is_qk_buf_float_),
    int8_mode_(attention_layer.int8_mode_) {
}

template <typename T>
GptContextAttentionLayer<T>::~GptContextAttentionLayer() {
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template <typename T>
void GptContextAttentionLayer<T>::allocateBuffer() {
    QK_CHECK(false);
}

template <typename T>
void GptContextAttentionLayer<T>::allocateBuffer(size_t batch_size, size_t seq_len, bool allocate_qk_buf) {
    // const auto type_size = int8_mode_ == 2 ? sizeof(int8_t) : sizeof(T);
    // NOTE (perkzz): use sizeof(T) here for cutlass int8 kernels.
    const auto type_size = sizeof(T);
    qkv_buf_ = (T *)allocator_->reMalloc(qkv_buf_, type_size * 3 * batch_size * seq_len * local_hidden_units_, true);
    q_buf_2_ = (T *)allocator_->reMalloc(q_buf_2_, sizeof(T) * batch_size * seq_len * 3 * local_hidden_units_, true);
    k_buf_2_ = q_buf_2_ + batch_size * seq_len * local_hidden_units_;
    v_buf_2_ = k_buf_2_ + batch_size * seq_len * local_hidden_units_;

    // save memory usage when using fmha
    if (allocate_qk_buf) {
        qk_buf_ = (T *)allocator_->reMalloc(qk_buf_, sizeof(T) * batch_size * local_head_num_ * seq_len * seq_len, true);
    } else {
        allocator_->free((void **)(&qk_buf_));
    }
    qkv_buf_2_ = (T *)allocator_->reMalloc(qkv_buf_2_, sizeof(T) * batch_size * seq_len * local_hidden_units_, true);
    qkv_buf_3_ = (T *)allocator_->reMalloc(qkv_buf_3_, type_size * batch_size * seq_len * local_hidden_units_, true);

    if (is_qk_buf_float_ == true) {
        if (allocate_qk_buf) {
            qk_buf_float_ = (float *)allocator_->reMalloc(
                qk_buf_float_, sizeof(float) * batch_size * local_head_num_ * seq_len * seq_len, true);
        } else {
            allocator_->free((void **)(&qk_buf_float_));
        }
    }

    if (int8_mode_ == 1) {
        // We use max_size for n and k since we reuse buffers for both FCs and want to allocate the max
        // possible memory that would be required by any of the individual gemms.
        // const int max_size = std::max(hidden_units_, 3 * local_hidden_units_);
        // mixed_gemm_ws_bytes_ = weight_only_int8_fc_runner_->getWorkspaceSize(batch_size * seq_len, max_size, max_size);
        // mixed_gemm_workspace_ = (char *)allocator_->reMalloc(mixed_gemm_workspace_, mixed_gemm_ws_bytes_, false);
    }

    if (int8_mode_ == 1) {
        // We use max_size for n and k since we reuse buffers for both FCs and want to allocate the max
        // possible memory that would be required by any of the individual gemms.
        // const int max_size = std::max(hidden_units_, 3 * local_hidden_units_);
        // mixed_gemm_ws_bytes_ = weight_only_int8_fc_runner_->getWorkspaceSize(batch_size * seq_len, max_size, max_size);
        // mixed_gemm_workspace_ = (char *)allocator_->reMalloc(mixed_gemm_workspace_, mixed_gemm_ws_bytes_, false);
    } else if (int8_mode_ == 2) {
        // const int max_size = std::max(hidden_units_, 3 * local_hidden_units_);
        // int8_gemm_ws_bytes_ = int8_fc_runner_->getWorkspaceSize(batch_size * seq_len, max_size, max_size);
        // int8_gemm_workspace_ = (char *)allocator_->reMalloc(int8_gemm_workspace_, int8_gemm_ws_bytes_, false);
    }
    is_allocate_buffer_ = true;
}

template <typename T>
void GptContextAttentionLayer<T>::freeBuffer() {
    if (is_allocate_buffer_) {
        allocator_->free((void **)(&qkv_buf_));
        allocator_->free((void **)(&q_buf_2_));
        allocator_->free((void **)(&qk_buf_));
        allocator_->free((void **)(&qkv_buf_2_));
        allocator_->free((void **)(&qkv_buf_3_));

        if (is_qk_buf_float_ == true) {
            allocator_->free((void **)(&qk_buf_float_));
        }

        allocator_->free((void **)(&mixed_gemm_workspace_));
        mixed_gemm_ws_bytes_ = 0;

        allocator_->free((void **)(&int8_gemm_workspace_));
        int8_gemm_ws_bytes_ = 0;

        is_allocate_buffer_ = false;
    }
}

template class GptContextAttentionLayer<float>;
template class GptContextAttentionLayer<half>;

} // namespace space_llm
