#include "layers/ffnLayer.h"

namespace space_llm {

template <typename T>
ffnLayer<T>::ffnLayer(size_t max_batch_size,
                      size_t max_seq_len,
                      size_t head_num,
                      size_t size_per_head,
                      size_t expert_num,
                      size_t inter_size,
                      cudaStream_t stream,
                      cublasMMWrapper *cublas_wrapper,
                      IAllocator *allocator,
                      bool is_free_buffer_after_forward,
                      bool sparse,
                      int int8_mode,
                      bool use_gated_activation) :
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    max_token_num_(max_batch_size * max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    expert_num_(expert_num),
    hidden_units_(head_num * size_per_head),
    max_inter_size_(inter_size),
    inter_size_(inter_size),
    int8_mode_(int8_mode),
    use_gated_activation_(use_gated_activation) {
    QK_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (int8_mode_ == 0) {
    } else if (int8_mode_ == 1) {
        QK_CHECK_WITH_INFO(!(std::is_same<T, float>::value), "Weight only quant not supported for fp32.");
    }
}

template <typename T>
ffnLayer<T>::ffnLayer(ffnLayer<T> const &ffn_layer) :
    BaseLayer(ffn_layer.stream_,
              ffn_layer.cublas_wrapper_,
              ffn_layer.allocator_,
              ffn_layer.is_free_buffer_after_forward_,
              ffn_layer.cuda_device_prop_,
              ffn_layer.sparse_),
    max_token_num_(ffn_layer.max_token_num_),
    head_num_(ffn_layer.head_num_),
    size_per_head_(ffn_layer.size_per_head_),
    expert_num_(ffn_layer.expert_num_),
    hidden_units_(ffn_layer.hidden_units_),
    max_inter_size_(ffn_layer.max_inter_size_),
    inter_size_(ffn_layer.inter_size_),
    int8_mode_(ffn_layer.int8_mode_),
    use_gated_activation_(ffn_layer.use_gated_activation_) {
    QK_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template <typename T>
ffnLayer<T>::~ffnLayer() {
    QK_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template <typename T>
void ffnLayer<T>::forward(std::vector<Tensor> *output_tensors,
                          const std::vector<Tensor> *input_tensors,
                          const ffnWeight<T> *ffn_weights) {
    TensorMap input_tensor({{"ffn_input", input_tensors->at(0)}});
    TensorMap output_tensor({{"ffn_output", output_tensors->at(0)}});
    forward(&output_tensor, &input_tensor, ffn_weights);
}

template <typename T>
void ffnLayer<T>::forward(TensorMap *output_tensors, TensorMap *input_tensors, const ffnWeight<T> *ffn_weights) {
    // input tensors:
    //      ffn_input [token_num, hidden_dimension],
    //      ia3_tasks [batch_size] (optional)
    //      moe_k     [1], uint64 (optional)
    //      padding_offset [token_num] (optional)
    //      seq_len [1], int32, (optional), only used for ia3

    // output tensors:
    //      ffn_output [token_num, hidden_dimension] or [moe_k * token_num, hidden_dimension] if use_moe
    //      expert_scales [token_num, moe_k] (optional)
    //      expanded_source_row_to_expanded_dest_row [token_num, moe_k] (optional)
    //      expert_for_source_row [token_num, moe_k] (optional)

    QK_LOG_DEBUG(__PRETTY_FUNCTION__);
    QK_CHECK(input_tensors->size() >= 1 && input_tensors->size() <= 5);
    QK_CHECK(output_tensors->size() >= 1 || output_tensors->size() <= 4);
    bool use_moe = false;
    size_t moe_k = 0;
    bool use_gated_activation = false;
    if (input_tensors->isExist("moe_k")) {
        use_moe = true;
        moe_k = input_tensors->at("moe_k").getVal<size_t>();
    }
    allocateBuffer(input_tensors->at("ffn_input").shape[0], moe_k, use_moe);

    const int m = input_tensors->at("ffn_input").shape[0];
    T *output_tensor = output_tensors->at("ffn_output").getPtr<T>();
    const T *input_tensor = input_tensors->at("ffn_input").getPtr<const T>();

    auto activation_type = getActivationType();

    const int *ia3_tasks = input_tensors->getPtr<const int>("ia3_tasks", nullptr);

    if (use_moe) {
        QK_LOG_INFO("FFN moe");
    }

    QK_LOG_INFO("FFM gemm 1");
    int m_tmp = input_tensors->at("ffn_input").shape[0];
    if (m_tmp % 8 != 0) {
        m_tmp = (m_tmp / 8 + 1) * 8;
    }
    constexpr bool use_sparse_gemm = false;
    int int8_mode_ = 0;

    if (use_sparse_gemm) {
        QK_LOG_INFO("use sparse gemm");
    } else {
        if (int8_mode_ == 1) {
            QK_LOG_INFO("int8 mode 1");
        } else if (int8_mode_ == 2) {
            QK_LOG_INFO("int8 mode 2");
        } else {
            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  inter_size_,
                                  m,
                                  hidden_units_,
                                  ffn_weights->intermediate_weight.kernel,
                                  inter_size_,
                                  input_tensor,
                                  hidden_units_,
                                  inter_buf_,
                                  inter_size_);
        }
    }

    if (int8_mode_ != 1 || ia3_tasks != nullptr || use_gated_activation) {
        genericActivation(m,
                          ffn_weights->intermediate_weight.bias,
                          use_gated_activation ? ffn_weights->intermediate_weight2.bias : nullptr,
                          input_tensors->at("ia3_tasks", {MEMORY_GPU, TYPE_INT32, {}, nullptr}).getPtr<const int>(),
                          ffn_weights->ia3_weight.kernel,
                          int8_mode_ == 2 ? ffn_weights->intermediate_weight.scale_out : (float *)nullptr,
                          int8_mode_ == 2 ? ffn_weights->output_weight.scale : (float *)nullptr,
                          input_tensors->getPtr<int>("padding_offset", nullptr),
                          input_tensors->getVal<int>("seq_len", 1));
    }
    sync_check_cuda_error();

    if (use_sparse_gemm) {
        QK_LOG_INFO("FFM gemm 2: use sparse gemm");
    } else {
        if (int8_mode_ == 1) {
            QK_LOG_INFO("int8 mode 1");
        } else if (int8_mode_ == 2) {
            QK_LOG_INFO("int8 mode 2");
        } else {
            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  hidden_units_,
                                  m,
                                  inter_size_,
                                  ffn_weights->output_weight.kernel,
                                  hidden_units_,
                                  inter_buf_,
                                  inter_size_,
                                  output_tensor,
                                  hidden_units_);
        }
    }
    sync_check_cuda_error();

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

} // namespace space_llm
