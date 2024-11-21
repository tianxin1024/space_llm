#include "layers/DynamicDecodeLayer.h"
#include "layers/sampling_layers/TopKSamplingLayer.h"
#include "layers/sampling_layers/TopPSamplingLayer.h"

namespace space_llm {

template <typename T>
void DynamicDecodeLayer<T>::initialize() {
    QK_LOG_DEBUG(__PRETTY_FUNCTION__);

    topk_decode_ = new TopKSamplingLayer<T>(0, vocab_size_, vocab_size_padded_,
                                            0,    // end_id, deprecated
                                            0,    // top_k_, deprecated
                                            0,    // random_seed_, deprecated
                                            1.0f, // temperature_, deprecated
                                            0.0f, // len_penalty_, deprecated
                                            1.0f, // repetition_penalty_, deprecated
                                            stream_, cublas_wrapper_, allocator_, false);

    topp_decode_ = new TopPSamplingLayer<T>(0, vocab_size_, vocab_size_padded_,
                                            0,    // end_id, deprecated
                                            0.0f, // top_p_, deprecated
                                            0,    // random_seed_, deprecated
                                            1.0f, // temperature_, deprecated
                                            0.0f, // len_penalty_, deprecated
                                            1.0f, // repetition_penalty_, deprecated
                                            stream_,
                                            cublas_wrapper_,
                                            allocator_,
                                            false,
                                            cuda_device_prop_);

    // TODO tianxin ...
}
template <typename T>
DynamicDecodeLayer<T>::DynamicDecodeLayer(size_t vocab_size,
                                          size_t vocab_size_padded,
                                          int end_id,
                                          cudaStream_t stream,
                                          cublasMMWrapper *cublas_wrapper,
                                          IAllocator *allocator,
                                          bool is_free_buffer_after_forward,
                                          cudaDeviceProp *cuda_device_prop) :
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    vocab_size_(vocab_size),
    vocab_size_padded_(vocab_size_padded),
    cuda_device_prop_(cuda_device_prop) {
    QK_LOG_DEBUG(__PRETTY_FUNCTION__);
    initialize();
}

template <typename T>
void DynamicDecodeLayer<T>::allocateBuffer() {
    QK_LOG_DEBUG(__PRETTY_FUNCTION__);
    h_pinned_finished_sum_ = (int *)allocator_->reMalloc(h_pinned_finished_sum_, sizeof(int), true, true);
    return;
}

template <typename T>
void DynamicDecodeLayer<T>::freeBuffer() {
    QK_LOG_DEBUG(__PRETTY_FUNCTION__);
    allocator_->free((void **)(&h_pinned_finished_sum_), true);
    return;
}

template <typename T>
DynamicDecodeLayer<T>::~DynamicDecodeLayer() {
    QK_LOG_DEBUG(__PRETTY_FUNCTION__);
    delete online_beamsearch_decode_;
    delete beamsearch_decode_;
    delete topk_decode_;
    delete topp_decode_;
    freeBuffer();
    return;
}

template <typename T>
void DynamicDecodeLayer<T>::setup(const size_t batch_size,
                                  const size_t beam_width,
                                  TensorMap *runtime_args) {
    /**
   * @brief Set up the dynamic decode layer for given input runtime arguments.
   *
   * runtime_args:
   *   \param  runtime_top_k [1] or [batch_size] on cpu, optional.
   *   \param  runtime_top_p [1] or [batch_size] on cpu, optional
   *   \param  beam_search_diversity_rate [1] or [batch_size] on cpu, optional
   *   \param  temperature [1] or [batch_size] on cpu, optional
   *   \param  len_penalty [1] or [batch_size] on cpu, optional
   *   \param  repetition_penalty [1] or [batch_size] on cpu, optional
   *   \param  presence_penalty [1] or [batch_size] on cpu, optional, float
   *   \param  min_length [1] or [batch_size], optional
   *   \param  top_p_decay [batch_size] on gpu, float, optional
   *   \param  top_p_min [batch_size] on gpu, float, optional
   *   \param  top_p_reset_ids [batch_size] on gpu, uint32, optional
   */

    QK_LOG_DEBUG(__PRETTY_FUNCTION__);
    has_diff_runtime_args_ = hasDiffRuntimeArgs(runtime_args);

    if (beam_width == 1) { // sampling layers
        topk_decode_->setup(batch_size, beam_width, runtime_args);
        topp_decode_->setup(batch_size, beam_width, runtime_args);
    }
}

template <typename T>
bool DynamicDecodeLayer<T>::hasDiffRuntimeArgs(TensorMap *input_tensors) {
    for (int i = 0; i < (int)runtime_arg_names_.size(); ++i) {
        if (input_tensors->isExist(runtime_arg_names_[i])) {
            auto tensor = input_tensors->at(runtime_arg_names_[i]);
            QK_CHECK(tensor.shape.size() == 1);
            for (int j = 1; j < (int)tensor.shape[0]; ++j) {
                const void *data = tensor.data;
                switch (tensor.type) {
                case TYPE_FP32:
                    if (((const float *)data)[0] != ((const float *)data)[j]) {
                        return true;
                    }
                    break;
                case TYPE_FP16:
                    if (((const half *)data)[0] != ((const half *)data)[j]) {
                        return true;
                    }
                    break;
                default:
                    QK_CHECK_WITH_INFO(false, runtime_arg_names_[i] + ": " + tensor.toString() + " is invalid.");
                    break;
                }
            }
        }
    }
    return false;
}

template <typename T>
void DynamicDecodeLayer<T>::forward(
    std::unordered_map<std::string, Tensor> *output_tensors,
    const std::unordered_map<std::string, Tensor> *input_tensors) {
    QK_LOG_DEBUG(__PRETTY_FUNCTION__);
    TensorMap input_map(*input_tensors);
    TensorMap output_map(*output_tensors);
    forward(&output_map, &input_map);
}

template <typename T>
void DynamicDecodeLayer<T>::forward(TensorMap *output_tensors,
                                    TensorMap *input_tensors) {
    /**
   * @brief
   * input_tensors:
   *   \param  logits [batch_size, beam_width, vocab_size_padded]
   *   \param  embedding_bias [vocab_size_padded], optional
   *   \param  step [1] on cpu
   *   \param  max_input_length [1] on cpu
   *   \param  input_lengths [batch_size, beam_width], optional
   *   \param  min_length [batch_size], optional
   *   \param  sequence_limit_length [batch_size]
   *   \param  ite [1] on cpu
   *   \param  local_batch_size [1] on cpu
   *   \param  stop_words_list [batch_size, 2, stop_words_length], optional
   *   \param  runtime_top_k [1] or [batch_size] on cpu, optional, uint
   *   \param  runtime_top_p [1] or [batch_size] on cpu, optional, float
   *   \param  temperature [1] or [batch_size] on cpu, optional, float
   *   \param  len_penalty [1] or [batch_size] on cpu, optional, float
   *   \param  repetition_penalty [1] or [batch_size] on cpu, optional, float
   *   \param  presence_penalty [1] or [batch_size] on cpu, optional, float
   *                Only one of repetition and presence penalties is allowed.
   *   \param  random_seed [1] or [batch_size] on cpu, optional, unsigned long
   * long int \param  bad_words_list [2, bad_words_length] or [batch_size, 2,
   * bad_words_length], optional \param  src_cache_indirection
   *                [local_batch_size, beam_width, max_seq_len]
   *                the k/v cache index for beam search
   *   \param  is_initialize_random_table [1] on cpu, bool
   *   \param  top_p_decay [batch_size] on gpu, float, optional
   *   \param  top_p_min [batch_size] on gpu, float, optional
   *   \param  top_p_reset_ids [batch_size] on gpu, uint32, optional
   *
   * output_tensors:
   *   \param  output_ids [max_seq_len, batch_size]
   *   \param  finished [batch_size * beam_width], optional
   *   \param  should_stop [1] on cpu
   *   \param  cum_log_probs [batch_size * beam_width], necessary in beam search
   *   \param  parent_ids [max_seq_len, batch_size * beam_width]
   *   \param  sequence_length [batch_size * beam_width], optional
   *   \param  output_log_probs [request_ouptut_length, batch_size *
   * beam_width], must be float*, optional \param  tgt_cache_indirection
   *                [local_batch_size, beam_width, max_seq_len]
   *                the k/v cache index for beam search
   *   \param  beam_hyps: [1] on cpu, a special structure which maintains some
   * pointers of beam search
   *
   */

    QK_LOG_DEBUG(__PRETTY_FUNCTION__);
    const int ite = (int)input_tensors->at("ite").getVal<uint>();
    const int step = input_tensors->at("step").getVal<int>();
    QK_CHECK(input_tensors->at("logits").shape.size() == 3);

    const size_t batch_size = input_tensors->at("logits").shape[0];
    const size_t beam_width = input_tensors->at("logits").shape[1];
    const size_t local_batch_size =
        (size_t)input_tensors->at("local_batch_size").getVal<int>();

    if (input_tensors->isExist("bad_words_list")) {
        const auto &bad_words = input_tensors->at("bad_words_list");
        const int *bad_words_ptr = bad_words.getPtr<const int>();
        QK_CHECK_WITH_INFO(bad_words.shape.size() == 2 || bad_words.shape.size() == 3,
                           "Bad words dimension must be 2 or 3.");

        const bool is_matrix = bad_words.shape.size() == 2;
        if (bad_words.shape.size() == 3) {
            QK_CHECK_WITH_INFO(bad_words.shape[0] == batch_size,
                               fmtstr("Shape of dim 0 of bad words is invalid. It "
                                      "must be equal to batch size."
                                      " However, it is %d and the batch size is %d.",
                                      bad_words.shape[0], batch_size));
        }

        const bool shared_bad_words = is_matrix || bad_words.shape[0] == 1;
        const size_t bad_words_len = bad_words.shape[is_matrix ? 1 : 2];
        // Add check on batch size of bad words
        const int id_offset = ite * local_batch_size;
        const int decode_vocab_size_units_offset = id_offset * vocab_size_padded_;

        invokeBanBadWords(
            (T *)input_tensors->at("logits").getPtrWithOffset(
                decode_vocab_size_units_offset),
            output_tensors->at("output_ids").getPtr<const int>(),
            beam_width > 1 ? output_tensors->at("parent_ids").getPtr<const int>() : nullptr,
            batch_size, local_batch_size, beam_width,
            shared_bad_words ? bad_words_ptr : bad_words.getPtrWithOffset<const int>(ite * local_batch_size * 2 * bad_words_len),
            shared_bad_words, bad_words_len, id_offset, vocab_size_padded_, step,
            stream_);
    }
}

template class DynamicDecodeLayer<float>;
template class DynamicDecodeLayer<half>;

} // namespace space_llm
