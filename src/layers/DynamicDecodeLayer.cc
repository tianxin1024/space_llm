#include "layers/DynamicDecodeLayer.h"
#include "layers/sampling_layers/TopKSamplingLayer.h"

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
    // topp_decode_ = new TopPSamplingLayer<T>(0, vocab_size_, vocab_size_padded_,
    //                                         0,    // end_id, deprecated
    //                                         0.0f, // top_p_, deprecated
    //                                         0,    // random_seed_, deprecated
    //                                         1.0f, // temperature_, deprecated
    //                                         0.0f, // len_penalty_, deprecated
    //                                         1.0f, // repetition_penalty_, deprecated
    //                                         stream_,
    //                                         cublas_wrapper_,
    //                                         allocator_,
    //                                         false,
    //                                         cuda_device_prop_);

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

template class DynamicDecodeLayer<float>;
template class DynamicDecodeLayer<half>;

} // namespace space_llm
