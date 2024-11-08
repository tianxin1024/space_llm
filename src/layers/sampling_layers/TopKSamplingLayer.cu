#include "layers/sampling_layers/TopKSamplingLayer.h"
#include "kernels/sampling_topk_kernels.h"

namespace space_llm {

template <typename T>
TopKSamplingLayer<T>::TopKSamplingLayer(size_t max_batch_size,
                                        size_t vocab_size,
                                        size_t vocab_size_padded,
                                        int end_id,
                                        size_t top_k,
                                        unsigned long long random_seed,
                                        float temperature,
                                        float len_penalty,
                                        float repetition_penalty,
                                        cudaStream_t stream,
                                        cublasMMWrapper *cublas_wrapper,
                                        IAllocator *allocator,
                                        bool is_free_buffer_after_forward) :
    BaseSamplingLayer<T>(max_batch_size,
                         vocab_size,
                         vocab_size_padded,
                         end_id,
                         top_k,
                         0.0f,
                         random_seed,
                         temperature,
                         len_penalty,
                         repetition_penalty,
                         stream,
                         cublas_wrapper,
                         allocator,
                         is_free_buffer_after_forward,
                         nullptr) {
}

template <typename T>
TopKSamplingLayer<T>::TopKSamplingLayer(TopKSamplingLayer<T> const &top_k_sampling_layer) :
    BaseSamplingLayer<T>(top_k_sampling_layer) {
}

template <typename T>
TopKSamplingLayer<T>::~TopKSamplingLayer() {
    QK_LOG_DEBUG(__PRETTY_FUNCTION__);
    freeBuffer();
}

template <typename T>
void TopKSamplingLayer<T>::allocateBuffer() {
    QK_CHECK(false);
}

template <typename T>
void TopKSamplingLayer<T>::allocateBuffer(size_t batch_size, Tensor top_k, Tensor top_p) {
    QK_LOG_DEBUG(__PRETTY_FUNCTION__);
    BaseSamplingLayer<T>::allocateBuffer(batch_size, top_k, top_p);
    uint max_top_k = top_k.size() > 0 ? top_k.max<uint>() : 1;
    if (max_top_k == 0) {
        max_top_k = 1;
    }
    invokeTopKSampling<T>(nullptr,
                          sampling_workspace_size_,
                          nullptr,
                          nullptr,
                          nullptr,
                          nullptr,
                          nullptr,
                          nullptr,
                          nullptr,
                          max_top_k,
                          1.0f,
                          vocab_size_padded_,
                          nullptr,
                          stream_,
                          batch_size,
                          skip_decode_buf_);
    sampling_workspace_ = allocator_->reMalloc(sampling_workspace_, sampling_workspace_size_, false);
    runtime_top_k_buf_ = reinterpret_cast<uint *>(allocator_->reMalloc(runtime_top_k_buf_, sizeof(uint) * batch_size, false));
    runtime_top_p_buf_ = reinterpret_cast<float *>(allocator_->reMalloc(runtime_top_p_buf_, sizeof(float) * batch_size, false));
    is_allocate_buffer_ = true;
}

template <typename T>
void TopKSamplingLayer<T>::freeBuffer() {
    QK_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void **)(&sampling_workspace_));
        allocator_->free((void **)(&runtime_top_k_buf_));
        allocator_->free((void **)(&runtime_top_p_buf_));
    }
    BaseSamplingLayer<T>::freeBuffer();
    is_allocate_buffer_ = false;
}

template <typename T>
void TopKSamplingLayer<T>::setup(const size_t batch_size, const size_t beam_width, TensorMap *runtime_args) {
}

template class TopKSamplingLayer<float>;
template class TopKSamplingLayer<half>;

} // namespace space_llm
