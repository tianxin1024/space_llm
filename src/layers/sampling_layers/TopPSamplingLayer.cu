#include "layers/sampling_layers/TopPSamplingLayer.h"

namespace space_llm {

template <typename T>
TopPSamplingLayer<T>::TopPSamplingLayer(size_t max_batch_size,
                                        size_t vocab_size,
                                        size_t vocab_size_padded,
                                        int end_id,
                                        float top_p,
                                        unsigned long long random_seed,
                                        float temperature,
                                        float len_penalty,
                                        float repetition_penalty,
                                        cudaStream_t stream,
                                        cublasMMWrapper *cublas_wrapper,
                                        IAllocator *allocator,
                                        bool is_free_buffer_after_forward,
                                        cudaDeviceProp *cuda_device_prop) :
    BaseSamplingLayer<T>(max_batch_size,
                         vocab_size,
                         vocab_size_padded,
                         end_id,
                         0,
                         top_p,
                         random_seed,
                         temperature,
                         len_penalty,
                         repetition_penalty,
                         stream,
                         cublas_wrapper,
                         allocator,
                         is_free_buffer_after_forward,
                         cuda_device_prop) {
}

template <typename T>
TopPSamplingLayer<T>::TopPSamplingLayer(TopPSamplingLayer<T> const &top_p_sampling_layer) :
    BaseSamplingLayer<T>(top_p_sampling_layer) {
}

template <typename T>
TopPSamplingLayer<T>::~TopPSamplingLayer() {
    freeBuffer();
}

template <typename T>
void TopPSamplingLayer<T>::allocateBuffer() {
    QK_CHECK(false);
}

template <typename T>
void TopPSamplingLayer<T>::allocateBuffer(size_t batch_size, Tensor top_k, Tensor top_p) {
    QK_LOG_DEBUG(__PRETTY_FUNCTION__);
    BaseSamplingLayer<T>::allocateBuffer(batch_size, top_k, top_p);
    invokeTopPSampling<T>(nullptr, // workspace
                          sampling_workspace_size_,
                          cub_temp_storage_size_,
                          nullptr, // output_ids
                          nullptr, // sequence_length
                          nullptr, // finished_buffer
                          nullptr, // cum_log_probs
                          nullptr, // output_log_probs
                          nullptr, // log_probs
                          topp_id_vals_buf_,
                          topp_offset_buf_,
                          begin_topp_offset_buf_,
                          curandstate_buf_,
                          batch_size,
                          vocab_size_padded_,
                          nullptr,
                          top_p.size() > 0 ? top_p.max<float>() : 0.0f,
                          stream_,
                          cuda_device_prop_,
                          skip_decode_buf_);
    sampling_workspace_ = allocator_->reMalloc(sampling_workspace_, sampling_workspace_size_, true);
    runtime_top_k_buf_ = reinterpret_cast<uint *>(allocator_->reMalloc(runtime_top_k_buf_, sizeof(uint) * batch_size, false));
    runtime_top_p_buf_ = reinterpret_cast<float *>(allocator_->reMalloc(runtime_top_p_buf_, sizeof(float) * batch_size, false));
    initial_top_p_buf_ = reinterpret_cast<float *>(allocator_->reMalloc(initial_top_p_buf_, sizeof(float) * batch_size, false));
    top_p_decay_buf_ = reinterpret_cast<float *>(allocator_->reMalloc(top_p_decay_buf_, sizeof(float) * batch_size, false));
    top_p_min_buf_ = reinterpret_cast<float *>(allocator_->reMalloc(top_p_min_buf_, sizeof(float) * batch_size, false));
    top_p_reset_ids_buf_ = reinterpret_cast<int32_t *>(allocator_->reMalloc(top_p_reset_ids_buf_, sizeof(int32_t) * batch_size, false));
    topp_id_vals_buf_ = reinterpret_cast<int *>(allocator_->reMalloc(topp_id_vals_buf_, sizeof(int) * batch_size * vocab_size_padded_, false));
    topp_offset_buf_ = reinterpret_cast<int *>(allocator_->reMalloc(topp_offset_buf_, sizeof(int) * (batch_size + 1), false));
    begin_topp_offset_buf_ = reinterpret_cast<int *>(allocator_->reMalloc(begin_topp_offset_buf_, sizeof(int) * (batch_size + 1), false));
    is_allocate_buffer_ = true;
}

template <typename T>
void TopPSamplingLayer<T>::freeBuffer() {
    QK_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void **)(&sampling_workspace_));
        allocator_->free((void **)(&topp_id_vals_buf_));
        allocator_->free((void **)(&topp_offset_buf_));
        allocator_->free((void **)(&begin_topp_offset_buf_));
        allocator_->free((void **)(&runtime_top_k_buf_));
        allocator_->free((void **)(&runtime_top_p_buf_));
        allocator_->free((void **)(&initial_top_p_buf_));
        allocator_->free((void **)(&top_p_decay_buf_));
        allocator_->free((void **)(&top_p_min_buf_));
        allocator_->free((void **)(&top_p_reset_ids_buf_));
    }
    BaseSamplingLayer<T>::freeBuffer();
    is_allocate_buffer_ = false;
}

template <typename T>
void TopPSamplingLayer<T>::setup(const size_t batch_size, const size_t beam_width, TensorMap *runtime_args) {
    /**
    * @brief Set up the sampling layer for given runtime arguments.

    * runtime_args:
    *   \param  runtime_top_k [1] or [batch_size] on cpu, optional.
    *   \param  runtime_top_p [1] or [batch_size] on cpu, optional
    *   \param  temperature [1] or [batch_size] on cpu, optional
    *   \param  repetition_penalty [1] or [batch_size] on cpu, optional
    *   \param  top_p_decay [batch_size] on gpu, float, optional
    *   \param  top_p_min [batch_size] on gpu, float, optional
    *   \param  top_p_reset_ids [batch_size] on gpu, uint32, optional
    **/

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    BaseSamplingLayer<T>::setup(batch_size, beam_width, runtime_args);
    const Tensor runtime_top_p = runtime_args->isExist("runtime_top_p") ? runtime_args->at("runtime_top_p") : Tensor();
    const size_t runtime_top_p_size = runtime_top_p.size();
    if (runtime_top_p_size == 0) {
        std::fill_n(skip_decode_, batch_size, true);
        return;
    }

    uint tmp_top_k = 0;
    const Tensor runtime_top_k = runtime_args->isExist("runtime_top_k") ?
                                     runtime_args->at("runtime_top_k") :
                                     Tensor(MEMORY_CPU, TYPE_UINT32, {1}, &tmp_top_k);
    const size_t runtime_top_k_size = runtime_top_k.size();
    // TODO ...
}

} // namespace space_llm
