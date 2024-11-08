#pragma once

#include "layers/sampling_layers/BaseSamplingLayer.h"

namespace space_llm {

template <typename T>
class TopPSamplingLayer : public BaseSamplingLayer<T> {
private:
    // void runSampling(TensorMap *output_tensors, TensorMap *input_tensors) override;
    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size, Tensor top_k, Tensor top_p) override;
    void freeBuffer() override;

    uint *runtime_top_k_buf_ = nullptr;
    float *runtime_top_p_buf_ = nullptr;
    float runtime_max_top_p_;
    float *initial_top_p_buf_ = nullptr;
    float *top_p_decay_buf_ = nullptr;
    float *top_p_min_buf_ = nullptr;
    int32_t *top_p_reset_ids_buf_ = nullptr;

    int *topp_id_vals_buf_ = nullptr;
    int *topp_offset_buf_ = nullptr;
    int *begin_topp_offset_buf_ = nullptr;
    size_t cub_temp_storage_size_;

    using BaseSamplingLayer<T>::vocab_size_;
    using BaseSamplingLayer<T>::vocab_size_padded_;

    using BaseSamplingLayer<T>::sampling_workspace_size_;
    using BaseSamplingLayer<T>::sampling_workspace_;
    using BaseSamplingLayer<T>::curandstate_buf_;
    using BaseSamplingLayer<T>::random_seeds_buf_;
    using BaseSamplingLayer<T>::skip_decode_buf_;
    using BaseSamplingLayer<T>::skip_decode_;

    using BaseSamplingLayer<T>::stream_;
    using BaseSamplingLayer<T>::allocator_;
    using BaseSamplingLayer<T>::is_allocate_buffer_;
    using BaseSamplingLayer<T>::cuda_device_prop_;

protected:
public:
    TopPSamplingLayer(size_t max_batch_size,
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
                      cudaDeviceProp *cuda_device_prop);
    TopPSamplingLayer(TopPSamplingLayer<T> const &top_p_sampling_layer);
    ~TopPSamplingLayer();

    void setup(const size_t batch_size, const size_t beam_width, TensorMap *runtime_args) override;
};

} // namespace space_llm
