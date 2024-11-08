#pragma once

#include "layers/sampling_layers/BaseSamplingLayer.h"
#include "utils/memory_utils.h"

namespace space_llm {

template <typename T>
class TopKSamplingLayer : public BaseSamplingLayer<T> {
private:
    void freeBuffer() override;
    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size, Tensor top_k, Tensor top_p) override;

    uint runtime_max_top_k_ = 1;
    uint *runtime_top_k_buf_ = nullptr;
    float *runtime_top_p_buf_ = nullptr;

    using BaseSamplingLayer<T>::vocab_size_;
    using BaseSamplingLayer<T>::vocab_size_padded_;

    using BaseSamplingLayer<T>::sampling_workspace_size_;
    using BaseSamplingLayer<T>::sampling_workspace_;
    using BaseSamplingLayer<T>::skip_decode_buf_;

    using BaseSamplingLayer<T>::stream_;
    using BaseSamplingLayer<T>::allocator_;
    using BaseSamplingLayer<T>::is_allocate_buffer_;

public:
    TopKSamplingLayer(size_t max_batch_size,
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
                      bool is_free_buffer_after_forward);
    TopKSamplingLayer(TopKSamplingLayer<T> const &top_k_sampling_layer);

    ~TopKSamplingLayer();

    void setup(const size_t batch_size, const size_t beam_width, TensorMap *runtime_args) override;
};

} // namespace space_llm
