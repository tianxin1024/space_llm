#pragma once

#include <curand_kernel.h>
#include "layers/DynamicDecodeBaseLayer.h"
#include "kernels/penalty_types.h"

namespace space_llm {

template <typename T>
class BaseSamplingLayer : public DynamicDecodeBaseLayer {
protected:
    size_t vocab_size_;
    size_t vocab_size_padded_;

    size_t sampling_workspace_size_;
    void *sampling_workspace_ = nullptr;
    curandState_t *curandstate_buf_ = nullptr;
    unsigned long long *random_seeds_buf_ = nullptr;

    float *temperature_buf_ = nullptr;
    float *repetition_penalty_buf_ = nullptr;
    int *min_lengths_buf_ = nullptr;
    bool *skip_decode_buf_ = nullptr;
    T *runtime_logits_buf_ = nullptr;

    float *temperature_ = nullptr;
    float *repetition_penalty_ = nullptr;
    int *min_lengths_ = nullptr;
    bool *skip_decode_ = nullptr;
    bool skip_any_ = false;

    RepetitionPenaltyType repetition_penalty_type_ = RepetitionPenaltyType::None;

    virtual void runSampling(TensorMap *output_tensors, TensorMap *input_tensors) = 0;

    virtual void freeBuffer();
    virtual void allocateBuffer() = 0;
    virtual void allocateBuffer(size_t batch_size, Tensor top_k, Tensor top_p);

public:
    BaseSamplingLayer(size_t max_batch_size,
                      size_t vocab_size,
                      size_t vocab_size_padded,
                      int end_id,
                      size_t top_k,
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

    BaseSamplingLayer(BaseSamplingLayer const &sampling_layer);
    ~BaseSamplingLayer();

    void setup(const size_t batch_size, const size_t beam_width, TensorMap *runtime_args) override;
    void forward(std::vector<Tensor> *output_tensors, const std::vector<Tensor> *input_tensors) override;
    void forward(std::unordered_map<std::string, Tensor> *output_tensors,
                 const std::unordered_map<std::string, Tensor> *input_tensors) override;
    void forward(TensorMap *output_tensors, TensorMap *input_tensors) override;
};

} // namespace space_llm
