#pragma once

#include <string>
#include <unordered_map>

#include "layers/BaseLayer.h"

namespace space_llm {

class DynamicDecodeBaseLayer : public BaseLayer {
protected:
    virtual void allocateBuffer() = 0;
    virtual void freeBuffer() = 0;

public:
    DynamicDecodeBaseLayer(cudaStream_t stream,
                           cublasMMWrapper *cublas_wrapper,
                           IAllocator *allocator,
                           bool is_free_buffer_after_forward,
                           cudaDeviceProp *cuda_device_prop) :
        BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop){};

    ~DynamicDecodeBaseLayer() = default;

    DynamicDecodeBaseLayer(DynamicDecodeBaseLayer const &dynamic_decode_layer) :
        BaseLayer(dynamic_decode_layer){};

    virtual void setup(const size_t batch_size, const size_t beam_width, TensorMap *runtime_args) = 0;
    virtual void forward(std::vector<Tensor> *output_tensors, const std::vector<Tensor> *input_tensors) = 0;
    virtual void forward(std::unordered_map<std::string, Tensor> *output_tensors,
                         const std::unordered_map<std::string, Tensor> *input_tensors) = 0;
    virtual void forward(TensorMap *output_tensors, TensorMap *input_tensors) = 0;
};

} // namespace space_llm
