#pragma once

#include <assert.h>

#include "utils/tensor.h"
#include "utils/allocator.h"
#include "utils/cublasMMWrapper.h"

namespace space_llm {

class BaseLayer {
public:
    BaseLayer(cudaStream_t stream,
              cublasMMWrapper *cublas_wrapper,
              IAllocator *allocator,
              bool is_free_buffer_after_forward,
              cudaDeviceProp *cuda_device_prop = nullptr,
              bool sparse = false) :
        stream_(stream),
        cublas_wrapper_(cublas_wrapper),
        allocator_(allocator),
        cuda_device_prop_(cuda_device_prop),
        is_free_buffer_after_forward_(is_free_buffer_after_forward),
        sparse_(sparse) {
    }

    virtual ~BaseLayer() = default;

    virtual cudaStream_t getStream() {
        return stream_;
    }

    virtual void setStream(cudaStream_t stream) {
        stream_ = stream;
    }

protected:
    virtual void allocateBuffer() = 0;
    virtual void freeBuffer() = 0;

    // device environments
    cudaStream_t stream_;
    cublasMMWrapper *cublas_wrapper_;
    IAllocator *allocator_;
    cudaDeviceProp *cuda_device_prop_ = nullptr;

    bool is_free_buffer_after_forward_;
    bool is_allocate_buffer_ = false;
    bool sparse_;
};

} // namespace space_llm
