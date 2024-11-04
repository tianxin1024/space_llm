#include "layers/BaseLayer.h"

namespace space_llm {

template <typename T>
class DynamicDecodeLayer : public BaseLayer {
public:
    DynamicDecodeLayer(size_t vocab_size,
                       size_t vocab_size_padded,
                       int end_id,
                       cudaStream_t stream,
                       cublasMMWrapper *cublas_wrapper,
                       IAllocator *allocator,
                       bool is_free_buffer_after_forward,
                       cudaDeviceProp *cuda_device_prop);

    ~DynamicDecodeLayer();
    DynamicDecodeLayer(DynamicDecodeLayer const &dynamic_decode_layer);

    void setup(const size_t batch_size, const size_t beam_width, TensorMap *runtime_args);
    void forward(TensorMap *output_tensors, TensorMap *input_tensors);
    void forward(std::unordered_map<std::string, Tensor> *output_tensors,
                 const std::unordered_map<std::string, Tensor> *input_tensors);
};

} // namespace space_llm
