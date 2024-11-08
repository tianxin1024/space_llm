#pragma once

#include "layers/BaseLayer.h"
#include "layers/DynamicDecodeBaseLayer.h"

namespace space_llm {

template <typename T>
class DynamicDecodeLayer : public BaseLayer {
protected:
    size_t vocab_size_;
    size_t vocab_size_padded_;
    cudaDeviceProp *cuda_device_prop_;

    DynamicDecodeBaseLayer *online_beamsearch_decode_;
    DynamicDecodeBaseLayer *beamsearch_decode_;
    DynamicDecodeBaseLayer *topk_decode_;
    DynamicDecodeBaseLayer *topp_decode_;

    void allocateBuffer() override;
    void freeBuffer() override;
    void initialize();
    bool hasDiffRuntimeArgs(TensorMap *input_tensors);

    // List of argument names which can have different values in runtime
    // and does not support a batched version of kernel in beam search.
    std::vector<std::string> runtime_arg_names_ = {"beam_search_diversity_rate",
                                                   "temperature",
                                                   "len_penalty",
                                                   "repetition_penalty",
                                                   "presence_penalty",
                                                   "min_length"};

    bool has_diff_runtime_args_ = false;
    int *h_pinned_finished_sum_ = nullptr;

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
