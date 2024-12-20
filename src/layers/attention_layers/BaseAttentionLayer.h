#pragma once

#include <vector>
#include <assert.h>

#include "layers/BaseLayer.h"
#include "layers/attention_layers/AttentionWeight.h"

namespace space_llm {

enum class AttentionType {
    UNFUSED_MHA,
    UNFUSED_PADDED_MHA,
    FUSED_MHA,
    FUSED_PADDED_MHA
};

template <typename T>
class BaseAttentionLayer : public BaseLayer {
public:
    virtual void forward(TensorMap *output_tensors, TensorMap *input_tensors, const AttentionWeight<T> *attention_weights) = 0;

    BaseAttentionLayer(cudaStream_t stream,
                       cublasMMWrapper *cublas_wrapper,
                       IAllocator *allocator,
                       bool is_free_buffer_after_forward,
                       bool sparse = false) :
        BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse) {
    }

    virtual ~BaseAttentionLayer() = default;
    virtual bool isValidSeqLen(const size_t seq_len) {
        return true;
    }
};

inline bool isUnPaddedMHA(AttentionType attention_type) {
    return attention_type == AttentionType::FUSED_MHA || attention_type == AttentionType::UNFUSED_MHA;
}

} // namespace space_llm
