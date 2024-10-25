#pragma once

#include "layers/attention_layers/BaseAttentionLayer.h"
#include "layers/ffnLayer.h"
#include "models/vit/ViTWeight.h"
#include <cudnn.h>

namespace space_llm {

template <typename T>
class ViTTransformer : public BaseLayer {
private:
    size_t max_batch_size_ = 0;
    size_t img_size_ = 224;
    size_t chn_num_ = 3;
    size_t patch_size_ = 16; // preproc patch size
    size_t max_seq_len_;
    size_t request_seq_len_;
    size_t embed_dim_;  // patch conv out units, size_per_head = embed_dim / head_num
    size_t head_num_;   // mha head num
    size_t head_dim_;   // mha head size
    size_t inter_size_; // FF internal size
    size_t num_layer_;
    size_t nopad_token_num_;
    bool with_cls_token_;
    int sm_;
    static constexpr float layernorm_eps_ = 1e-6f;
    float q_scaling_;
    AttentionType attention_type_;
    // TODO add conv op
    cudnnHandle_t cudnn_handle_;

    BaseAttentionLayer<T> *attention_layer_;
    ffnLayer<T> *ffn_layer_;

    void allocateBuffer();
    void freeBuffer();

    bool setSeqLenVec(size_t batch_size);
    void setDefaultMask(size_t batch_size);
    void setDefaultPaddingOffset(size_t batch_size);

    void initialize();

protected:
    T *embed_buf_1_ = nullptr;
    T *embed_buf_2_ = nullptr;
    T *embed_buf_3_ = nullptr;
    T *mask_buf_ = nullptr;
    int *trt_mha_padding_offset_ = nullptr;
    int *seq_len_vec_ = nullptr;
    int *padding_offset_ = nullptr;
    size_t *h_pinned_token_num_ptr_ = nullptr;

public:
    ViTTransformer(size_t max_batch_size,
                   size_t img_size,
                   size_t chn_num,
                   size_t patch_size,
                   size_t embed_dim,
                   size_t head_num,
                   size_t inter_size,
                   size_t num_layer,
                   bool with_cls_token,
                   int sm,
                   float q_scaling,
                   cudaStream_t stream,
                   cudnnHandle_t cudnn_handle,
                   cublasMMWrapper *cublas_wrapper,
                   IAllocator *allocator,
                   bool is_free_buffer_after_forward,
                   AttentionType attention_type);

    ViTTransformer(ViTTransformer<T> const &vit_layer);

    ~ViTTransformer();

    void forward(std::vector<Tensor> *output_tensors, const std::vector<Tensor> *input_tensors, const ViTWeight<T> *weights);
};

} // namespace space_llm
