#include "models/vit/ViT.h"
#include "layers/attention_layers/UnfusedAttentionLayer.h"
#include "layers/ffnLayer.h"

namespace space_llm {

template <typename T>
void ViTTransformer<T>::initialize() {
    QK_LOG_DEBUG("img_size: %lu, patch_size:%lu\n"
                 "batch_size:%lu, chn_num  : %lu\n"
                 "seq_len   :%lu, embed_dim: %lu\n"
                 "head_num  :%lu, head_dim : %lu\n"
                 "inter_size:%lu, num_layer: %lu\n"
                 "att_type  : %d, \n",
                 img_size_,
                 patch_size_,
                 max_batch_size_,
                 chn_num_,
                 max_seq_len_,
                 embed_dim_,
                 head_num_,
                 head_dim_,
                 inter_size_,
                 num_layer_,
                 int(attention_type_));

    if (img_size_ % patch_size_ != 0) {
        std::ostringstream buffer;
        buffer << "[QK][ERROR] IMG size & PITCH size mismatch. " << img_size_ << " % " << patch_size_ << " !=0 \n";
        throw std::runtime_error(buffer.str());
    }

    if (head_num_ * head_dim_ != embed_dim_) {
        std::ostringstream buffer;
        buffer << "[QK][ERROR] Embed size and head number mismatch. Embed_dim=" << embed_dim_
               << "; head_num * head_dim = "
               << "(" << head_num_ << "*" << head_dim_ << ")=" << head_num_ * head_num_ << std::endl;
        throw std::runtime_error(buffer.str());
    }

    max_seq_len_ = request_seq_len_;

    if ((attention_type_ == AttentionType::FUSED_MHA) && std::is_same<T, half>::value == true) {
        QK_LOG_INFO("[QK] Attention type: FUSED_MHA\n");
    } else if (attention_type_ == AttentionType::UNFUSED_MHA) {
        if (request_seq_len_ % 8 != 0 && std::is_same<half, T>::value) {
            max_seq_len_ = (request_seq_len_ + 7) / 8 * 8;
            QK_LOG_DEBUG("Request sequence length(%lu) is odd with unfused mha. Padding to %lu\n", request_seq_len_, max_seq_len_);
        }

        attention_type_ = new UnfusedAttentionLayer<T>(max_batch_size_,
                                                       max_seq_len_,
                                                       head_num_,
                                                       head_dim_,
                                                       q_scaling_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       is_free_buffer_after_forward_,
                                                       false);
    } else {
        throw std::runtime_error(std::string("[QK][ERROR] Invalid attention type or sequence length\n"));
    }

    ffn_layer_ = new GeluffnLayer<T>(max_batch_size_,
                                     max_seq_len_,
                                     head_num_,
                                     head_dim_,
                                     0,
                                     inter_size_,
                                     stream_,
                                     cublas_wrapper_,
                                     allocator_,
                                     is_free_buffer_after_forward_,
                                     false);
}

template <typename T>
ViTTransformer<T>::ViTTransformer(size_t max_batch_size,
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
                                  AttentionType attention_type) :
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    img_size_(img_size),
    chn_num_(chn_num),
    patch_size_(patch_size),
    embed_dim_(embed_dim),
    head_num_(head_num),
    inter_size_(inter_size),
    num_layer_(num_layer),
    with_cls_token_(with_cls_token_),
    sm_(sm),
    q_scaling_(q_scaling),
    cudnn_handle_(cudnn_handle),
    attention_type_(attention_type) {
    initialize();
}

template <typename T>
ViTTransformer<T>::ViTTransformer(ViTTransformer<T> const &vit) :
    BaseLayer(vit),
    max_batch_size_(vit.max_batch_size_),
    img_size_(vit.img_size_),
    chn_num_(vit.chn_num_),
    patch_size_(vit.patch_size_),
    max_seq_len_(vit.max_seq_len_),
    request_seq_len_(vit.request_seq_len_),
    embed_dim_(vit.embed_dim_),
    head_num_(vit.head_num_),
    head_dim_(vit.head_dim_),
    inter_size_(vit.inter_size_),
    num_layer_(vit.num_layer_),
    with_cls_token_(vit.with_cls_token_),
    sm_(vit.sm_),
    q_scaling_(vit.q_scaling_),
    attention_type_(vit.attention_type_),
    cudnn_handle_(vit.cudnn_handle_) {
    initialize();
}

template <typename T>
ViTTransformer<T>::~ViTTransformer() {
    delete attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template <typename T>
void ViTTransformer<T>::allocateBuffer() {
    if (is_allocate_buffer_ == false) {
        embed_buf_1_ = (T *)allocator_->reMalloc(embed_buf_1_, sizeof(T) * max_batch_size_ * max_seq_len_ * embed_dim_, false);
        embed_buf_2_ = (T *)allocator_->reMalloc(embed_buf_2_, sizeof(T) * max_batch_size_ * max_seq_len_ * embed_dim_, false);
        embed_buf_3_ = (T *)allocator_->reMalloc(embed_buf_3_, sizeof(T) * max_batch_size_ * max_seq_len_ * embed_dim_, false);
        mask_buf_ = (T *)allocator_->reMalloc(mask_buf_, sizeof(T) * max_batch_size_ * max_seq_len_, false);
        padding_offset_ = (int *)allocator_->reMalloc(padding_offset_, sizeof(int) * max_batch_size_ * max_seq_len_, false);
        h_pinned_token_num_ptr_ = (size_t *)allocator_->reMalloc(h_pinned_token_num_ptr_, sizeof(size_t), true, true);

        trt_mha_padding_offset_ = (int *)allocator_->reMalloc(trt_mha_padding_offset_, sizeof(int) * (2 * max_batch_size_ + 1), false);
        seq_len_vec_ = (int *)allocator_->reMalloc(seq_len_vec_, sizeof(int) * max_batch_size_, false);

        setSeqLenVec(max_batch_size_);
        setDefaultMask(max_batch_size_);
        setDefaultPaddingOffset(max_batch_size_);

        is_allocate_buffer_ = true;
    }
}

template <typename T>
bool ViTTransformer<T>::setSeqLenVec(size_t batch_size) {
    int *seq_len_vec = new int[batch_size];
    for (int i = 0; i < batch_size; ++i) {
        seq_len_vec[i] = request_seq_len_;
    }
    cudaMemcpy(seq_len_vec_, seq_len_vec, sizeof(int) * batch_size, cudaMemcpyHostToDevice);
    delete[] seq_len_vec;
    return true;
}

template <typename T>
void ViTTransformer<T>::setDefaultMask(size_t batch_size) {
    invokeBuildEncoderAttentionMask(mask_buf_, seq_len_vec_, batch_size, max_seq_len_, stream_);
}

template <typename T>
void ViTTransformer<T>::setDefaultPaddingOffset(size_t batch_size) {
    invokeGetPaddingOffset(
        h_pinned_token_num_ptr_, &nopad_token_num_, padding_offset_, seq_len_vec_, batch_size, max_seq_len_, stream_);
}

} // namespace space_llm
