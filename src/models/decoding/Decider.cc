#include "models/decoding/Decoder.h"

namespace space_llm {

template <typename T>
void Decoder<T>::initialize() {
    self_attention_layer_ = new DecoderSelfAttentionLayer<T>(max_batch_size_,
                                                             head_num_,
                                                             size_per_head_,
                                                             stream_,
                                                             cublas_wrapper_,
                                                             allocator_,
                                                             is_free_buffer_after_forward_);

    cross_attention_layer_ = new DecoderCrossAttentionLayer<T>(max_batch_size_,
                                                               head_num_,
                                                               size_per_head_,
                                                               stream_,
                                                               cublas_wrapper_,
                                                               allocator_,
                                                               is_free_buffer_after_forward_);

    ffn_layer_ = new ReluffnLayer<T>(max_batch_size_,
                                     1,
                                     head_num_,
                                     size_per_head_,
                                     0, // expert_num
                                     inter_size_,
                                     stream_,
                                     cublas_wrapper_,
                                     allocator_,
                                     is_free_buffer_after_forward_);
}

template <typename T>
Decoder<T>::Decoder(size_t max_batch_size,
                    size_t head_num,
                    size_t size_per_head,
                    size_t inter_size,
                    size_t num_layer,
                    cudaStream_t stream,
                    cublasMMWrapper *cublas_wrapper,
                    IAllocator *allocator,
                    bool is_free_buffer_after_forward) :
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    hidden_units_(head_num_ * size_per_head_) {
    initialize();
}

template <typename T>
Decoder<T>::Decoder(Decoder<T> const &decoder) :
    BaseLayer(decoder),
    max_batch_size_(decoder.max_batch_size_),
    head_num_(decoder.head_num_),
    size_per_head_(decoder.size_per_head_),
    inter_size_(decoder.inter_size_),
    num_layer_(decoder.num_layer_),
    hidden_units_(decoder.head_num_ * decoder.size_per_head_) {
    initialize();
}

template <typename T>
Decoder<T>::~Decoder() {
    delete self_attention_layer_;
    delete cross_attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template <typename T>
void Decoder<T>::forward(std::vector<Tensor> *output_tensors,
                         const std::vector<Tensor> *input_tensors,
                         const std::vector<DecoderLayerWeight<T>> *decoder_layer_weights) {
    printf(">>>>>>>>>>>>>> Decoder forward\n\n\n");
}

} // namespace space_llm
