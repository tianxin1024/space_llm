#include "models/decoding/Decoding.h"

namespace space_llm {

template <typename T>
void Decoding<T>::initialize() {
    decoder_ = new Decoder<T>(max_batch_size_ * beam_width_,
                              head_num_,
                              size_per_head_,
                              inter_size_,
                              num_layer_,
                              stream_,
                              cublas_wrapper_,
                              allocator_,
                              is_free_buffer_after_forward_);

    dynamic_decode_layer_ = new DynamicDecodeLayer<DynamicDecodeType>(vocab_size_,
                                                                      vocab_size_padded_,
                                                                      end_id_,
                                                                      stream_,
                                                                      cublas_wrapper_,
                                                                      allocator_,
                                                                      is_free_buffer_after_forward_,
                                                                      cuda_device_prop_);
}

template <typename T>
Decoding<T>::Decoding(size_t max_batch_size,
                      size_t max_seq_len,
                      size_t mem_max_seq_len,
                      size_t beam_width,
                      size_t head_num,
                      size_t size_per_head,
                      size_t inter_size,
                      size_t num_layer,
                      size_t vocab_size,
                      int start_id,
                      int end_id,
                      float beam_search_diversity_rate,
                      uint top_k,
                      float top_p,
                      float temperature,
                      float len_penalty,
                      float repetition_penalty,
                      cudaStream_t stream,
                      cublasMMWrapper *cublas_wrapper,
                      IAllocator *allocator,
                      bool is_free_buffer_after_forward,
                      cudaDeviceProp *cuda_device_prop) :
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    mem_max_seq_len_(mem_max_seq_len),
    beam_width_(beam_width),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    start_id_(start_id),
    end_id_(end_id),
    beam_search_diversity_rate_(beam_search_diversity_rate),
    top_k_(top_k),
    top_p_(top_p),
    temperature_(temperature),
    len_penalty_(len_penalty),
    repetition_penalty_(repetition_penalty) {
    vocab_size_padded_ = vocab_size_;
    if (std::is_same<half, T>::value) {
        vocab_size_padded_ = ((size_t)ceil(vocab_size_padded_ / 8.) * 8);
    }

    initialize();
}

template <typename T>
Decoding<T>::Decoding(Decoding<T> const &decoding) :
    BaseLayer(decoding),
    max_batch_size_(decoding.max_batch_size),
    max_seq_len_(decoding.max_seq_len),
    mem_max_seq_len_(decoding.mem_max_seq_len),
    beam_width_(decoding.beam_width),
    head_num_(decoding.head_num),
    size_per_head_(decoding.size_per_head),
    inter_size_(decoding.inter_size),
    num_layer_(decoding.num_layer),
    start_id_(decoding.start_id),
    end_id_(decoding.end_id),
    beam_search_diversity_rate_(decoding.beam_search_diversity_rate),
    top_k_(decoding.top_k),
    top_p_(decoding.top_p),
    temperature_(decoding.temperature),
    len_penalty_(decoding.len_penalty),
    repetition_penalty_(decoding.repetition_penalty) {
    initialize();
}

template <typename T>
Decoding<T>::~Decoding() {
    delete decoder_;
    delete dynamic_decode_layer_;
    freeBuffer();
}

} // namespace space_llm
