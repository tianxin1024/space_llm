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
bool Decoding<T>::isValidSeqLen(size_t seq_len) {
    if (max_seq_len_ == 0) {
        // allocator additional one to put the start token
        max_seq_len_ = seq_len + 1;
        return true;
    } else {
        return seq_len <= max_seq_len_;
    }
}

template <typename T>
bool Decoding<T>::isValidBatchSize(size_t batch_size) {
    if (max_batch_size_ == 0) {
        max_batch_size_ = batch_size;
        return true;
    } else {
        return batch_size <= max_batch_size_;
    }
}

template <typename T>
bool Decoding<T>::isValidMemSeqLen(size_t seq_len) {
    if (mem_max_seq_len_ == 0) {
        mem_max_seq_len_ = seq_len;
        return true;
    } else {
        return seq_len <= mem_max_seq_len_;
    }
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

template <typename T>
void Decoding<T>::forward(std::vector<Tensor> *output_tensors,
                          const std::vector<Tensor> *input_tensors,
                          const DecodingWeight<T> *decoding_weights) {
    // input_tensors:
    //      encoder_output [batch_size * beam, mem_max_seq_len, memory_hidden_dimension]
    //      encoder_sequence_length [batch_size * beam]

    // output_tensors:
    //      output_ids [max_seq_len, batch_size, beam]
    //      parent_ids [max_seq_len, batch_size, beam]
    //      sequence_length [batch_size, beam], record the number of generated token, except the start token

    // Step is from 1 ~ max_seq_len,
    // When step = k,  we put output ids and caches at step k, and the sequence_length would be k - 1 before
    // complete this step.

    std::unordered_map<std::string, Tensor> input_tensors_map{{"encoder_output", input_tensors->at(0)},
                                                              {"encoder_sequence_length", input_tensors->at(1)}};
    std::unordered_map<std::string, Tensor> output_tensors_map{{"output_ids", output_tensors->at(0)},
                                                               {"parent_ids", output_tensors->at(1)},
                                                               {"sequence_length", output_tensors->at(2)}};
    forward(&output_tensors_map, &input_tensors_map, decoding_weights);
}

template <typename T>
void Decoding<T>::forward(std::unordered_map<std::string, Tensor> *output_tensors,
                          const std::unordered_map<std::string, Tensor> *input_tensors,
                          const DecodingWeight<T> *decoding_weights) {
    TensorMap input_map(*input_tensors);
    TensorMap output_map(*output_tensors);
    forward(&output_map, &input_map, decoding_weights);
}

template <typename T>
void Decoding<T>::forward(TensorMap *output_tensors,
                          TensorMap *input_tensors,
                          const DecodingWeight<T> *decoding_weights) {
    // input_tensors:
    //      encoder_output [batch_size * beam, mem_max_seq_len, memory_hidden_dimension]
    //      encoder_sequence_length [batch_size * beam]

    // output_tensors:
    //      output_ids [max_seq_len, batch_size, beam]
    //      parent_ids [max_seq_len, batch_size, beam]
    //      sequence_length [batch_size, beam], record the number of generated token, except the start token
    //      cum_log_probs [batch_size, beam], optional, must be float*.

    // Step is from 1 ~ max_seq_len,
    // When step = k,  we put output ids and caches at step k, and the sequence_length would be k - 1 before
    // complete this step.

    QK_CHECK(input_tensors->size() >= 2);
    QK_CHECK(output_tensors->size() >= 3);
    isValidSeqLen(output_tensors->at("output_ids").shape[0]);
    isValidBatchSize(output_tensors->at("output_ids").shape[1]);
    isValidMemSeqLen(input_tensors->at("encoder_output").shape[1]);
}

} // namespace space_llm
