#include <iostream>
#include "models/decoding/Decoding.h"
#include <cuda_profiler_api.h>

using namespace space_llm;

template <typename T>
int decodingExample(const size_t batch_size,
                    const size_t beam_width,
                    const size_t head_num,
                    const size_t size_per_head,
                    const size_t inter_size,
                    const size_t vocab_size,
                    const size_t num_layers,
                    const size_t max_seq_len,
                    const size_t memory_max_seq_len,
                    const size_t memory_hidden_units,
                    const int top_k,
                    const int top_p);

int main(int argc, char **argv) {
    if (argc != 14) {
        printf("[ERROR] decoding_example batch_size beam_width head_num size_per_head inter_size vocab_size"
               " num_layers max_seq_len memory_max_seq_len memory_hidden_units top_k top_p data_type\n");
        printf("e.g. ./bin/decoding_example 4 1 8 64 2048 30000 6 32 32 512 0 0.6 0\n");
        return 0;
    }

    int batch_size = atoi(argv[1]);
    int beam_width = atoi(argv[2]);
    int head_num = atoi(argv[3]);
    int size_per_head = atoi(argv[4]);
    int inter_size = atoi(argv[5]);
    int vocab_size = atoi(argv[6]);
    int num_layers = atoi(argv[7]);
    int max_seq_len = atoi(argv[8]);
    int memory_max_seq_len = atoi(argv[9]);
    int memory_hidden_units = atoi(argv[10]);
    int top_k = atoi(argv[11]);
    float top_p = atof(argv[12]);
    const CublasDataType data_type = static_cast<CublasDataType>(atoi(argv[13])); // 0 FP32, 1 FP16, 2 BF16

    if (data_type == FLOAT_DATATYPE) {
        return decodingExample<float>(batch_size,
                                      beam_width,
                                      head_num,
                                      size_per_head,
                                      inter_size,
                                      vocab_size,
                                      num_layers,
                                      max_seq_len,
                                      memory_max_seq_len,
                                      memory_hidden_units,
                                      top_k,
                                      top_p);
    } else if (data_type == HALF_DATATYPE) {
        return decodingExample<half>(batch_size,
                                     beam_width,
                                     head_num,
                                     size_per_head,
                                     inter_size,
                                     vocab_size,
                                     num_layers,
                                     max_seq_len,
                                     memory_max_seq_len,
                                     memory_hidden_units,
                                     top_k,
                                     top_p);

    } else {
        throw std::runtime_error(std::string("[QK][ERROR] is_fp16 should be 0 (use float)"
                                             "or 1 (use half). \n"));
    }

    return 0;
}

template <typename T>
int decodingExample(const size_t batch_size,
                    const size_t beam_width,
                    const size_t head_num,
                    const size_t size_per_head,
                    const size_t inter_size,
                    const size_t vocab_size,
                    const size_t num_layers,
                    const size_t max_seq_len,
                    const size_t memory_max_seq_len,
                    const size_t memory_hidden_units,
                    const int top_k,
                    const int top_p) {
    const size_t hidden_units = head_num * size_per_head;
    const int start_id = 0;
    const int end_id = 1;

    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStreamCreate(&stream);
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
    cublasSetStream(cublas_handle, stream);
    cublasAlgoMap *cublas_algo_map = new cublasAlgoMap("gemm_config.in");

    Allocator allocator(getDevice());

    std::mutex *cublas_wrapper_mutex = new std::mutex();
    cublasMMWrapper cublas_wrapper = cublasMMWrapper(cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, &allocator);

    if (std::is_same<T, half>::value) {
        cublas_wrapper.setFP16GemmConfig();
    } else if (std::is_same<T, float>::value) {
        cublas_wrapper.setFP32GemmConfig();
    }

    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));

    DecodingWeight<T> decoding_weights(
        hidden_units, inter_size, vocab_size, num_layers, max_seq_len, memory_hidden_units);

    Decoding<T> decoding = Decoding<T>(batch_size,
                                       max_seq_len,
                                       memory_max_seq_len,
                                       beam_width,
                                       head_num,
                                       size_per_head,
                                       inter_size,
                                       num_layers,
                                       vocab_size,
                                       start_id,
                                       end_id,
                                       0.0f,
                                       top_k,
                                       top_p,
                                       1.0,  // temperature
                                       1.0f, // len_penalty
                                       1.0,  // repetition_penalty
                                       stream,
                                       &cublas_wrapper,
                                       &allocator,
                                       false,
                                       &prop);
}
