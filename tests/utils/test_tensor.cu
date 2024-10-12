#include <iostream>
#include <vector>
#include "utils/tensor.h"

using namespace space_llm;

template <typename T>
void deviceMalloc(T **ptr, size_t size, bool is_random_initialize) {
    check_cuda_error(cudaMalloc((void **)(ptr), sizeof(T) * size));
}

template <typename T>
void test_tensor(int batch_size, int img_size, int embed_dim, int in_channel, int seq_len) {
    T *input_d, *output_d;
    deviceMalloc(&input_d, batch_size * img_size * img_size * in_channel, false);
    deviceMalloc(&output_d, batch_size * seq_len * embed_dim, false);

    // input_tensors:
    //      input_img, BCHW [batch, chn_num, img_size, img_size]

    std::vector<Tensor> input_tensors = std::vector<Tensor>{
        Tensor{MEMORY_GPU, getTensorType<T>(),
               std::vector<size_t>{(size_t)batch_size, (size_t)in_channel, (size_t)img_size, (size_t)img_size},
               input_d}};

    std::vector<Tensor> output_tensors = std::vector<Tensor>{
        Tensor{MEMORY_GPU, getTensorType<T>(),
               std::vector<size_t>{(size_t)batch_size, (size_t)seq_len, (size_t)embed_dim}, output_d}};

    const size_t input_batch_size = input_tensors.at(0).shape[0];
    const size_t input_chn_num = input_tensors.at(0).shape[1];
    const size_t input_img_size = input_tensors.at(0).shape[2];

    QK_CHECK(input_batch_size == batch_size);
    QK_CHECK(input_chn_num == in_channel);
    QK_CHECK(input_img_size == img_size);

    // TensorMap input_tensor({{"ffn_input", input_tensors.at(0)}});
    // TensorMap output_tensor({{"ffn_output", output_tensors.at(0)}});
    printf("Input_tensor size: [%ld, %ld, %ld, %ld]\n", input_batch_size, input_chn_num, input_img_size, input_img_size);

    // free data
    check_cuda_error(cudaFree(input_d));
    check_cuda_error(cudaFree(output_d));
}

int main(int argc, char *argv[]) {
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device %s\n", prop.name);

    if (argc != 7) {
        printf("Usage: test_tensor batch_size img_size embed_dim in_channel seq_len is_fp16\n");
        printf("e.g. ./bin/test_tensor 1 224 768 12 768 0 \n");
        return 0;
    }

    const int batch_size = atoi(argv[1]);
    const int img_size = atoi(argv[2]);
    const int embed_dim = atoi(argv[3]);
    const int in_channel = atoi(argv[4]);
    const int seq_len = atoi(argv[5]);
    const int is_fp16 = atoi(argv[6]);

    if (is_fp16) {
        test_tensor<half>(batch_size, img_size, embed_dim, in_channel, seq_len);
    } else {
        test_tensor<float>(batch_size, img_size, embed_dim, in_channel, seq_len);
    }
}
