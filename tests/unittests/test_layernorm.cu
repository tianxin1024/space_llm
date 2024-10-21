#include <iostream>
#include <sys/time.h>

#include "utils/cuda_utils.h"
#include "utils/memory_utils.h"
#include "kernels/layernorm_kernels.h"

using namespace space_llm;

template <typename T>
void test_layernorm(const int m, const int n);

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("[ERROR] test_layernorm max_m, max_n data_type\n");
        printf("e.g., ./bin/test_layernorm 1 1024 1\n");
        return 0;
    }

    int max_m = atoi(argv[1]);
    int max_n = atoi(argv[2]);
    const FtCudaDataType data_type = static_cast<FtCudaDataType>(atoi(argv[3])); // 0 FP32, 1 FP16, 2 BF16

    for (int m = 1; m <= max_m; m *= 2) {
        for (int n = 128; n <= max_n; n *= 2) {
            if (data_type == FP16) {
                test_layernorm<half>(m, n);
            } else if (data_type == FP32) {
                test_layernorm<float>(m, n);
            } else {
                QK_LOG_ERROR("data_type should be fp32, fp16");
                exit(-1);
            }
        }
    }

    return 0;
}

template <typename T>
void test_layernorm(const int m, const int n) {
    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));
    printf("Device %s\n", prop.name);

    const float layernorm_eps = 1e-4f;
    T *input, *output_opt, *output_baseline, *gamma, *beta;
    deviceMalloc(&input, m * n);
    deviceMalloc(&output_baseline, m * n);
    deviceMalloc(&output_opt, m * n);
    deviceMalloc(&gamma, n);
    deviceMalloc(&beta, n);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    const int ite = 5000;

    // warmup
    for (int i = 0; i < 1000; i++) {
        invokeGeneralLayerNorm<T>(output_baseline, input, gamma, beta, layernorm_eps, m, n, (float *)nullptr, 0, stream);
        invokeGeneralLayerNorm<T>(output_opt, input, gamma, beta, layernorm_eps, m, n, (float *)nullptr, 0, stream, true);
    }

    struct timeval start, end;
    cudaDeviceSynchronize();
    gettimeofday(&start, NULL);
    for (int i = 0; i < ite; i++) {
        invokeGeneralLayerNorm<T>(output_baseline, input, gamma, beta, layernorm_eps, m, n, (float *)nullptr, 0, stream);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    float baseline_time = ((end.tv_sec - start.tv_sec) * 1000000. + (end.tv_usec - start.tv_usec) * 1.) / ite;

    struct timeval start_2, end_2;
    cudaDeviceSynchronize();
    gettimeofday(&start_2, NULL);
    for (int i = 0; i < ite; i++) {
        invokeGeneralLayerNorm<T>(output_opt, input, gamma, beta, layernorm_eps, m, n, (float *)nullptr, 0, stream, true);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end_2, NULL);
    float opt_time = ((end_2.tv_sec - start_2.tv_sec) * 1000000. + (end_2.tv_usec - start.tv_usec) * 1.) / ite;

    print_abs_mean(output_baseline, m * n, stream, "output_baseline");
    print_abs_mean(output_opt, m * n, stream, "output_opt");

    printf("[INFO] baseline time: %f us \n", baseline_time);
    printf("[INFO] opt time: %f us\n", opt_time);
    printf("[INFO] m %d, n %d, speedup: %f\n", m, n, baseline_time / opt_time);

    return;
}
