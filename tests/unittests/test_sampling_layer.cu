#include <algorithm> // std::min, std::max
#include <iostream>  // snprintf
#include <math.h>    // expf, log
#include <stdlib.h>  // rand
#include <string>    // std::string
#include <vector>    // std::vector

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include "gtest_utils.h"
#include "kernels/sampling_topk_kernels.h"
#include "kernels/sampling_topp_kernels.h"
#include "layers/DynamicDecodeLayer.h"
#include "utils/cuda_utils.h"

using namespace space_llm;

namespace {

struct SamplingKernelTestParam {
    size_t batch_size;
    size_t vocab_size;
    size_t beam_width;
    uint top_k;
    float top_p;
    size_t output_len;

    std::string toString() {
        return fmtstr("SamplingKernelTestParam[batch=%ld, vocab=%ld, beam=%ld, k=%u, p=%3.1f, output_len=%ld]",
                      batch_size,
                      vocab_size,
                      beam_width,
                      top_k,
                      top_p,
                      output_len);
    }
};

template <typename T>
class SamplingDecodeTest : public testing::Test {
protected:
    unsigned long long seed = 0;
    const static unsigned long long max_seed = 30;
    const size_t batch_size = 6;
    const size_t beam_width = 1;
    const size_t batchxbeam = batch_size * beam_width;
    const size_t vocab_size = 8;
    const size_t max_input_len = 0;
    const size_t max_output_len = 3;
    const size_t max_seq_len = max_input_len + max_output_len;
    const int end_id = vocab_size - 1;
    const DataType data_type = getTensorType<T>();

    // vocab size 8 & length 3
    T *test_input_logits;

    cudaStream_t stream;
    Allocator *allocator;
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    std::mutex *cublas_wrapper_mutex;
    cublasMMWrapper *cublas_wrapper;
    DynamicDecodeLayer<T> *dynamic_decode_layer;

    int *h_output_ids;
    T *h_logits;
    T *h_probs;
    T *h_log_probs;
    float *h_cum_log_probs;
    float *h_output_log_probs;

    T *d_logits;
    int *d_input_lengths;
    float *d_cum_log_probs;
    float *d_output_log_probs;
    int *d_output_ids;
    int *d_end_ids;

    void setup(unsigned long long seed = 0) {
        this->seed = seed;

        check_cuda_error(cudaStreamCreate(&stream));
        allocator = new Allocator(getDevice());
        allocator->setStream(stream);

        struct cudaDeviceProp prop;
        check_cuda_error(cudaGetDeviceProperties(&prop, 0));
        check_cuda_error(cublasCreate(&cublas_handle));
        check_cuda_error(cublasLtCreate(&cublaslt_handle));
        check_cuda_error(cublasSetStream(cublas_handle, stream));
        cublasAlgoMap cublas_algo_map(GEMM_CONFIG);
        cublas_wrapper_mutex = new std::mutex();

        cublas_wrapper = new cublasMMWrapper(cublas_handle,
                                             cublaslt_handle,
                                             stream,
                                             &cublas_algo_map,
                                             cublas_wrapper_mutex,
                                             allocator);

        dynamic_decode_layer = new DynamicDecodeLayer<T>(vocab_size,
                                                         vocab_size,
                                                         end_id,
                                                         stream,
                                                         cublas_wrapper,
                                                         allocator,
                                                         false,  // is_free_buffer_after_forward
                                                         &prop); // cuda_device_prop

        h_output_ids = new int[batchxbeam];
        h_logits = new T[batchxbeam * vocab_size];
        h_probs = new T[batchxbeam * vocab_size];
        h_log_probs = new T[batchxbeam * vocab_size];
        h_cum_log_probs = new float[batchxbeam];
        h_output_log_probs = new float[max_output_len * batchxbeam];

        // prob = (0.4, 0,3, 0.2, 0.1, ...)
        test_input_logits = new T[24]{
            -0.9163, -1.2040, -1.6094, -2.3026, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, // step 0
            -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -0.9163, -1.2040, -1.6094, -2.3026, // step 1
            -FLT_MAX, -FLT_MAX, -0.9163, -1.2040, -1.6094, -2.3026, -FLT_MAX, -FLT_MAX  // step 2
        };

        d_logits = reinterpret_cast<T *>(allocator->malloc(sizeof(T) * batchxbeam * vocab_size, true));
        d_input_lengths = reinterpret_cast<int *>(allocator->malloc(sizeof(int) * batchxbeam));
        d_cum_log_probs = reinterpret_cast<float *>(allocator->malloc(sizeof(float) * batchxbeam));
        d_output_log_probs = reinterpret_cast<float *>(allocator->malloc(sizeof(float) * max_output_len * batchxbeam));
        d_output_ids = reinterpret_cast<int *>(allocator->malloc(sizeof(int) * max_seq_len * batchxbeam));
        d_end_ids = reinterpret_cast<int *>(allocator->malloc(sizeof(int) * batchxbeam));

        // Init by zero.
        cudaMemset(d_cum_log_probs, 0, sizeof(float) * batchxbeam);
        cudaMemset(d_output_log_probs, 0, sizeof(float) * max_output_len * batchxbeam);
        cudaMemset(d_output_ids, 0, sizeof(int) * max_seq_len * batchxbeam);
        deviceFill(d_end_ids, batchxbeam, end_id, stream);
    }

    // void teardown() {
    //     delete[] test_input_logits;
    //     delete[] h_output_ids;
    //     delete[] h_logits;
    //     delete[] h_probs;
    //     delete[] h_log_probs;
    //     delete[] h_cum_log_probs;
    //     delete[] h_output_log_probs;
    //     delete dynamic_decode_layer;
    //     delete cublas_wrapper;
    //     delete cublas_wrapper_mutex;
    //     delete allocator;
    //     check_cuda_error(cublasDestroy(cublas_handle));
    //     check_cuda_error(cublasLtDestroy(cublaslt_handle));
    //     check_cuda_error(cudaStreamDestroy(stream));
    // }

    void teardown() {
        delete[] test_input_logits;
        delete[] h_output_ids;
        delete[] h_logits;
        delete[] h_probs;
        delete[] h_log_probs;
        delete[] h_cum_log_probs;
        delete[] h_output_log_probs;
        // delete dynamic_decode_layer;
        // delete cublas_wrapper;
        // delete cublas_wrapper_mutex;
        // delete allocator;
        // check_cuda_error(cublasDestroy(cublas_handle));
        // check_cuda_error(cublasLtDestroy(cublaslt_handle));
        // check_cuda_error(cudaStreamDestroy(stream));
    }

    TensorMap *createInputTensors(int *topk,
                                  size_t topk_size,
                                  float *topp,
                                  size_t topp_size,
                                  float *temperature,
                                  float *repetition_penalty) {
        // construct common input tensors
        TensorMap *input_tensors = new TensorMap();
        if (topk != nullptr) {
            input_tensors->insert({"runtime_top_k", {MEMORY_CPU, TYPE_INT32, {topk_size}, topk}});
        }
        if (topp != nullptr) {
            input_tensors->insert({"runtime_top_p", {MEMORY_CPU, TYPE_FP32, {topp_size}, topp}});
        }
        if (temperature != nullptr) {
            input_tensors->insert({"temperature", Tensor{MEMORY_CPU, TYPE_FP32, {1}, temperature}});
        }
        if (repetition_penalty != nullptr) {
            input_tensors->insert({"repetition_penalty"}, Tensor{MEMORY_CPU, TYPE_FP32, {1}, repetition_penalty});
        }
        input_tensors->insert({"logits", Tensor{MEMORY_GPU, TYPE_FP32, {batch_size, beam_width, vocab_size}, d_logits}});
        input_tensors->insert({"embedding_bias", Tensor{MEMORY_GPU, data_type, {vocab_size}, nullptr}});
        input_tensors->insert({"max_input_len", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_len}});
        input_tensors->insert({"max_input_length", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_len}});
        input_tensors->insert({"input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size, beam_width}, d_input_lengths}});
        input_tensors->insert({"end_id", Tensor{MEMORY_CPU, TYPE_INT32, {batchxbeam}, &d_end_ids}});
        input_tensors->insert({"random_seed", Tensor{MEMORY_CPU, TYPE_UINT64, {1}, &seed}});
        return input_tensors;
    }

    TensorMap *createOutputTensors() {
        // construct common output tensors
        TensorMap *output_tensors = new TensorMap();
        output_tensors->insert(
            {"output_ids", Tensor{MEMORY_GPU, TYPE_INT32, {max_seq_len, batch_size, beam_width}, d_output_ids}});
        output_tensors->insert({"finished", Tensor{MEMORY_GPU, TYPE_BOOL, {batch_size * beam_width}, nullptr}});
        output_tensors->insert({"cum_log_probs", Tensor{MEMORY_GPU, TYPE_FP32, {batch_size * beam_width}, d_cum_log_probs}});
        output_tensors->insert(
            {"output_log_probs", Tensor{MEMORY_GPU, TYPE_FP32, {max_seq_len, batch_size, beam_width}, d_output_log_probs}});
        output_tensors->insert(
            {"sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size * beam_width}, nullptr}});
        return output_tensors;
    }

    void batchH2Dcpy(T *dst, T *src, size_t m, size_t n) {
        for (size_t i = 0; i < m; ++i) {
            cudaH2Dcpy(dst + i * n, src, n);
        }
    }

    bool checkResult(int *d_output_ids, std::vector<std::set<int>> &expected_ids) {
        assert(expected_ids.size() == max_seq_len * batchxbeam);
        int *h_output_ids = new int[max_seq_len * batchxbeam];
        cudaD2Hcpy(h_output_ids, d_output_ids, max_seq_len * batchxbeam);
        int failures = 0;
        for (size_t i = 0; i < max_seq_len * batchxbeam; ++i) {
            size_t s = i / batchxbeam;
            size_t b = i % batchxbeam;
            std::set<int> expts = expected_ids.at(i);
            if (expts.count(h_output_ids[i]) == 0) {
                if (failures < 10) {
                    std::stringstream ss;
                    ss << " - Fail "
                       << " (step=" << s << ", batch=" << b << ") "
                       << "actual=" << h_output_ids[i] << ", expected";
                    for (auto &expt : expts) {
                        ss << " " << expt;
                    }
                    QK_LOG_DEBUG("%s", ss.str().c_str());
                }
                ++failures;
            }
        }
        QK_LOG_DEBUG("check...%6s : failures: %d / %d",
                     failures == 0 ? "....OK" : "FAILED", failures, max_seq_len * batchxbeam);
        delete[] h_output_ids;
        return failures == 0;
    }

public:
    void runTest(std::vector<std::set<int>> expected_output_ids,
                 int *top_ks,
                 size_t top_k_size,
                 float *top_ps,
                 size_t top_p_size,
                 float *temperature,
                 float *repetition_penalty,
                 bool use_local_batch = false) {
        size_t local_batch_size = use_local_batch ? batch_size / 3 : batch_size;
        uint ite = use_local_batch ? 1 : 0;
        for (unsigned long long seed = 0; seed < max_seed; ++seed) {
            this->setup();
            size_t step = max_input_len;
            TensorMap *input_tensors = createInputTensors(
                top_ks, top_k_size, top_ps, top_p_size, temperature, repetition_penalty);
            input_tensors->insert({"step", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step}});
            input_tensors->insert({"ite", Tensor{MEMORY_CPU, TYPE_UINT32, {1}, &ite}});
            input_tensors->insert({"local_batch_size", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &local_batch_size}});

            TensorMap *output_tensors = createOutputTensors();

            dynamic_decode_layer->setup(batch_size, beam_width, input_tensors);
            for (step = max_input_len; step < max_output_len; ++step) {
                // Reset by the test value since the sampling layer internally update the logits buffer.
                batchH2Dcpy(input_tensors->at("logits").getPtr<T>(),
                            test_input_logits + step * vocab_size,
                            batchxbeam,
                            vocab_size);
                dynamic_decode_layer->forward(output_tensors, input_tensors);
            }
            bool passed = checkResult(d_output_ids, expected_output_ids);
            EXPECT_TRUE(passed) << "Failed at seed " << seed;

            if (!passed) {
                QK_LOG_ERROR("actual output ids");
                printMatrix(d_output_ids, max_seq_len, batch_size, batch_size, true);
            }

            delete output_tensors;
            delete input_tensors;
            this->teardown();
        }
    }
};

TYPED_TEST_SUITE(SamplingDecodeTest, FloatAndHalfTypes);

// TYPED_TEST(SamplingDecodeTest, TopK) {
//     // clang-format off
//     int top_k = 2;
//     std::vector<std::set<int>> expected_output_ids {
//         // batch
//         //  0       1       2       3       4       5
//         {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, // step 0
//         {4, 5}, {4, 5}, {4, 5}, {4, 5}, {4, 5}, {4, 5}, // step 1
//         {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}  // step 2
//     };
//     // clang-format on
//     this->runTest(expected_output_ids, &top_k, 1, nullptr, 0, nullptr, nullptr);
// }

TYPED_TEST(SamplingDecodeTest, TopP) {
    // clang-format off
    float top_p = 0.3;
    std::vector<std::set<int>> expected_output_ids {
        // batch
        {0}, {0}, {0}, {0}, {0}, {0}, // step 0
        {4}, {4}, {4}, {4}, {4}, {4}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}  // step 2
    };
    // clang-format on
    this->runTest(expected_output_ids, nullptr, 0, &top_p, 1, nullptr, nullptr);
}

} // namespace
