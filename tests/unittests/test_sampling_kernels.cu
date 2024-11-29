#include <algorithm> // std::fill_n
#include <iostream>  // snprintf
#include <math.h>    // expf, log
#include <stdlib.h>  // rand
#include <string>    // std::string
#include <vector>    // std::vector

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <gtest/gtest.h>

#include "utils/cuda_utils.h"
#include "utils/tensor.h"
#include "gtest_utils.h"

#include "kernels/sampling_topk_kernels.h"
#include "kernels/sampling_topp_kernels.h"

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

/////////////////////////////////// Tests //////////////////////////////////////////

template <typename T>
void computeProb(T *probs, T *logits, int batch_size, int vocab_size) {
    // Compute the log probability from logits
    //      logits = batch_size * vocab_size
    //      probs = softmax(logits) (softmax along with vocab dimension)
    // float is used for either T=float ot half, since operations of half are
    // not fully supported in a host function.
    for (int bidx = 0; bidx < batch_size; ++bidx) {
        float maxval = -FLT_MAX;
        for (int i = 0; i < vocab_size; ++i) {
            float logit = static_cast<float>(logits[bidx * vocab_size + i]);
            if (logit > maxval) {
                maxval = logit;
            }
        }
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            sum += expf(static_cast<float>(logits[bidx * vocab_size + i]) - maxval);
        }
        for (int i = 0; i < vocab_size; ++i) {
            int idx = bidx * vocab_size + i;
            float logit = static_cast<float>(logits[idx]) - maxval;
            probs[idx] = static_cast<T>(expf(logit) / (sum + EPSILON));
        }
    }
}

template <typename T>
void computeLogProb(T *logprobs, T *logits, int batch_size, int vocab_size) {
    // Compute the log probability from logits
    //      logits = batch_size * vocab_size
    //      probs = softmax(logits) (softmax along with vocab dimension)
    // float is used for either T=float ot half, since operations of half are
    // not fully supported in a host function.
    for (int bidx = 0; bidx < batch_size; ++bidx) {
        float maxval = -FLT_MAX;
        for (int i = 0; i < vocab_size; ++i) {
            float logit = static_cast<float>(logits[bidx + vocab_size + i]);
            if (logit > maxval) {
                maxval = logit;
            }
        }

        float sum = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            sum += expf(static_cast<float>(logits[bidx * vocab_size + i]) - maxval);
        }
        for (int i = 0; i < vocab_size; ++i) {
            int idx = bidx * vocab_size + i;
            float logit = static_cast<float>(logits[idx]) - maxval;
            logprobs[idx] = static_cast<T>(logit - logf(sum + EPSILON));
        }
    }
}

template <typename T>
class SamplingKernelTest : public testing::Test {
public:
    void SetUp() override {
        check_cuda_error(cudaStreamCreate(&stream));
        allocator = new Allocator(getDevice());
        allocator->setStream(stream);
    }

    void TearDown() override {
        delete allocator;
        check_cuda_error(cudaStreamDestroy(stream));
    }

protected:
    unsigned long long seed = 0;
    cudaStream_t stream;
    Allocator *allocator;
    curandState_t *curand_states;
};

template <typename T>
class TopPSamplingKernelTest : public SamplingKernelTest<T> {
protected:
    const int end_id = 0;
    using SamplingKernelTest<T>::seed;
    using SamplingKernelTest<T>::stream;
    using SamplingKernelTest<T>::allocator;
    using SamplingKernelTest<T>::curand_states;

public:
    void runTest(SamplingKernelTestParam param) {
        size_t batch_size = param.batch_size;
        size_t vocab_size = param.vocab_size;
        size_t output_len = param.output_len;
        size_t seq_len = output_len;

        float top_p = param.top_p;

        // Logit values in the host of shape (batch_size x vocab_size).
        T *h_logits = new T[batch_size * vocab_size];
        T *h_probs = new T[batch_size * vocab_size];
        T *h_lprobs = new T[batch_size * vocab_size];

        float *expected_cum_lprobs = new float[batch_size];
        std::fill_n(expected_cum_lprobs, batch_size, 0);

        int *h_output_ids = new int[batch_size];
        int *h_seq_lengths = new int[batch_size];
        bool *h_finished = new bool[batch_size];

        initRandom(h_logits, batch_size * vocab_size, -3.0f, 3.0f);

        int device;
        cudaGetDevice(&device);
        struct cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, device);

        curandState_t *curand_states = reinterpret_cast<curandState_t *>(
            allocator->malloc(sizeof(curandState_t) * batch_size, false));
        invokeCurandInitialize(curand_states, batch_size, seed, stream);

        int *end_ids = reinterpret_cast<int *>(allocator->malloc(sizeof(int) * batch_size));
        int *seq_lengths = reinterpret_cast<int *>(allocator->malloc(sizeof(int) * batch_size));
        int *output_ids = reinterpret_cast<int *>(allocator->malloc(sizeof(int) * seq_len * batch_size));

        bool *finished = reinterpret_cast<bool *>(allocator->malloc(sizeof(bool) * batch_size));
        bool *skip_decode = reinterpret_cast<bool *>(allocator->malloc(sizeof(bool) * batch_size));

        T *probs = reinterpret_cast<T *>(allocator->malloc(sizeof(T) * batch_size * vocab_size));
        float *cum_lprobs = reinterpret_cast<float *>(allocator->malloc(sizeof(float) * batch_size));
        float *output_lprobs = reinterpret_cast<float *>(allocator->malloc(sizeof(float) * output_len * batch_size));

        int *begin_offsets = reinterpret_cast<int *>(allocator->malloc(sizeof(int) * (batch_size + 1)));
        int *end_offsets = reinterpret_cast<int *>(allocator->malloc(sizeof(int) * (batch_size + 1)));
        int *topp_id_vals_buf = reinterpret_cast<int *>(allocator->malloc(sizeof(int) * batch_size * vocab_size));

        size_t workspace_size = 0;
        size_t cub_temp_storage_size = 0;
        // retrieve the workspace size of the top-p sampling kernel.
        invokeTopPSampling<T>(nullptr, // workspace
                              workspace_size,
                              cub_temp_storage_size,
                              nullptr,      // output_ids
                              nullptr,      // sequence_length
                              nullptr,      // finished_buffer
                              nullptr,      // cum_log_probs
                              nullptr,      // output_log_probs
                              (T *)nullptr, // log_probs
                              topp_id_vals_buf,
                              end_offsets,
                              begin_offsets,
                              curand_states,
                              batch_size,
                              vocab_size,
                              nullptr,
                              top_p,
                              stream,
                              &device_prop,
                              nullptr);
        void *workspace = allocator->malloc(workspace_size);

        // Initialize.
        deviceFill(end_ids, batch_size, end_id);
        deviceFill(seq_lengths, batch_size, 0);
        deviceFill(finished, batch_size, false);
        deviceFill(cum_lprobs, batch_size, 0.0f);
        deviceFill(output_lprobs, output_len * batch_size, 0.0f);
        deviceFill(output_ids, seq_len * batch_size, 0);

        for (size_t step = 0; step < output_len; ++step) {
            initRandom(h_logits, batch_size * vocab_size, -3.0f, 3.0f);
            computeProb(h_probs, h_logits, batch_size, vocab_size);
            cudaH2Dcpy(probs, h_probs, batch_size * vocab_size);

            invokeTopPInitialize(topp_id_vals_buf,
                                 end_offsets,
                                 begin_offsets,
                                 batch_size,
                                 vocab_size,
                                 stream);

            invokeTopPSampling<T>(workspace,
                                  workspace_size,
                                  cub_temp_storage_size,
                                  output_ids + step * batch_size,
                                  seq_lengths,
                                  finished,
                                  cum_lprobs,
                                  output_lprobs + step * batch_size,
                                  probs,
                                  topp_id_vals_buf,
                                  end_offsets,
                                  begin_offsets,
                                  curand_states,
                                  batch_size,
                                  vocab_size,
                                  end_ids,
                                  top_p,
                                  stream,
                                  &device_prop,
                                  nullptr);

            // Compute reference
            cudaD2Hcpy(h_output_ids, output_ids + step * batch_size, batch_size);
            cudaD2Hcpy(h_seq_lengths, seq_lengths, batch_size);
            cudaD2Hcpy(h_finished, finished, batch_size);
            computeLogProb(h_lprobs, h_logits, batch_size, vocab_size);
            for (size_t i = 0; i < batch_size; ++i) {
                int idx = i * vocab_size + h_output_ids[i];
                expected_cum_lprobs[i] += (int)step < h_seq_lengths[i] ? (float)h_lprobs[idx] : 0.0f;
                EXPECT_EQ(h_finished[i], h_output_ids[i] == end_id);
            }
        }
        bool passed = checkResult(param.toString(), cum_lprobs, expected_cum_lprobs, batch_size);
        EXPECT_TRUE(passed);

        delete[] expected_cum_lprobs;
        delete[] h_seq_lengths;
        delete[] h_logits;
        delete[] h_lprobs;
        delete[] h_probs;
        delete[] h_output_ids;
    }
};

TYPED_TEST_SUITE(TopPSamplingKernelTest, FloatAndHalfTypes);

TYPED_TEST(TopPSamplingKernelTest, CorrectnessSmallP) {
    this->runTest({6, 4, 1, 0, 0.2f, 1});
};

TYPED_TEST(TopPSamplingKernelTest, CorrectnessLargeP) {
    this->runTest({6, 4, 1, 0, 0.9f, 1});
};

TYPED_TEST(TopPSamplingKernelTest, CorrectnessAncestral) {
    this->runTest({6, 4, 1, 0, 1.0f, 1});
};

TYPED_TEST(TopPSamplingKernelTest, CorrectnessLargeVocabSmallP) {
    this->runTest({32, 51200, 1, 0, 0.2f, 16});
};

TYPED_TEST(TopPSamplingKernelTest, CorrectnessLargeVocabLargeP) {
    this->runTest({32, 51200, 1, 0, 0.9f, 16});
};

} // end of namespace
