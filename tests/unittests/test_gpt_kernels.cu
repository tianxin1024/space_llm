#include <vector>
#include <random>

#include "kernels/gpt_kernels.h"
#include "utils/memory_utils.h"
#include "gtest_utils.h"

using namespace space_llm;

int test_find_context_dups();
int test_compact();

int main(int argc, char *argv[]) {
    bool all_passed = true;
    bool passed;

    // passed = test_find_context_dups() == EXIT_SUCCESS;
    // all_passed |= passed;
    // printf("%s", passed ? "." : "X");
    // if (!passed) {
    //     puts("\ntest_find_context_dups: FAILED");
    // }

    passed = test_compact() == EXIT_SUCCESS;
    all_passed |= passed;
    printf("%s", passed ? "." : "X");
    if (!passed) {
        puts("\ntest_compact: FAILED");
    }

    puts("");
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

int test_find_context_dups() {
    const size_t vec_size = 1234;
    const size_t batch_size = 8;

    // Reference to the first unique vector
    const std::vector<int> shared_contexts_ref{0, 0, 2, 3, 4, 4, 3, 3};

    // Which compact index belong to what vector
    const std::vector<int> batch_idx_to_compact_idx{0, 0, 1, 2, 3, 3, 2, 2};
    std::vector<int> batch_idx_to_compact_idx_test(batch_size);

    // Reverse map of batch_idx_to_compact_idx
    const std::vector<int> compact_idx_to_batch_idx{0, 2, 3, 4, -1, -1, -1, -1};
    std::vector<int> compact_idx_to_batch_idx_test(batch_size, -1);

    std::vector<int> input_ids;
    std::vector<int> default_vector(vec_size, 0);

    for (size_t i = 0; i < batch_size; ++i) {
        default_vector[vec_size - 1] = shared_contexts_ref[i];
        input_ids.insert(input_ids.end(), default_vector.begin(), default_vector.end());
    }

    std::vector<int> shared_contexts_test(batch_size);

    int *d_input_ids;
    int *d_shared_contexts_test;
    int *d_batch_idx_to_compact_idx;
    int *d_compact_to_batch;
    int *d_compact_size;
    cudaMalloc(&d_input_ids, batch_size * vec_size * sizeof(int));
    cudaMalloc(&d_shared_contexts_test, batch_size * sizeof(int));
    cudaMalloc(&d_batch_idx_to_compact_idx, batch_size * sizeof(int));
    cudaMalloc(&d_compact_size, sizeof(int));

    cudaH2Dcpy(d_input_ids, input_ids.data(), batch_size * vec_size);
    /*
    invokeFIndContextDups(d_shared_contexts_test,
                          d_batch_idx_to_compact_idx,
                          d_compact_to_batch,
                          d_compact_size,
                          batch_size,
                          1, // beam_width
                          vec_size);
    */

    int compact_size;
    cudaD2Hcpy(shared_contexts_test.data(), d_shared_contexts_test, batch_size);
    cudaD2Hcpy(batch_idx_to_compact_idx_test.data(), d_batch_idx_to_compact_idx, batch_size);
    cudaD2Hcpy(compact_idx_to_batch_idx_test.data(), d_compact_to_batch, batch_size);
    cudaD2Hcpy(&compact_size, d_compact_size, 1);

    cudaFree(d_input_ids);
    cudaFree(d_shared_contexts_test);

    EXPECT_TRUE(shared_contexts_test == shared_contexts_ref);
    EXPECT_TRUE(batch_idx_to_compact_idx == batch_idx_to_compact_idx_test);
    EXPECT_TRUE(compact_idx_to_batch_idx_test == compact_idx_to_batch_idx);
    EXPECT_TRUE(compact_size == 4);

    return EXIT_SUCCESS;
}

int test_compact() {
    size_t batch_size = 128;
    size_t compact_size = 5;
    size_t seq_len = 40;
    size_t hidden_dimension = 8;

    auto generator_f = std::bind(std::uniform_real_distribution<float>(-1.0, 1.0), std::mt19937());
    auto generator_i = std::bind(std::uniform_int_distribution<int>(0, 128), std::mt19937());

    // decoder_input [batch_size, seq_len, hidden_dimention] ->
    // compact_decoder_input [compact_size, seq_len, hidden_dimension]
    std::vector<float> decoder_input(batch_size * seq_len * hidden_dimension);
    std::vector<float> compact_decoder_input(compact_size * seq_len * hidden_dimension);
    std::generate(decoder_input.begin(), decoder_input.end(), generator_f);
    float *d_decoder_input, *d_compact_decoder_input;
    cudaMalloc(&d_decoder_input, decoder_input.size() * sizeof(float));
    cudaMalloc(&d_compact_decoder_input, compact_decoder_input.size() * sizeof(float));
    cudaH2Dcpy(d_decoder_input, decoder_input.data(), decoder_input.size());

    // attention_mask [batch_size, seq_len, seq_len] ->
    // compact_attention_mask [compact_size, seq_len, seq_len]
    std::vector<float> attention_mask(batch_size * seq_len * seq_len);
    std::vector<float> compact_attention_mask(compact_size * seq_len * seq_len);
    std::generate(attention_mask.begin(), attention_mask.end(), generator_f);
    float *d_attention_mask, *d_compact_attention_mask;
    cudaMalloc(&d_attention_mask, attention_mask.size() * sizeof(float));
    cudaMalloc(&d_compact_attention_mask, compact_decoder_input.size() * sizeof(float));
    cudaH2Dcpy(d_attention_mask, attention_mask.data(), attention_mask.size());

    // input_lengths [batch_size] -> compact_input_lengths [compact_size]
    std::vector<int> input_lengths(batch_size);
    std::vector<int> compact_input_lengths(compact_size);
    std::generate(input_lengths.begin(), input_lengths.end(), generator_i);
    int *d_input_lengths, *d_compact_input_lengths;
    cudaMalloc(&d_input_lengths, input_lengths.size() * sizeof(int));
    cudaMalloc(&d_compact_input_lengths, compact_input_lengths.size() * sizeof(int));
    cudaH2Dcpy(d_input_lengths, input_lengths.data(), input_lengths.size());

    // compact_idx [compact_size]
    std::vector<int> compact_idx{0, 29, 42, 44, 100};
    int *d_compact_idx;
    cudaMalloc(&d_compact_idx, compact_idx.size() * sizeof(int));
    cudaH2Dcpy(d_compact_idx, compact_idx.data(), compact_idx.size());

    invokeCompactInputs<float>(d_compact_decoder_input,
                               d_compact_attention_mask,
                               d_compact_input_lengths,
                               d_decoder_input,
                               d_attention_mask,
                               d_input_lengths,
                               d_compact_idx,
                               compact_size,
                               seq_len,
                               hidden_dimension);

    cudaD2Hcpy(compact_decoder_input.data(), d_compact_decoder_input, compact_size * seq_len * hidden_dimension);
    cudaD2Hcpy(compact_attention_mask.data(), d_compact_attention_mask, compact_size * seq_len * seq_len);
    cudaD2Hcpy(compact_input_lengths.data(), d_compact_input_lengths, compact_size);

    for (size_t i = 0; i < compact_size; ++i) {
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t h = 0; h < hidden_dimension; ++h) {
                EXPECT_TRUE(compact_decoder_input[(i * seq_len + t) * hidden_dimension + h]
                            == decoder_input[(compact_idx[i] * seq_len + t) * hidden_dimension + h]);
            }
        }
    }

    for (size_t i = 0; i < compact_size; ++i) {
        for (size_t t1 = 0; t1 < seq_len; ++t1) {
            for (size_t t2 = 0; t2 < seq_len; ++t2) {
                EXPECT_TRUE(compact_attention_mask[(i * seq_len + t1) * seq_len + t2]
                            == attention_mask[(compact_idx[i] * seq_len + t1) * seq_len + t2]);
            }
        }
    }

    for (size_t i = 0; i < compact_size; ++i) {
        EXPECT_TRUE(compact_input_lengths[i] == input_lengths[compact_idx[i]]);
    }

    cudaFree(d_decoder_input);
    cudaFree(d_compact_decoder_input);
    cudaFree(d_attention_mask);
    cudaFree(d_compact_attention_mask);
    cudaFree(d_input_lengths);
    cudaFree(d_compact_input_lengths);
    cudaFree(d_compact_idx);

    return EXIT_SUCCESS;
}
