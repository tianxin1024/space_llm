#include <vector>
#include <random>

#include "kernels/gpt_kernels.h"
#include "utils/memory_utils.h"
#include "gtest_utils.h"

using namespace space_llm;

int test_find_context_dups();

int main(int argc, char *argv[]) {
    bool all_passed = true;
    bool passed;

    passed = test_find_context_dups() == EXIT_SUCCESS;
    all_passed |= passed;
    printf("%s", passed ? "." : "X");
    if (!passed) {
        puts("\ntest_find_context_dups: FAILED");
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
