#include "kernels/stop_criteria_kernels.h"
#include "kernels/reduce_kernel_utils.cuh"
#include "utils/cuda_utils.h"

namespace space_llm {

__global__ void length_criterion(bool *finished,
                                 bool *should_stop,
                                 int *finished_sum,
                                 const uint32_t *sequence_limit_length,
                                 int batch_size,
                                 int beam_width,
                                 int step) {
    int thread_finished_count = 0;
    for (int index = threadIdx.x; index < batch_size * beam_width; index += blockDim.x) {
        const int batch_idx = index / beam_width;

        finished[index] |= step >= sequence_limit_length[batch_idx];
        thread_finished_count += finished[index] ? 1 : 0;
    }
    int block_finished_count = 0;
    if (blockDim.x <= 32) {
        block_finished_count = warpReduceSum(thread_finished_count);
    } else {
        block_finished_count = blockReduceSum(thread_finished_count);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        finished_sum[0] = block_finished_count;
    }
}

void invokeLengthCriterion(bool *finished,
                           bool *should_stop,
                           int *h_pinned_finished_sum_,
                           const uint32_t *sequence_limit_length,
                           int batch_size,
                           int beam_width,
                           int step,
                           cudaStream_t stream) {
    // Check if we have attained the sequence length limit. If so, stop the sequence.
    // In addition, check if all sequences are stopped and return the result in should_stop
    dim3 block(std::min(512, static_cast<int>(batch_size * beam_width)));
    dim3 grid{1};
    h_pinned_finished_sum_[0] = -1;

    length_criterion<<<grid, block, 0, stream>>>(
        finished, should_stop, h_pinned_finished_sum_, sequence_limit_length, batch_size, beam_width, step);
    while (((volatile int *)h_pinned_finished_sum_)[0] == -1) {};
    sync_check_cuda_error();

    *should_stop = h_pinned_finished_sum_[0] == batch_size * beam_width;
}

} // namespace space_llm
