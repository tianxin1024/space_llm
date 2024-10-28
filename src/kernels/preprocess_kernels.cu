#include "kernels/preprocess_kernels.h"
#include "utils/cuda_utils.h"

namespace space_llm {

__global__ void getPaddingOffsetAndCuSeqLensKernel(size_t *h_valid_word_num,
                                                   int *tmp_mask_offset,
                                                   int *cu_seqlens,
                                                   const int *sequence_length,
                                                   const int batch_size,
                                                   const int max_seq_len) {
    // do cumulated sum
    int total_seq_len = 0;
    int cum_offset = 0;
    int index = 0;
    const bool calculate_cu_seqlens = cu_seqlens != nullptr;
    for (int i = 0; i < batch_size; ++i) {
        const int seq_len = sequence_length[i];
        if (calculate_cu_seqlens) {
            cu_seqlens[i] = total_seq_len;
        }
        for (int j = 0; j < seq_len; ++j) {
            tmp_mask_offset[index] = cum_offset;
            index++;
        }
        cum_offset += max_seq_len - seq_len;
        total_seq_len += seq_len;
    }
    if (calculate_cu_seqlens) {
        cu_seqlens[batch_size] = total_seq_len;
    }
    h_valid_word_num[0] = (size_t)total_seq_len;
}

void invokeGetPaddingOffsetAndCuSeqLens(size_t *h_pinned_token_num,
                                        size_t *h_token_num,
                                        int *tmp_mask_offset,
                                        int *cu_seqlens,
                                        const int *sequence_lengths,
                                        const int batch_size,
                                        const int max_seq_len,
                                        cudaStream_t stream) {
    h_pinned_token_num[0] = 0;
    getPaddingOffsetAndCuSeqLensKernel<<<1, 1, 0, stream>>>(
        h_pinned_token_num, tmp_mask_offset, cu_seqlens, sequence_lengths, batch_size, max_seq_len);
    while (((volatile size_t *)h_pinned_token_num)[0] == 0) {}
    sync_check_cuda_error();
}

__global__ void getTrtPaddingOffsetKernel(int *trt_mha_padding_offset, const int *sequence_length, const int batch_size) {
    // use for get tensorrt fused mha padding offset
    // when we remove the padding
    extern __shared__ int tmp_offset[];
    if (threadIdx.x == 0) {
        tmp_offset[0] = 0;
        for (int i = 0; i < batch_size; ++i) {
            tmp_offset[i + 1] = tmp_offset[i] + sequence_length[i];
        }
    }

    __syncthreads();

    for (int i = threadIdx.x; i < batch_size + 1; i += blockDim.x) {
        trt_mha_padding_offset[i] = tmp_offset[i];
    }
}

void invokeGetTrtPaddingOffset(int *trt_mha_padding_offset,
                               const int *sequence_length,
                               const int batch_size,
                               cudaStream_t stream) {
    getTrtPaddingOffsetKernel<<<1, 256, sizeof(int) * (batch_size + 1), stream>>>(
        trt_mha_padding_offset, sequence_length, batch_size);
}

__global__ void getTrtPaddingOffsetKernel(int *trt_mha_padding_offset,
                                          const int *sequence_length,
                                          const int request_batch_size,
                                          const int request_seq_len) {
    // use for get tensorrt fused mha padding offset
    // when we keep the padding

    extern __shared__ int tmp_offset[];
    if (threadIdx.x == 0) {
        tmp_offset[0] = 0;
        for (int i = 0; i < request_batch_size; i++) {
            tmp_offset[i * 2 + 1] = tmp_offset[i * 2] + sequence_length[i];
            tmp_offset[i * 2 + 2] = request_seq_len * (i + 1);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 2 * request_batch_size + 1; i += blockDim.x) {
        trt_mha_padding_offset[i] = tmp_offset[i];
    }
}

void invokeGetTrtPaddingOffset(int *trt_mha_padding_offset,
                               const int *sequence_length,
                               const int request_batch_size,
                               const int request_seq_len,
                               cudaStream_t stream) {
    getTrtPaddingOffsetKernel<<<1, 256, sizeof(int) * (2 * request_batch_size + 1), stream>>>(
        trt_mha_padding_offset, sequence_length, request_batch_size, request_seq_len);
}

template <typename T>
__global__ void rebuild_sequence_length_padding(const T *src, T *dst, const int *padding_offset, const int n) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int dst_seq_id = bid + padding_offset[bid];
    const int src_seq_id = bid;

    for (int i = tid; i < n; i += blockDim.x) {
        dst[dst_seq_id * n + i] = src[src_seq_id * n + i];
    }
}

template <typename T>
void invokeRebuildPadding(
    T *dst, const T *src, const int *padding_offset, const int token_num, const int hidden_dim, cudaStream_t stream) {
    // src: [token_num, hidden_dim]
    // dst: [batch_size * max_seq_len, hidden_dim]
    rebuild_sequence_length_padding<<<token_num, 256, 0, stream>>>(src, dst, padding_offset, hidden_dim);
}

} // namespace space_llm
