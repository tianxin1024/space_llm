#include "kernels/sampling_topk_kernels.h"
#include "kernels/reduce_kernel_utils.cuh"

namespace space_llm {

__global__ void curandInitialize(curandState_t *state, const int size, const unsigned long long random_seed) {
    if (blockIdx.x * blockDim.x + threadIdx.x < size) {
        curand_init(random_seed, 0, 0, &state[blockIdx.x * blockDim.x + threadIdx.x]);
    }
}

void invokeCurandInitialize(curandState_t *state,
                            const size_t batch_size,
                            unsigned long long random_seed,
                            cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((int)(ceil(batch_size * 1.0 / 256)));
    curandInitialize<<<grid, block, 0, stream>>>(state, batch_size, random_seed);
}

#define CASE_K(K_MIN, K_MAX, BLOCK_SIZE_1_, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_)                                       \
    case K_MIN ... K_MAX:                                                                                          \
        top_stage1<T, BLOCK_SIZE_1_, BLOCKS_PER_BEAM_>                                                             \
            <<<batch_size * BLOCKS_PER_BEAM_, BLOCK_SIZE_1_, 0, stream>>>(log_probs,                               \
                                                                          temp_log_probs,                          \
                                                                          topk_tmp_id_buf,                         \
                                                                          topk_tmp_val_buf,                        \
                                                                          finished,                                \
                                                                          max_top_k,                               \
                                                                          top_ks,                                  \
                                                                          vocab_size,                              \
                                                                          end_ids,                                 \
                                                                          skip_decode);                            \
        topk_stage2_sampling<T, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_>                                                   \
            <<<batch_size, BLOCK_SIZE_2_, K_MAX * sizeof(int) + K_MAX * sizeof(float), stream>>>(topk_tmp_id_buf,  \
                                                                                                 topk_tmp_val_buf, \
                                                                                                 ids,              \
                                                                                                 sequence_length,  \
                                                                                                 finished,         \
                                                                                                 cum_log_probs,    \
                                                                                                 output_log_probs, \
                                                                                                 max_top_k,        \
                                                                                                 top_ks,           \
                                                                                                 top_p,            \
                                                                                                 top_ps,           \
                                                                                                 curandstate,      \
                                                                                                 end_ids,          \
                                                                                                 vocab_size,       \
                                                                                                 skip_decode);

template <typename T>
void invokeBatchTopKSampling(void *workspace,
                             size_t &workspace_size,
                             const T *log_probs,
                             int *ids,
                             int *sequence_length,
                             bool *finished,
                             float *cum_log_probs,
                             float *output_log_probs,
                             curandState_t *curandstate,
                             const int max_top_k,
                             const int *top_ks,
                             const float top_p,
                             const float *top_ps,
                             const int vocab_size_padded,
                             const int *end_ids,
                             cudaStream_t stream,
                             const int batch_size,
                             const bool *skip_decode) {
    // Not allow an ambiguous inputs top_p and top_ps.
    assert(top_p == 1.0f || top_ps == nullptr);
    const int vocab_size = vocab_size_padded;
    const int max_block_per_beam = 8;
    int temp_log_probs_buf_size = batch_size * vocab_size;                   // type float
    int topk_tmp_ids_buf_size = batch_size * max_top_k * max_block_per_beam; // type int
    int topk_tmp_val_buf_size = batch_size * max_top_k * max_block_per_beam; // type float

    // prevent memory misaligned address
    temp_log_probs_buf_size = (int)(ceil(temp_log_probs_buf_size / 4.)) * 4;
    topk_tmp_ids_buf_size = (int)(ceil(topk_tmp_ids_buf_size / 4.)) * 4;
    topk_tmp_val_buf_size = (int)(ceil(topk_tmp_val_buf_size / 4.)) * 4;

    if (workspace == nullptr) {
        workspace_size = sizeof(T) * temp_log_probs_buf_size + sizeof(int) * topk_tmp_ids_buf_size
                         + sizeof(T) * topk_tmp_val_buf_size;
        return;
    }

    T *temp_log_probs = (T *)workspace;
    int *topk_tmp_id_buf = (int *)(temp_log_probs + temp_log_probs_buf_size);
    T *topk_tmp_val_buf = (T *)(topk_tmp_id_buf + topk_tmp_ids_buf_size);

    switch (max_top_k) {
        CASE_K(1, 16, 128, 128, 8);
        CASE_K(17, 32, 256, 128, 8);
        CASE_K(33, 64, 256, 256, 8);
        CASE_K(65, 1024, 256, 256, 8);
    default:
        throw std::domain_error(fmtstr("top-k kernel supports 1<=k<=1024 but got k=%d", max_top_k));
    }
}

#undef CASE_K

template void invokeBatchTopKSampling(void *workspace,
                                      size_t &workspace_size,
                                      const float *log_probs,
                                      int *ids,
                                      int *sequence_length,
                                      bool *finished_buf,
                                      float *cum_log_probs,
                                      float *output_log_probs,
                                      curandState_t *curandstate,
                                      const int max_top_k,
                                      const int *top_ks,
                                      const float top_p,
                                      const float *top_ps,
                                      const int vocab_size_padded,
                                      const int *end_ids,
                                      cudaStream_t stream,
                                      const int batch_size,
                                      const bool *skip_decode);

template void invokeBatchTopKSampling(void *workspace,
                                      size_t &workspace_size,
                                      const half *log_probs,
                                      int *ids,
                                      int *sequence_length,
                                      bool *finished_buf,
                                      float *cum_log_probs,
                                      float *output_log_probs,
                                      curandState_t *curandstate,
                                      const int max_top_k,
                                      const int *top_ks,
                                      const float top_p,
                                      const float *top_ps,
                                      const int vocab_size_padded,
                                      const int *end_ids,
                                      cudaStream_t stream,
                                      const int batch_size,
                                      const bool *skip_decode);

template <typename T>
void invokeTopKSampling(void *workspace,
                        size_t &workspace_size,
                        const T *log_probs,
                        int *ids,
                        int *sequence_length,
                        bool *finished_buf,
                        float *cum_log_probs,
                        float *output_log_probs,
                        curandState_t *curandstate,
                        const int top_k,
                        const float top_p,
                        const int vocab_size_padded,
                        const int *end_ids,
                        cudaStream_t stream,
                        const int batch_size,
                        const bool *skip_decode) {
    invokeBatchTopKSampling(workspace,
                            workspace_size,
                            log_probs,
                            ids,
                            sequence_length,
                            finished_buf,
                            cum_log_probs,
                            output_log_probs,
                            curandstate,
                            top_k,
                            nullptr,
                            top_p,
                            nullptr,
                            vocab_size_padded,
                            end_ids,
                            stream,
                            batch_size,
                            skip_decode);
}

template void invokeTopKSampling(void *workspace,
                                 size_t &workspace_size,
                                 const float *log_probs,
                                 int *ids,
                                 int *sequence_length,
                                 bool *finished_buf,
                                 float *cum_log_probs,
                                 float *output_log_probs,
                                 curandState_t *curandstate,
                                 const int top_k,
                                 const float top_p,
                                 const int vocab_size_padded,
                                 const int *end_ids,
                                 cudaStream_t stream,
                                 const int batch_size,
                                 const bool *skip_decode);

template void invokeTopKSampling(void *workspace,
                                 size_t &workspace_size,
                                 const half *log_probs,
                                 int *ids,
                                 int *sequence_length,
                                 bool *finished_buf,
                                 float *cum_log_probs,
                                 float *output_log_probs,
                                 curandState_t *curandstate,
                                 const int top_k,
                                 const float top_p,
                                 const int vocab_size_padded,
                                 const int *end_ids,
                                 cudaStream_t stream,
                                 const int batch_size,
                                 const bool *skip_decode);

} // namespace space_llm
