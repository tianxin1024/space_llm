#include <cub/cub.cuh>
#include "kernels/sampling_topp_kernels.h"
#include "kernels/reduce_kernel_utils.cuh"
#include "utils/cuda_utils.h"

constexpr int ENABLE_SINGLE_PASS_TOP_P = 0;
constexpr float SINGLE_PASS_THRESHOLD = 0.9;

namespace space_llm {

template <typename T, int BLOCK_SIZE>
__global__ void topp_sampling(T *sorted_log_probs,
                              int *sorted_id_vals,
                              int *ids,
                              int *sequence_length,
                              bool *finished_buf,
                              float *cum_log_probs,
                              float *output_log_probs,
                              const int *begin_offset_buf,
                              const int *offset_buf,
                              const int vocab_size,
                              curandState_t *curandstate,
                              const float top_p,
                              const float *top_ps,
                              const int *end_ids,
                              const int batch_size,
                              const bool *skip_decode) {
    __shared__ int stop_shared;
    __shared__ float rand_num_s;

    const int tid = threadIdx.x;
    const int batch_id = blockIdx.x;
    if (skip_decode != nullptr && skip_decode[batch_id]) {
        return;
    }

    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const float prob_threshold = (top_ps != nullptr) ? top_ps[batch_id] : top_p;

    if (threadIdx.x == 0) {
        stop_shared = 0;
        rand_num_s = curand_uniform(curandstate + blockIdx.x) * prob_threshold;
    }

    // if begin_offset_buf and offset_buf of sorting have same value,
    // this means that we have find best one in beam_topK_kernel_for_topP
    // and skip the sorting. So, we can skip then during sampling.
    if (begin_offset_buf[batch_id] == offset_buf[batch_id]) {
        if (tid == 0) {
            int offset = batch_id * vocab_size;
            ids[batch_id] = sorted_id_vals[offset];

            if (cum_log_probs != nullptr || output_log_probs != nullptr) {
                float lprob = logf(sorted_log_probs[offset]);
                if (cum_log_probs != nullptr) {
                    cum_log_probs[batch_id] += lprob;
                }
                if (output_log_probs != nullptr) {
                    output_log_probs[batch_id] = lprob;
                }
            }
            if (sequence_length != nullptr && finished_buf != nullptr) {
                sequence_length[batch_id] =
                    finished_buf[batch_id] ? sequence_length[batch_id] : sequence_length[batch_id] + 1;
                finished_buf[batch_id] = ids[batch_id] == end_ids[batch_id] ? 1 : 0;
            }
        }
        return;
    }

    typedef cub::BlockScan<float, BLOCK_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    __shared__ uint32_t selected_shared[NUM_WARPS];
    // Initialize running total
    BlockPrefixCallbackOp prefix_op(0);

    if (lane_id == 0) {
        selected_shared[warp_id] = 0;
    }

    __syncthreads();

    int offset = batch_id * vocab_size;
    ids[batch_id] = sorted_id_vals[offset];
    int end = ((vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    int i_active = 0;
    float thread_offset = 0;
    for (int i = tid; i < end; i += BLOCK_SIZE) {
        float thread_count = (i < vocab_size) ? (float)sorted_log_probs[offset + i] : 0.f;
        BlockScan(temp_storage).InclusiveSum(thread_count, thread_offset, prefix_op);

        uint32_t active_mask = __ballot_sync(0xFFFFFFFF, rand_num_s <= thread_offset);

        i_active = i;
        if (active_mask != 0) {
            if (lane_id == 0) {
                atomicAdd(&stop_shared, 1);
                selected_shared[warp_id] = active_mask;
            }
        }
        __syncthreads();
        if (stop_shared > 0) {
            break;
        }
    };

    // select first active warp
    bool skip = (selected_shared[warp_id] > 0) ? false : true;
    for (int i = 0; i < warp_id; i++) {
        if (selected_shared[i] != 0) {
            skip = true;
        }
    }
    if (!skip) {
        int active_lane_id = WARP_SIZE - __popc(selected_shared[warp_id]);
        if (lane_id == active_lane_id) {
            ids[batch_id] = sorted_id_vals[offset + i_active];
            if (cum_log_probs != nullptr || output_log_probs != nullptr) {
                float lprob = logf(sorted_log_probs[offset + i_active]);
                if (cum_log_probs != nullptr) {
                    cum_log_probs[batch_id] += lprob;
                }
                if (output_log_probs != nullptr) {
                    output_log_probs[batch_id] = lprob;
                }
            }
            if (sequence_length != nullptr && finished_buf != nullptr) {
                sequence_length[batch_id] =
                    finished_buf[batch_id] ? sequence_length[batch_id] : sequence_length[batch_id] + 1;
                finished_buf[batch_id] = ids[batch_id] == end_ids[batch_id] ? 1 : 0;
            }
        }
    }
}

template <typename T>
void invokeBatchTopPSampling(void *workspace,
                             size_t &workspace_size,
                             size_t &cub_temp_storage_size,
                             int *output_ids,
                             int *sequence_length,
                             bool *finished_buf,
                             float *cum_log_probs,
                             float *output_log_probs,
                             const T *log_probs,
                             const int *id_vals,
                             int *offset_buf,
                             int *begin_offset_buf,
                             curandState_t *curandstate,
                             const int batch_size,
                             const size_t vocab_size_padded,
                             const int *end_ids,
                             const float max_top_p,
                             const float *top_ps,
                             cudaStream_t stream,
                             cudaDeviceProp *cuda_device_prop,
                             const bool *skip_decode) {
    // Here, we put batch size as an argument because the batch size of initialization
    // and inference may be different due to pipeline parallelism.
    const int vocab_size = vocab_size_padded;
    const int block_size = 256;

    size_t sorted_log_prob_buf_size = batch_size * vocab_size * sizeof(T);  // type T
    size_t sorted_id_vals_buf_size = batch_size * vocab_size * sizeof(int); // type int
    sorted_log_prob_buf_size = div_up(sorted_log_prob_buf_size, 256) * 256;
    sorted_id_vals_buf_size = div_up(sorted_id_vals_buf_size, 256) * 256;

    void *cub_temp_storage = workspace;
    T *sorted_log_probs = (T *)((char *)cub_temp_storage + cub_temp_storage_size);
    int *sorted_id_vals = (int *)((char *)sorted_log_probs + sorted_log_prob_buf_size);

    bool do_radix_sort = (ENABLE_SINGLE_PASS_TOP_P == 0 || max_top_p >= SINGLE_PASS_THRESHOLD);
    int smem_size = -1;

    segmented_topp_impl::TopKPerSegmentContext context;
    segmented_topp_impl::TopKPerSegmentParams params;
    segmented_topp_impl::DType_t dataTypeKind =
        (std::is_same<T, float>::value) ? segmented_topp_impl::kFLOAT : segmented_topp_impl::kHALF;

    if (!do_radix_sort) {
        QK_CHECK(cuda_device_prop != nullptr);
        memset(&context, 0, sizeof(context));
        context.sm_count = cuda_device_prop->multiProcessorCount;
        context.sm_shared_size = cuda_device_prop->sharedMemPerMultiprocessor;
        context.sm_version = cuda_device_prop->major * 100 + cuda_device_prop->minor * 10;

        memset(&params, 0, sizeof(params));
        params.gmem_src_keys = reinterpret_cast<void *>(const_cast<T *>(log_probs));
        params.gmem_dst_keys = sorted_log_probs;
        params.gmem_src_vals = reinterpret_cast<void *>(const_cast<int *>(id_vals));
        params.gmem_dst_vals = reinterpret_cast<void *>(sorted_id_vals);
        params.gmem_begin_offsets = begin_offset_buf;
        params.gmem_end_offsets = offset_buf + 1;
        params.workspace = nullptr;
        params.num_items = vocab_size * batch_size;
        params.num_segments = batch_size;
        params.top_p = max_top_p;
        params.confidence_threshold = 0.0F;

        smem_size = getSmemSizeAndCheck(context, params, dataTypeKind);
        do_radix_sort = smem_size < 0;
    }

    if (do_radix_sort) {
        if (workspace == nullptr) {
            check_cuda_error(
                cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr,
                                                                   cub_temp_storage_size,
                                                                   log_probs,
                                                                   (T *)nullptr,
                                                                   id_vals,
                                                                   (int *)nullptr,
                                                                   vocab_size * batch_size,
                                                                   batch_size,
                                                                   begin_offset_buf,
                                                                   offset_buf + 1,
                                                                   0,             // begin_bit
                                                                   sizeof(T) * 8, // end_bit = sizeof(KeyT) * 8
                                                                   stream));      // cudaStream_t
            cub_temp_storage_size = div_up(cub_temp_storage_size, 256) * 256;
            workspace_size = sorted_log_prob_buf_size + sorted_id_vals_buf_size + cub_temp_storage_size;
            return;
        }

        topp_beam_topk_kernel<T, 1, block_size><<<batch_size, block_size, 0, stream>>>(log_probs,
                                                                                       sorted_id_vals,
                                                                                       sorted_log_probs,
                                                                                       vocab_size,
                                                                                       offset_buf,
                                                                                       begin_offset_buf,
                                                                                       max_top_p,
                                                                                       top_ps,
                                                                                       skip_decode);

        check_cuda_error(
            cub::DeviceSegmentedRadixSort::SortPairsDescending(cub_temp_storage,
                                                               cub_temp_storage_size,
                                                               log_probs,
                                                               sorted_log_probs,
                                                               id_vals,
                                                               sorted_id_vals,
                                                               vocab_size * batch_size,
                                                               batch_size,
                                                               begin_offset_buf,
                                                               offset_buf + 1,
                                                               0,             // begin_bit
                                                               sizeof(T) * 8, // end_bit = sizeof(KeyT) * 8
                                                               stream));      // cudaStream_t
    } else {
        if (workspace == nullptr) {
            segmented_topp_impl::topPPerSegment(
                context, params, dataTypeKind, cub_temp_storage, cub_temp_storage_size, stream);
            workspace_size = sorted_log_prob_buf_size + sorted_id_vals_buf_size + cub_temp_storage_size;
            return;
        } else {
            topp_beam_topk_kernel<T, 1, block_size><<<batch_size, block_size, 0, stream>>>(log_probs,
                                                                                           sorted_id_vals,
                                                                                           sorted_log_probs,
                                                                                           vocab_size,
                                                                                           offset_buf,
                                                                                           begin_offset_buf,
                                                                                           max_top_p,
                                                                                           top_ps,
                                                                                           skip_decode);
            segmented_topp_impl::topPPerSegment(
                context, params, dataTypeKind, cub_temp_storage, cub_temp_storage_size, stream);
        }
    }

    constexpr int SAMPLING_BLOCK_SIZE = 256;
    dim3 grid(batch_size);
    topp_sampling<T, SAMPLING_BLOCK_SIZE><<<grid, SAMPLING_BLOCK_SIZE, 0, stream>>>(sorted_log_probs,
                                                                                    sorted_id_vals,
                                                                                    output_ids,
                                                                                    sequence_length,
                                                                                    finished_buf,
                                                                                    cum_log_probs,
                                                                                    output_log_probs,
                                                                                    begin_offset_buf,
                                                                                    offset_buf + 1,
                                                                                    vocab_size,
                                                                                    curandstate,
                                                                                    max_top_p,
                                                                                    top_ps,
                                                                                    end_ids,
                                                                                    batch_size,
                                                                                    skip_decode);
}

template void invokeBatchTopPSampling(void *workspace,
                                      size_t &workspace_size,
                                      size_t &cub_temp_storage_size,
                                      int *output_ids,
                                      int *sequence_length,
                                      bool *finished_buf,
                                      float *cum_log_probs,
                                      float *output_log_probs,
                                      const float *log_probs,
                                      const int *id_vals,
                                      int *offset_buf,
                                      int *begin_offset_buf,
                                      curandState_t *curandstate,
                                      const int batch_size,
                                      const size_t vocab_size_padded,
                                      const int *end_ids,
                                      const float max_top_p,
                                      const float *top_ps,
                                      cudaStream_t stream,
                                      cudaDeviceProp *cuda_device_prop,
                                      const bool *skip_decode);

template void invokeBatchTopPSampling(void *workspace,
                                      size_t &workspace_size,
                                      size_t &cub_temp_storage_size,
                                      int *output_ids,
                                      int *sequence_length,
                                      bool *finished_buf,
                                      float *cum_log_probs,
                                      float *output_log_probs,
                                      const half *log_probs,
                                      const int *id_vals,
                                      int *offset_buf,
                                      int *begin_offset_buf,
                                      curandState_t *curandstate,
                                      const int batch_size,
                                      const size_t vocab_size_padded,
                                      const int *end_ids,
                                      const float max_top_p,
                                      const float *top_ps,
                                      cudaStream_t stream,
                                      cudaDeviceProp *cuda_device_prop,
                                      const bool *skip_decode);

template <typename T>
void invokeTopPSampling(void *workspace,
                        size_t &workspace_size,
                        size_t &cub_temp_storage_size,
                        int *output_ids,
                        int *sequence_length,
                        bool *finished_buf,
                        float *cum_log_probs,
                        float *output_log_probs,
                        const T *log_probs,
                        const int *id_vals,
                        int *offset_buf,
                        int *begin_offset_buf,
                        curandState_t *curandstate,
                        const int batch_size,
                        const size_t vocab_size_padded,
                        const int *end_ids,
                        const float top_p,
                        cudaStream_t stream,
                        cudaDeviceProp *cuda_device_prop,
                        const bool *skip_decode) {
    invokeBatchTopPSampling(workspace,
                            workspace_size,
                            cub_temp_storage_size,
                            output_ids,
                            sequence_length,
                            finished_buf,
                            cum_log_probs,
                            output_log_probs,
                            log_probs,
                            id_vals,
                            offset_buf,
                            begin_offset_buf,
                            curandstate,
                            batch_size,
                            vocab_size_padded,
                            end_ids,
                            top_p,
                            nullptr,
                            stream,
                            cuda_device_prop,
                            skip_decode);
}

template void invokeTopPSampling(void *workspace,
                                 size_t &workspace_size,
                                 size_t &cub_temp_storage_size,
                                 int *output_ids,
                                 int *sequence_length,
                                 bool *finished_buf,
                                 float *cum_log_probs,
                                 float *output_log_probs,
                                 const float *log_probs,
                                 const int *id_vals,
                                 int *offset_buf,
                                 int *begin_offset_buf,
                                 curandState_t *curandstate,
                                 const int batch_size,
                                 const size_t vocab_size_padded,
                                 const int *end_ids,
                                 const float top_p,
                                 cudaStream_t stream,
                                 cudaDeviceProp *cuda_device_prop,
                                 const bool *skip_decode);

template void invokeTopPSampling(void *workspace,
                                 size_t &workspace_size,
                                 size_t &cub_temp_storage_size,
                                 int *output_ids,
                                 int *sequence_length,
                                 bool *finished_buf,
                                 float *cum_log_probs,
                                 float *output_log_probs,
                                 const half *log_probs,
                                 const int *id_vals,
                                 int *offset_buf,
                                 int *begin_offset_buf,
                                 curandState_t *curandstate,
                                 const int batch_size,
                                 const size_t vocab_size_padded,
                                 const int *end_ids,
                                 const float top_p,
                                 cudaStream_t stream,
                                 cudaDeviceProp *cuda_device_prop,
                                 const bool *skip_decode);

} // namespace space_llm
