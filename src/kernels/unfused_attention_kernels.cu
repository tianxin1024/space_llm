#include <assert.h>
#include <cuda_fp16.h>

#include "kernels/unfused_attention_kernels.h"
#include "kernels/reduce_kernel_utils.cuh"
#include "utils/cuda_utils.h"

namespace space_llm {

__inline__ __device__ int target_index(int id1, int id2, int id3, int id4, int dim_1, int dim_2, int dim_3, int dim_4) {
    return id1 * (dim_2 * dim_3 * dim_4) + id3 * (dim_2 * dim_4) + id2 * dim_4 + id4;
}

template <typename T>
__global__ void addQKVBiasIA3Transpose(T *q_out,
                                       T *k_out,
                                       T *v_out,
                                       const T *__restrict q_in,
                                       const T *__restrict bias_q,
                                       const T *__restrict k_in,
                                       const T *__restrict bias_k,
                                       const T *__restrict v_in,
                                       const T *__restrict bias_v,
                                       const int *ia3_tasks,
                                       const T *ia3_key_weights,
                                       const T *ia3_value_weights,
                                       const int batch_size,
                                       const int seq_len,
                                       const int head_num,
                                       const int size_per_head) {
    const int n = head_num * size_per_head;
    const int batch_id = blockIdx.x;
    const int word_id = blockIdx.y;
    const int row_id = batch_id * seq_len + word_id;

    const bool use_ia3 = ia3_tasks != nullptr;
    const int ia3_task = use_ia3 ? ia3_tasks[batch_id] : 0;
    const bool use_ia3_key = use_ia3 && (ia3_key_weights != nullptr);
    const bool use_ia3_value = use_ia3 && (ia3_value_weights != nullptr);

    for (int col_id = threadIdx.x; col_id < n; col_id += blockDim.x) {
        const int head_id = col_id / size_per_head;
        const int size_id = col_id % size_per_head;
        const int target_id = batch_id * (head_num * seq_len * size_per_head) + head_id * seq_len * size_per_head
                              + word_id * size_per_head + size_id;
        const int src_id = row_id * n + col_id;

        T q = ldg(&q_in[src_id]);
        q_out[target_id] = add(q, ldg(&bias_q[col_id]));

        T k = add(ldg(&k_in[src_id]), ldg(&bias_k[col_id]));
        if (use_ia3_key) {
            k = k * ia3_key_weights[ia3_task * n + col_id];
        }
        k_out[target_id] = k;

        T v = add(ldg(&v_in[src_id]), ldg(&bias_v[col_id]));
        if (use_ia3_value) {
            v = v * ia3_value_weights[ia3_task * n + col_id];
        }
        v_out[target_id] = v;
    }
}

template <typename T>
__global__ void QKVIA3Transpose(T *q_out,
                                T *k_out,
                                T *v_out,
                                const T *__restrict q_in,
                                const T *__restrict k_in,
                                const T *__restrict v_in,
                                const int *ia3_tasks,
                                const T *__restrict ia3_key_weights,
                                const T *__restrict ia3_value_weights,
                                const int batch_size,
                                const int seq_len,
                                const int head_num,
                                const int size_per_head) {
    const int n = head_num * size_per_head;
    const int batch_id = blockIdx.x;                 // batch id
    const int word_id = blockIdx.y;                  // seq_len id
    const int row_id = batch_id * seq_len + word_id; // 行id

    const bool use_ia3 = ia3_tasks != nullptr;
    const int ia3_task = use_ia3 ? ia3_tasks[batch_id] : 0;
    const bool use_ia3_key = use_ia3 && (ia3_key_weights != nullptr);
    const bool use_ia3_value = use_ia3 && (ia3_value_weights != nullptr);

    for (int col_id = threadIdx.x; col_id < n; col_id += blockDim.x) {
        const int head_id = col_id / size_per_head;
        const int size_id = col_id % size_per_head;
        const int target_id = batch_id * (head_num * seq_len * size_per_head) + head_id * seq_len * size_per_head
                              + word_id * size_per_head + size_id;
        const int src_id = row_id * n + col_id;

        q_out[target_id] = ldg(&q_in[src_id]);

        T k = ldg(&k_in[src_id]);
        if (use_ia3_key) {
            k = k * ia3_key_weights[ia3_task * n + col_id];
        }
        k_out[target_id] = k;

        T v = ldg(&v_in[src_id]);
        if (use_ia3_value) {
            v = v * ia3_value_weights[ia3_task * n + col_id];
        }
        v_out[target_id] = v;
    }
}

template <typename T>
void invokeAddQKVBiasIA3Transpose(T *q_buf,
                                  T *k_buf,
                                  T *v_buf,
                                  T *Q,
                                  const T *bias_Q,
                                  T *K,
                                  const T *bias_K,
                                  T *V,
                                  const T *bias_V,
                                  const int batch_size,
                                  const int seq_len,
                                  const int head_num,
                                  const int size_per_head,
                                  const int *ia3_tasks,
                                  const T *ia3_key_weights,
                                  const T *ia3_value_weights,
                                  cudaStream_t stream) {
    const int k = head_num * size_per_head;
    dim3 grid(batch_size, seq_len);
    bool is_add_bias = bias_Q != nullptr;

    if (sizeof(T) == 4 || k % 2 != 0) {
        dim3 block(std::min(k, 512));
        if (is_add_bias) {
            addQKVBiasIA3Transpose<T><<<grid, block, 0, stream>>>(q_buf,
                                                                  k_buf,
                                                                  v_buf,
                                                                  Q,
                                                                  bias_Q,
                                                                  K,
                                                                  bias_K,
                                                                  V,
                                                                  bias_V,
                                                                  ia3_tasks,
                                                                  ia3_key_weights,
                                                                  ia3_value_weights,
                                                                  batch_size,
                                                                  seq_len,
                                                                  head_num,
                                                                  size_per_head);
        } else {
            QKVIA3Transpose<T><<<grid, block, 0, stream>>>(q_buf,
                                                           k_buf,
                                                           v_buf,
                                                           Q,
                                                           K,
                                                           V,
                                                           ia3_tasks,
                                                           ia3_key_weights,
                                                           ia3_value_weights,
                                                           batch_size,
                                                           seq_len,
                                                           head_num,
                                                           size_per_head);
        }
        sync_check_cuda_error();
    } else {
        using T2 = typename TypeConverter<T>::Type; // fp16 to half2, bf16 to bf162
        dim3 block(std::min(k / 2, 512));
        if (is_add_bias) {
            addQKVBiasIA3Transpose<T2><<<grid, block, 0, stream>>>((T2 *)q_buf,
                                                                   (T2 *)k_buf,
                                                                   (T2 *)v_buf,
                                                                   (const T2 *)Q,
                                                                   (const T2 *)bias_Q,
                                                                   (const T2 *)K,
                                                                   (const T2 *)bias_K,
                                                                   (const T2 *)V,
                                                                   (const T2 *)bias_V,
                                                                   ia3_tasks,
                                                                   (const T2 *)ia3_key_weights,
                                                                   (const T2 *)ia3_value_weights,
                                                                   batch_size,
                                                                   seq_len,
                                                                   head_num,
                                                                   size_per_head / 2);
        } else {
            QKVIA3Transpose<T2><<<grid, block, 0, stream>>>((T2 *)q_buf,
                                                            (T2 *)k_buf,
                                                            (T2 *)v_buf,
                                                            (const T2 *)Q,
                                                            (const T2 *)K,
                                                            (const T2 *)V,
                                                            ia3_tasks,
                                                            (const T2 *)ia3_key_weights,
                                                            (const T2 *)ia3_value_weights,
                                                            batch_size,
                                                            seq_len,
                                                            head_num,
                                                            size_per_head / 2);
        }
        sync_check_cuda_error();
    }
}

#define INSTANTIATEADDQKVBIASIA3TRANSPOSE(T)                               \
    template void invokeAddQKVBiasIA3Transpose(T *q_buf,                   \
                                               T *k_buf,                   \
                                               T *v_buf,                   \
                                               T *Q,                       \
                                               const T *bias_Q,            \
                                               T *K,                       \
                                               const T *bias_K,            \
                                               T *V,                       \
                                               const T *bias_V,            \
                                               const int batch_size,       \
                                               const int seq_len,          \
                                               const int head_num,         \
                                               const int size_per_head,    \
                                               const int *ia3_tasks,       \
                                               const T *ia3_key_weights,   \
                                               const T *ia3_value_weights, \
                                               cudaStream_t stream)
INSTANTIATEADDQKVBIASIA3TRANSPOSE(float);
INSTANTIATEADDQKVBIASIA3TRANSPOSE(half);
#ifdef ENABLE_BF16
INSTANTIATEADDQKVBIASIA3TRANSPOSE(__nv_bfloat16);
#endif
#undef INSTANTIATEADDQKVBIASIA3TRANSPOSE

template <typename T>
__global__ void rebuild_padding_ia3(const T *Q,
                                    const T *K,
                                    const T *V,
                                    T *q_buf_,
                                    T *k_buf_,
                                    T *v_buf_,
                                    const int *ia3_tasks,
                                    const T *ia3_key_weights,
                                    const T *ia3_value_weights,
                                    const int batch_size,
                                    const int seq_len,
                                    const int head_num,
                                    const int size_per_head,
                                    const int *mask_offset) {
    const int bid = blockIdx.x;

    const int tgt_batch_id = (bid + mask_offset[bid]) / seq_len;
    const int tgt_seq_id = (bid + mask_offset[bid]) % seq_len;
    const int n = head_num * size_per_head;

    const bool use_ia3 = ia3_tasks != nullptr;
    const int ia3_task = use_ia3 ? ia3_tasks[tgt_batch_id] : 0;
    const bool use_ia3_key = use_ia3 && (ia3_key_weights != nullptr);
    const bool use_ia3_value = use_ia3 && (ia3_value_weights != nullptr);

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        const int tgt_head_id = idx / size_per_head;
        const int tgt_hidden_id = idx % size_per_head;

        const int src_id = bid * n + idx;
        const int tgt_id = tgt_batch_id * head_num * seq_len * size_per_head + tgt_head_id * seq_len * size_per_head
                           + tgt_seq_id * size_per_head + tgt_hidden_id;

        q_buf_[tgt_id] = ldg(&Q[src_id]);

        T k = ldg(&K[src_id]);
        if (use_ia3_key) {
            k = k * ia3_key_weights[ia3_task * n + idx];
        }
        k_buf_[tgt_id] = k;

        T v = ldg(&V[src_id]);
        if (use_ia3_value) {
            v = v * ia3_value_weights[ia3_task * n + idx];
        }
        v_buf_[tgt_id] = v;
    }
}

template <typename T>
__global__ void add_QKV_bias_rebuild_padding_ia3(const T *Q,
                                                 const T *bias_Q,
                                                 const T *K,
                                                 const T *bias_K,
                                                 const T *V,
                                                 const T *bias_V,
                                                 T *q_buf_,
                                                 T *k_buf_,
                                                 T *v_buf_,
                                                 const int *ia3_tasks,
                                                 const T *ia3_key_weights,
                                                 const T *ia3_value_weights,
                                                 const int batch_size,
                                                 const int seq_len,
                                                 const int head_num,
                                                 const int size_per_head,
                                                 const int *mask_offset) {
    const int bid = blockIdx.x;

    const int tgt_batch_id = (bid + mask_offset[bid]) / seq_len;
    const int tgt_seq_id = (bid + mask_offset[bid]) % seq_len;
    const int n = head_num * size_per_head;

    const bool use_ia3 = ia3_tasks != nullptr;
    const int ia3_task = use_ia3 ? ia3_tasks[tgt_batch_id] : 0;
    const bool use_ia3_key = use_ia3 && (ia3_key_weights != nullptr);
    const bool use_ia3_value = use_ia3 && (ia3_value_weights != nullptr);

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        const int tgt_head_id = idx / size_per_head;
        const int tgt_hidden_id = idx % size_per_head;

        const int src_id = bid * n + idx;
        const int tgt_id = tgt_batch_id * head_num * seq_len * size_per_head + tgt_head_id * seq_len * size_per_head
                           + tgt_seq_id * size_per_head + tgt_hidden_id;

        q_buf_[tgt_id] = add(ldg(&Q[src_id]), ldg(&bias_Q[idx]));

        T k = ldg(&K[src_id]);
        if (use_ia3_key) {
            k = k * ia3_key_weights[ia3_task * n + idx];
        }
        k_buf_[tgt_id] = add(k, ldg(&bias_K[idx]));

        T v = ldg(&V[src_id]);
        if (use_ia3_value) {
            v = v * ia3_value_weights[ia3_task * n + idx];
        }
        v_buf_[tgt_id] = add(v, ldg(&bias_V[idx]));
    }
}

template <typename T>
void invokeAddQKVBiasIA3RebuildPadding(T *Q,
                                       const T *bias_Q,
                                       T *K,
                                       const T *bias_K,
                                       T *V,
                                       const T *bias_V,
                                       T *q_buf,
                                       T *k_buf,
                                       T *v_buf,
                                       const int batch_size,
                                       const int seq_len,
                                       const int head_num,
                                       const int size_per_head,
                                       const int valid_word_num,
                                       const int *mask_offset,
                                       const int *ia3_tasks,
                                       const T *ia3_key_weights,
                                       const T *ia3_value_weights,
                                       cudaStream_t stream) {
#ifdef ENABLE_BF16
    bool is_half2 = (std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value) && (size_per_head % 2 == 0);
#else
    bool is_half2 = (std::is_same<T, half>::value) && (size_per_head % 2 == 0);
#endif
    using T2 = typename TypeConverter<T>::Type; // fp16 to half2, bf16 to bf162
    int block_size = head_num * size_per_head;
    if (is_half2) {
        while (block_size > 512) {
            if (block_size % 2 == 0) {
                block_size /= 2;
            } else {
                is_half2 = false;
                block_size = std::min(block_size, 512);
                break;
            }
        }
    } else {
        block_size = std::min(block_size, 512);
    }

    if (bias_Q == nullptr && bias_K == nullptr && bias_V == nullptr) {
        if (is_half2) {
            rebuild_padding_ia3<<<valid_word_num, block_size, 0, stream>>>((T2 *)Q,
                                                                           (T2 *)K,
                                                                           (T2 *)V,
                                                                           (T2 *)q_buf,
                                                                           (T2 *)k_buf,
                                                                           (T2 *)v_buf,
                                                                           ia3_tasks,
                                                                           (const T2 *)ia3_key_weights,
                                                                           (const T2 *)ia3_value_weights,
                                                                           batch_size,
                                                                           seq_len,
                                                                           head_num,
                                                                           size_per_head / 2,
                                                                           mask_offset);
        } else {
            rebuild_padding_ia3<<<valid_word_num, block_size, 0, stream>>>(Q,
                                                                           K,
                                                                           V,
                                                                           q_buf,
                                                                           k_buf,
                                                                           v_buf,
                                                                           ia3_tasks,
                                                                           ia3_key_weights,
                                                                           ia3_value_weights,
                                                                           batch_size,
                                                                           seq_len,
                                                                           head_num,
                                                                           size_per_head,
                                                                           mask_offset);
        }
    } else if (bias_Q != nullptr && bias_K != nullptr && bias_V != nullptr) {
        if (is_half2) {
            add_QKV_bias_rebuild_padding_ia3<<<valid_word_num, block_size, 0, stream>>>((T2 *)Q,
                                                                                        (const T2 *)bias_Q,
                                                                                        (T2 *)K,
                                                                                        (const T2 *)bias_K,
                                                                                        (T2 *)V,
                                                                                        (const T2 *)bias_V,
                                                                                        (T2 *)q_buf,
                                                                                        (T2 *)k_buf,
                                                                                        (T2 *)v_buf,
                                                                                        ia3_tasks,
                                                                                        (const T2 *)ia3_key_weights,
                                                                                        (const T2 *)ia3_value_weights,
                                                                                        batch_size,
                                                                                        seq_len,
                                                                                        head_num,
                                                                                        size_per_head / 2,
                                                                                        mask_offset);
        } else {
            add_QKV_bias_rebuild_padding_ia3<<<valid_word_num, block_size, 0, stream>>>(Q,
                                                                                        bias_Q,
                                                                                        K,
                                                                                        bias_K,
                                                                                        V,
                                                                                        bias_V,
                                                                                        q_buf,
                                                                                        k_buf,
                                                                                        v_buf,
                                                                                        ia3_tasks,
                                                                                        ia3_key_weights,
                                                                                        ia3_value_weights,
                                                                                        batch_size,
                                                                                        seq_len,
                                                                                        head_num,
                                                                                        size_per_head,
                                                                                        mask_offset);
        }
    } else {
        QK_CHECK(false);
    }
}

#define INSTANTIATEADDQKVBIASIA3REBUILDPADDING(T)                               \
    template void invokeAddQKVBiasIA3RebuildPadding(T *Q,                       \
                                                    const T *bias_Q,            \
                                                    T *K,                       \
                                                    const T *bias_K,            \
                                                    T *V,                       \
                                                    const T *bias_V,            \
                                                    T *q_buf,                   \
                                                    T *k_buf,                   \
                                                    T *v_buf,                   \
                                                    const int batch_size,       \
                                                    const int seq_len,          \
                                                    const int head_num,         \
                                                    const int size_per_head,    \
                                                    const int valid_word_num,   \
                                                    const int *mask_offset,     \
                                                    const int *ia3_tasks,       \
                                                    const T *ia3_key_weights,   \
                                                    const T *ia3_value_weights, \
                                                    cudaStream_t stream)
INSTANTIATEADDQKVBIASIA3REBUILDPADDING(float);
INSTANTIATEADDQKVBIASIA3REBUILDPADDING(half);
#ifdef ENABLE_BF16
INSTANTIATEADDQKVBIASIA3REBUILDPADDING(__nv_bfloat16);
#endif
#undef INSTANTIATEADDQKVBIASIA3REBUILDPADDING

template <typename T>
__global__ void addRelativeAttentionBias(
    T *qk_buf, const T *relative_attention_bias, const int batch_size, const int head_num, const int seq_len) {
    for (int i = threadIdx.x; i < batch_size * seq_len; i += blockDim.x) {
        int batch_id = i / seq_len;
        int seq_id = i % seq_len;

        const int bias_index = blockIdx.x * seq_len + seq_id;
        const int qk_index = batch_id * gridDim.x * seq_len + bias_index;
        qk_buf[qk_index] = add(qk_buf[qk_index], relative_attention_bias[bias_index]);
    }
}

template <typename T>
void invokeAddRelativeAttentionBias(T *qk_buf,
                                    const T *relative_attention_bias,
                                    const int batch_size,
                                    const int head_num,
                                    const int seq_len,
                                    cudaStream_t stream) {
    // qk_buf: [batch_size, head_num, seq_len, seq_len]
    // relative_attention_bias: [1, head_num, seq_len, seq_len]
    dim3 grid(head_num * seq_len);
    dim3 block(512);
    using T2 = typename TypeConverter<T>::Type;

#ifdef ENABLE_BF16
    const bool is_half2 = (std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value) && (seq_len % 2 == 0);
#else
    const bool is_half2 = (std::is_same<T, half>::value) && (seq_len % 2 == 0);
#endif
    if (is_half2) {
        addRelativeAttentionBias<T2><<<grid, block, 0, stream>>>(
            (T2 *)qk_buf, (const T2 *)relative_attention_bias, batch_size, head_num, seq_len / 2);
    } else {
        addRelativeAttentionBias<<<grid, block, 0, stream>>>(
            qk_buf, relative_attention_bias, batch_size, head_num, seq_len);
    }
}

#define INSTANTIATEADDRELATIVEATTENTIONBIAS(T)                                     \
    template void invokeAddRelativeAttentionBias(T *qk_buf,                        \
                                                 const T *relative_attention_bias, \
                                                 const int batch_size,             \
                                                 const int head_num,               \
                                                 const int seq_len,                \
                                                 cudaStream_t stream)

INSTANTIATEADDRELATIVEATTENTIONBIAS(float);
INSTANTIATEADDRELATIVEATTENTIONBIAS(half);
#ifdef ENABLE_BF16
INSTANTIATEADDRELATIVEATTENTIONBIAS(__nv_bfloat16);
#endif
#undef INSTANTIATEADDRELATIVEATTENTIONBIAS

template <typename T, typename T_IN, int ITEMS_PER_THREAD>
__global__ void softmax_kernel(T *attn_score,
                               const T_IN *qk,
                               const T *attn_mask,
                               const T *linear_bias_slopes,
                               const int batch_size,
                               const int head_num,
                               const int q_length,
                               const int k_length,
                               const float qk_scale) {
    // attn_score, [batch_size, num_heads, q_length, k_length]
    // qk, [batch_size, num_heads, q_length, k_length]
    // attn_mask, [batch_size, q_length, k_length]
    // linear_bias_slopes, [num_heads]
    const int64_t bi = blockIdx.y; // Batch index.
    const int64_t hi = blockIdx.z; // Head index.

    __shared__ float s_mean, s_max;

    const float linear_bias_slope = linear_bias_slopes != nullptr ? (float)linear_bias_slopes[hi] : 0.0f;

    for (int64_t qi = blockIdx.x; qi < q_length; qi += gridDim.x) {
        float data[ITEMS_PER_THREAD];
        int64_t qk_offset;
        float local_max = -1e20f;
        // Loop along with K dimension.
        for (int64_t i = 0; blockDim.x * i + threadIdx.x < k_length; i++) {
            int64_t ki = blockDim.x * i + threadIdx.x; // Index of K dimension.
            qk_offset = ((bi * head_num + hi) * q_length + qi) * k_length + ki;

            float qk_val = static_cast<float>(qk[qk_offset]);
            float qk_bias = 0.0f;

            if (linear_bias_slopes != nullptr) {
                // We don't handle the upper diagonal (ki > qi) separately, whose values
                // are negligible due to the negative infinity mask. And it matches with
                // the HF's implementation.
                qk_bias += static_cast<float>(linear_bias_slope * (ki - qi));
            }

            int64_t mask_offset = (bi * q_length + qi) * k_length + ki;
            float mask_val = static_cast<float>(ldg(&attn_mask[mask_offset]));
            qk_bias += (1.0f - mask_val) * -10000.0f;

            data[i] = qk_scale * qk_val + qk_bias;
            local_max = fmax(local_max, data[i]);
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float local_sum = 0;
        for (int64_t i = 0; blockDim.x * i + threadIdx.x < k_length; i++) {
            data[i] = __expf(data[i] - s_max);
            local_sum += data[i];
        }

        float sum_val = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);
        if (threadIdx.x == 0) {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();

        for (int64_t i = 0; blockDim.x * i + threadIdx.x < k_length; i++) {
            qk_offset = ((bi * head_num + hi) * q_length + qi) * k_length + blockDim.x * i + threadIdx.x;
            attn_score[qk_offset] = (T)(data[i] * s_mean);
        }
    }
}

template <typename T, int K_ITEMS_PER_THREAD, int Q_ITEMS_PER_THREAD>
__global__ void softmax_kernel_h2_v2(T *attn_score,
                                     const T *qk_buf,
                                     const T *attn_mask,
                                     const T *linear_bias_slopes,
                                     const int batch_size,
                                     const int head_num,
                                     const int q_length,
                                     const int k_length,
                                     const T scalar) {
    // attn_score, [batch_size, num_heads, q_length, k_length]
    // qk, [batch_size, num_heads, q_length, k_length]
    // attn_mask, [batch_size, q_length, k_length]
    // linear_bias_slopes, [num_heads]

    using T2 = typename TypeConverter<T>::Type;

    // QK^T matrix of shape (batch_size, head_num, q_length, k_length / 2)
    T2 *attn_score_h2 = reinterpret_cast<T2 *>(attn_score);
    const T2 *qk_buf_h2 = reinterpret_cast<const T2 *>(qk_buf);
    const T2 *attn_mask_h2 = reinterpret_cast<const T2 *>(attn_mask);

    const int bi = blockIdx.y; // Batch index
    const int hi = blockIdx.z; // Head index.

    // Constant values that will be used repeately in the q/k loop.
    const T2 ONE = cuda_cast<T2>(1.0f);
    const T2 ZERO = cuda_cast<T2>(0.0f);
    const T2 NEG_INFTY = cuda_cast<T2>(-10000.0f);

    // The normalization factor of QK.
    const T2 qk_scale = cuda_cast<T2>(scalar);
    // The slope of a linear position bias of the current attention head.
    const T2 linear_bias_slope = linear_bias_slopes != nullptr ? cuda_cast<T2>(linear_bias_slopes[hi]) : ZERO;

    __shared__ float s_sum[Q_ITEMS_PER_THREAD], s_max[Q_ITEMS_PER_THREAD];

    // Loop over q dimension.
    for (int qi = blockIdx.x; qi < q_length; qi += gridDim.x * Q_ITEMS_PER_THREAD) {
        T2 data[Q_ITEMS_PER_THREAD][K_ITEMS_PER_THREAD];

        int qk_offset[Q_ITEMS_PER_THREAD];

        float local_max[Q_ITEMS_PER_THREAD];
#pragma unroll
        for (int j = 0; j < Q_ITEMS_PER_THREAD; j++) {
            local_max[j] = -1e20f;
        }

        // Loop over k dimension.
        const int Q_ITEMS = min((q_length - qi + gridDim.x - 1) / gridDim.x, Q_ITEMS_PER_THREAD);
        for (int i = 0; blockDim.x * i + threadIdx.x < k_length / 2 && i < K_ITEMS_PER_THREAD; ++i) {
            // The half of the index of k dimension. We will use the elements at {2 * ki, 2 * ki + 1}.
            int ki = blockDim.x * i + threadIdx.x;

            int mask_offset[Q_ITEMS_PER_THREAD];
#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                qk_offset[j] = ((bi * head_num + hi) * q_length + qi + j * gridDim.x) * (k_length / 2) + ki;
                mask_offset[j] = (bi * q_length + qi + j * gridDim.x) * (k_length / 2) + ki;
            }

            T2 mask_val[Q_ITEMS_PER_THREAD];
#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                mask_val[j] = ldg(&attn_mask_h2[mask_offset[j]]);
            }

            T2 qk[Q_ITEMS_PER_THREAD];
#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                qk[j] = qk_buf_h2[qk_offset[j]];
            }

            T2 pos_bias[Q_ITEMS_PER_THREAD];
            if (linear_bias_slopes != nullptr) {
#pragma unroll
                for (int j = 0; j < Q_ITEMS; j++) {
                    // The position bias depends on the distance between qi/ki and is zero if qi >= 2*ki
                    // or qi >= 2*ki+1. For T2 vectorization, we should handle every two elements along
                    // with k-dim simultaneously. To do this, we check qi / 2 > ki at ones instead of
                    // qi >= 2*ki or 2*ki+1. It works because an diagonal element for an odd qi will be
                    // zero due to slope * (qi - 2*ki+1) = 0. Thus, we don't handle the upper diagonal
                    // separately, whose values are negligible due to the negative infinity mask.
                    int qidx = qi + j * gridDim.x;
                    T2 dist(2.0f * ki - qidx, 2.0f * ki + 1 - qidx);
                    pos_bias[j] = hmul2<T2>(linear_bias_slope, dist);
                }
            }
#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                mask_val[j] = hmul2<T2>(hsub2<T2>(ONE, mask_val[j]), NEG_INFTY);
            }

#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                T2 val = hadd2<T2>(hmul2<T2>(qk_scale, qk[j]), mask_val[j]);
                if (linear_bias_slopes != nullptr) {
                    val = hadd2<T2>(val, pos_bias[j]);
                }
                data[j][i] = val;
                local_max[j] = fmax(local_max[j], fmax((float)data[j][i].x, (float)data[j][i].y));
            }
        }

        if (blockDim.x <= 32) {
            warpReduceMaxV2<float, Q_ITEMS_PER_THREAD>(local_max);
        } else {
            blockReduceMaxV2<float, Q_ITEMS_PER_THREAD>(local_max);
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int j = 0; j < Q_ITEMS_PER_THREAD; j++) {
                s_max[j] = local_max[j];
            }
        }
        __syncthreads();

        float local_sum[Q_ITEMS_PER_THREAD];
#pragma unroll
        for (int j = 0; j < Q_ITEMS_PER_THREAD; j++) {
            local_sum[j] = {0.f};
        }

        for (int i = 0; blockDim.x * i + threadIdx.x < k_length / 2 && i < K_ITEMS_PER_THREAD; ++i) {
#pragma unroll
            for (int j = 0; j < Q_ITEMS; ++j) {
                data[j][i] = hexp2<T2>(hsub2<T2>(data[j][i], cuda_cast<T2>(s_max[j])));
            }

#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                local_sum[j] += (float)(data[j][i].x + data[j][i].y);
            }
        }

        if (blockDim.x <= 32) {
            warpReduceSumV2<float, Q_ITEMS_PER_THREAD>(local_sum);
        } else {
            blockReduceSumV2<float, Q_ITEMS_PER_THREAD>(local_sum);
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int j = 0; j < Q_ITEMS_PER_THREAD; j++) {
                s_sum[j] = __fdividef(1.0f, local_sum[j] + 1e-6f);
            }
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < k_length / 2 && i < K_ITEMS_PER_THREAD; ++i) {
#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                qk_offset[j] = ((bi * head_num + hi) * q_length + qi + j * gridDim.x) * (k_length / 2) + blockDim.x * i
                               + threadIdx.x;
            }

#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                attn_score_h2[qk_offset[j]] = hmul2<T2>(data[j][i], cuda_cast<T2>(s_sum[j]));
            }
        }
    }
}

template <typename T, int ITEMS_PER_THREAD>
__global__ void softmax_kernel_h2(T *attn_score,
                                  const T *qk_buf,
                                  const T *attn_mask,
                                  const T *linear_bias_slopes,
                                  const int batch_size,
                                  const int head_num,
                                  const int q_length,
                                  const int k_length,
                                  const T qk_scale) {
    // attn_score, [batch_size, num_heads, q_length, k_length]
    // qk, [batch_size, num_heads, q_length, k_length]
    // attn_mask, [batch_size, q_length, k_length]
    // linear_bias_slopes, [num_heads]

    using T2 = typename TypeConverter<T>::Type; // half2

    T2 *attn_score_h2 = reinterpret_cast<T2 *>(attn_score);
    const T2 *qk_buf_h2 = reinterpret_cast<const T2 *>(qk_buf);
    const T2 *attn_mask_h2 = reinterpret_cast<const T2 *>(attn_mask);

    const int bi = blockIdx.y; // Batch index
    const int hi = blockIdx.z; // Head index

    __shared__ float s_mean, s_max;

    // Constant values that will be used repeately in the q/k loop.
    const T2 ONE = cuda_cast<T2>(1.0f);
    const T2 ZERO = cuda_cast<T2>(0.0f);
    const T2 NEG_INFTY = cuda_cast<T2>(-10000.0f);

    // The normalization factor of QK.
    const T2 qk_scale_h2 = cuda_cast<T2>(qk_scale);
    // The slope of a linear position bias of the current attention head.
    const T2 linear_bias_slope = linear_bias_slopes != nullptr ? cuda_cast<T2>(linear_bias_slopes[hi]) : ZERO;

    // Loop over Q dimention.
    for (int qi = blockIdx.x; qi < q_length; qi += gridDim.x) {
        T2 data[ITEMS_PER_THREAD];
        int qk_offset;
        float local_max = -1e20f;

        // Loop over K dimention
        for (int i = 0; blockDim.x * i + threadIdx.x < (k_length / 2) && i < ITEMS_PER_THREAD; i++) {
            // The half of the index of k dimention. We will use the elements at {2 * ki, 2 * ki + 1}.
            int ki = blockDim.x * i + threadIdx.x;
            qk_offset = ((bi * head_num + hi) * q_length + qi) * (k_length / 2) + ki;
            int mask_offset = (bi * q_length + qi) * (k_length / 2) + ki;

            // The value of QK^T matrix at (qi, ki).
            T2 qk = qk_buf_h2[qk_offset];
            // The bias value to the position (qi, ki) including both mask and positional bias.
            T2 qk_bias = ZERO;

            if (linear_bias_slopes != nullptr) {
                // The position bias depends on the distance between qi/ki and is zero if qi >= 2*ki
                // or qi >= 2*ki+1. For T2 vectorization, we should handle every two elements along
                // with k-dim simultaneously. To do this, we check qi / 2 > ki at ones instead of
                // qi >= 2*ki or 2*ki+1. It works because an diagonal element for an odd qi will be
                // zero due to slope * (qi - 2*ki+1) = 0. Thus, we don't handle the upper diagonal
                // separately, whose values are negligible due to the negative infinity mask.
                T2 dist(2.0f * ki - qi, 2.0f * ki + 1 - qi);
                qk_bias = hadd2<T2>(qk_bias, hmul2<T2>(linear_bias_slope, dist));
            }

            T2 mask_val = ldg(&attn_mask_h2[mask_offset]);
            qk_bias = hadd2<T2>(qk_bias, hmul2<T2>(hsub2<T2>(ONE, mask_val), NEG_INFTY));

            data[i] = hadd2<T2>(hmul2<T2>(qk, qk_scale_h2), qk_bias);
            local_max = fmax(local_max, fmax((float)data[i].x, (float)data[i].y));
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float local_sum = 0.0f;
        for (int i = 0; blockDim.x * i + threadIdx.x < (k_length / 2) && i < ITEMS_PER_THREAD; i++) {
            data[i] = hexp2<T2>(hsub2<T2>(data[i], cuda_cast<T2>(s_max)));
            local_sum += (float)(data[i].x + data[i].y);
        }

        float sum_val = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);

        if (threadIdx.x == 0) {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < (k_length / 2) && i < ITEMS_PER_THREAD; i++) {
            qk_offset = ((bi * head_num + hi) * q_length + qi) * (k_length / 2) + blockDim.x * i + threadIdx.x;
            attn_score_h2[qk_offset] = hmul2<T2>(data[i], cuda_cast<T2>(s_mean));
        }
    }
}

#define LAUNCH_MAKSED_SOFTMAX_(T_, ITEMS_PER_THREAD)                                                                  \
    block.x /= ITEMS_PER_THREAD;                                                                                      \
    block.x = (block.x + 31) / 32 * 32;                                                                               \
    assert(block.x <= 1024);                                                                                          \
    if (is_half2) {                                                                                                   \
        if (grid.x % 4 == 0) {                                                                                        \
            grid.x /= 4;                                                                                              \
            softmax_kernel_h2_v2<T_, ITEMS_PER_THREAD, 4>                                                             \
                <<<grid, block, 0, stream>>>((T_ *)param.attention_score,                                             \
                                             (const T_ *)param.qk,                                                    \
                                             (const T_ *)param.attention_mask,                                        \
                                             (const T_ *)param.linear_bias_slopes,                                    \
                                             param.batch_size,                                                        \
                                             param.num_heads,                                                         \
                                             param.q_length,                                                          \
                                             param.k_length,                                                          \
                                             (const T_)param.qk_scale);                                               \
        } else {                                                                                                      \
            softmax_kernel_h2<T_, ITEMS_PER_THREAD><<<grid, block, 0, stream>>>((T_ *)param.attention_score,          \
                                                                                (const T_ *)param.qk,                 \
                                                                                (const T_ *)param.attention_mask,     \
                                                                                (const T_ *)param.linear_bias_slopes, \
                                                                                param.batch_size,                     \
                                                                                param.num_heads,                      \
                                                                                param.q_length,                       \
                                                                                param.k_length,                       \
                                                                                (const T_)param.qk_scale);            \
        }                                                                                                             \
    } else {                                                                                                          \
        softmax_kernel<T, T_IN, ITEMS_PER_THREAD><<<grid, block, 0, stream>>>(param.attention_score,                  \
                                                                              param.qk,                               \
                                                                              param.attention_mask,                   \
                                                                              param.linear_bias_slopes,               \
                                                                              param.batch_size,                       \
                                                                              param.num_heads,                        \
                                                                              param.q_length,                         \
                                                                              param.k_length,                         \
                                                                              param.qk_scale);                        \
    }

#define LAUNCH_MAKSED_SOFTMAX(ITEMS_PER_THREAD) LAUNCH_MAKSED_SOFTMAX_(half, ITEMS_PER_THREAD)

template <typename T, typename T_IN>
void invokeMaskedSoftmax(MaskedSoftmaxParam<T, T_IN> &param, cudaStream_t stream) {
    // attention_score,    (batch_size, head_num, q_length, k_length), softmax output.
    // qk,                 (batch_size, head_num, q_length, k_length), QK^T.
    // attention_mask,     (batch_size, q_length, k_length), attention mask.
    // linear_bias_slopes, (head_num,) the slopes of the linear position bias.

    dim3 grid(param.q_length, param.batch_size, param.num_heads);
    if (param.batch_size * param.num_heads > 360) {
        grid.x = ceil(float(param.q_length) / 32.0f);
    }

    bool is_half2 = sizeof(T) == 2 && sizeof(T_IN) == 2 && param.k_length % 2 == 0;
    dim3 block((param.k_length / (is_half2 ? 2 : 1) + 31) / 32 * 32);

    if (block.x > 2048 && block.x <= 4096) {
        LAUNCH_MAKSED_SOFTMAX(4)
    } else if (block.x > 1024) {
        LAUNCH_MAKSED_SOFTMAX(2)
    } else if (block.x > 0) {
        LAUNCH_MAKSED_SOFTMAX(1)
    } else {
        QK_CHECK(param.k_length <= 4096);
    }
}

template void invokeMaskedSoftmax(MaskedSoftmaxParam<float, float> &param, cudaStream_t stream);
template void invokeMaskedSoftmax(MaskedSoftmaxParam<half, float> &param, cudaStream_t stream);
template void invokeMaskedSoftmax(MaskedSoftmaxParam<half, half> &param, cudaStream_t stream);

template <typename T>
__global__ void transpose_attentions(
    T *attentions_out, const T *attentions_in, size_t batch_size, size_t num_layers, size_t num_heads, size_t seq_len) {
    // attentions_in  shape [B, H, S, S]
    // attentions_out shape [B, L, H, S, S].
    // Note that we write the L dimension as if it was index 0.
    // In reality, the pointer has already been shifted to point to the correct layer.

    const auto batch_idx = blockIdx.x;
    const auto head_idx = blockIdx.y;

    const auto dst_offset = (batch_idx * num_layers * num_heads + head_idx) * seq_len * seq_len;
    const auto src_offset = (batch_idx * num_heads + head_idx) * seq_len * seq_len;

    for (auto x = threadIdx.x; x < seq_len * seq_len; x += blockDim.x) {
        attentions_out[dst_offset + x] = attentions_in[src_offset + x];
    }
}

template <typename T>
void invokeTransposeAttentions(Tensor &attentions_out, const Tensor &attentions_in, cudaStream_t stream) {
    const size_t batch_size = attentions_in.shape[0];
    const size_t num_heads = attentions_in.shape[1];
    const size_t seq_len = attentions_in.shape[2];
    const size_t num_layers = attentions_out.shape[1];

    const dim3 gridSize(batch_size, num_heads);
    const dim3 blockSize(512);

    transpose_attentions<<<gridSize, blockSize, 0, stream>>>(
        attentions_out.getPtr<T>(), attentions_in.getPtr<const T>(), batch_size, num_layers, num_heads, seq_len);
}

#define INSTANTIATETRANSPOSEATTENTIONS(T)       \
    template void invokeTransposeAttentions<T>( \
        Tensor & attentions_out, const Tensor &attentions_in, cudaStream_t stream)
INSTANTIATETRANSPOSEATTENTIONS(float);
INSTANTIATETRANSPOSEATTENTIONS(half);
#ifdef ENABLE_BF16
INSTANTIATETRANSPOSEATTENTIONS(__nv_bfloat16);
#endif
#undef INSTANTIATETRANSPOSEATTENTIONS

template <typename T>
__global__ void transpose(const T *src,
                          T *dst,
                          const int batch_size,
                          const int seq_len,
                          const int head_num,
                          const int size_per_head,
                          const float *scale,
                          int int8_mode) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int batch_id = tid / (head_num * seq_len * size_per_head);
    int head_id = (tid % (head_num * seq_len * size_per_head)) / (seq_len * size_per_head);
    int seq_id = (tid % (seq_len * size_per_head)) / size_per_head;
    int id = tid % size_per_head;

    int target_id = target_index(batch_id, head_id, seq_id, id, batch_size, head_num, seq_len, size_per_head);

    if (int8_mode == 2) {
        using Int8_Packed_T = typename packed_as<int8_t, num_elems<T>::value>::type;
        using Float_Packed_T = typename packed_as<float, num_elems<T>::value>::type;

        const Float_Packed_T scale_val = cuda_cast<Float_Packed_T>(*scale);
        reinterpret_cast<Int8_Packed_T *>(dst)[target_id] =
            cuda_cast<Int8_Packed_T>(cuda_cast<Float_Packed_T>(src[tid]) * scale_val);
    } else {
        dst[target_id] = src[tid];
    }
}

template <>
__global__ void transpose(const float *src,
                          float *dst,
                          const int batch_size,
                          const int seq_len,
                          const int head_num,
                          const int size_per_head,
                          const float *scale,
                          int int8_mode) {
    int batch_id = blockIdx.x / (head_num * seq_len);
    int seq_id = blockIdx.x % seq_len;
    int head_id = (blockIdx.x % (head_num * seq_len)) / seq_len;

    const int target_id = batch_id * (head_num * seq_len * size_per_head) + seq_id * head_num * size_per_head
                          + head_id * size_per_head + threadIdx.x;
    const int src_id = blockIdx.x * size_per_head + threadIdx.x;

    if (int8_mode == 2) {
        const float scale_val = *scale;
        reinterpret_cast<int8_t *>(dst)[target_id] = cuda_cast<int8_t>(src[src_id] * scale_val);
    } else {
        dst[target_id] = src[src_id];
    }
}

template <typename T>
void invokeTransposeQKV(T *dst,
                        T *src,
                        const int batch_size,
                        const int seq_len,
                        const int head_num,
                        const int size_per_head,
                        const float *scale,
                        const int int8_mode,
                        cudaStream_t stream) {
    dim3 grid, block;
    if (sizeof(T) == 2) {
        int seq_per_block = 1;
        grid.x = batch_size * head_num * seq_len / seq_per_block;
        while (seq_per_block < 4 && grid.x % 2 == 0) {
            grid.x /= 2;
            seq_per_block *= 2;
        }

        QK_CHECK(grid.x * seq_per_block == (size_t)batch_size * head_num * seq_len);

        if (seq_per_block * size_per_head % 2 == 0) {
            block.x = seq_per_block * size_per_head / 2;
            if (std::is_same<T, half>::value) {
                transpose<half2><<<grid, block, 0, stream>>>(
                    (half2 *)src, (half2 *)dst, batch_size, seq_len, head_num, size_per_head / 2, scale, int8_mode);
            }
#ifdef ENABLE_BF16
            else {
                transpose<__nv_bfloat162><<<grid, block, 0, stream>>>((__nv_bfloat162 *)src,
                                                                      (__nv_bfloat162 *)dst,
                                                                      batch_size,
                                                                      seq_len,
                                                                      head_num,
                                                                      size_per_head / 2,
                                                                      scale,
                                                                      int8_mode);
            }
#endif
        } else {
            block.x = seq_per_block * size_per_head;
            transpose<T>
                <<<grid, block, 0, stream>>>(src, dst, batch_size, seq_len, head_num, size_per_head, scale, int8_mode);
        }
    } else {
        const int seq_per_block = 1;
        grid.x = batch_size * head_num * seq_len / seq_per_block;
        block.x = seq_per_block * size_per_head;
        transpose<T>
            <<<grid, block, 0, stream>>>(src, dst, batch_size, seq_len, head_num, size_per_head, scale, int8_mode);
    }
}

#define INSTANTIATETRANSPOSEQKV(T)                            \
    template void invokeTransposeQKV(T *src,                  \
                                     T *dst,                  \
                                     const int batch_size,    \
                                     const int seq_len,       \
                                     const int head_num,      \
                                     const int size_per_head, \
                                     const float *scale,      \
                                     const int int8_mode,     \
                                     cudaStream_t stream)
INSTANTIATETRANSPOSEQKV(float);
INSTANTIATETRANSPOSEQKV(half);
#ifdef ENABLE_BF16
INSTANTIATETRANSPOSEQKV(__nv_bfloat16);
#endif
#undef INSTANTIATETRANSPOSEQKV

template <typename T>
__global__ void transpose_remove_padding(const T *src,
                                         T *dst,
                                         const int batch_size,
                                         const int seq_len,
                                         const int head_num,
                                         const int size_per_head,
                                         const int *mask_offset,
                                         const float *scale,
                                         const int int8_mode) {
    // TODO: optimize this kernel?
    // do remove_sequence_length_padding
    const int bid = blockIdx.x; // batch * seq_len or valid_word_num

    const int src_batch_id = (bid + mask_offset[bid]) / seq_len;
    const int src_seq_id = (bid + mask_offset[bid]) % seq_len;

    const int dst_seq_id = bid;

    const int src_offset_base = src_batch_id * seq_len * head_num * size_per_head + src_seq_id * size_per_head;
    const int dst_offset_base = dst_seq_id * head_num * size_per_head;

    using Int8_Packed_T = typename packed_as<int8_t, num_elems<T>::value>::type;
    using Float_Packed_T = typename packed_as<float, num_elems<T>::value>::type;
    const Float_Packed_T scale_val =
        int8_mode == 2 ? cuda_cast<Float_Packed_T>(*scale) : cuda_cast<Float_Packed_T>(0.0f);

    for (int idx = threadIdx.x; idx < head_num * size_per_head; idx += blockDim.x) {
        const int head_id = idx / size_per_head;
        const int hidden_id = idx % size_per_head;
        const T src_elem = ldg(&src[src_offset_base + head_id * seq_len * size_per_head + hidden_id]);
        if (int8_mode == 2) {
            reinterpret_cast<Int8_Packed_T *>(dst)[dst_offset_base + idx] =
                cuda_cast<Int8_Packed_T>(cuda_cast<Float_Packed_T>(src_elem) * scale_val);
        } else {
            dst[dst_offset_base + idx] = src_elem;
        }
    }
}

// clang-format off
template<typename T>
void invokeTransposeAttentionOutRemovePadding(T*           src,
                                              T*           dst,
                                              const int    valid_word_num,
                                              const int    batch_size,
                                              const int    seq_len,
                                              const int    head_num,
                                              const int    size_per_head,
                                              const int*   mask_offset,
                                              const float* scale,
                                              const int    int8_mode,
                                              cudaStream_t stream)
{
#ifdef ENABLE_BF16
    bool is_half2 = (std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value) && (size_per_head % 2 == 0);
#else
    bool is_half2 = (std::is_same<T, half>::value) && (size_per_head % 2 == 0);
#endif
    using T2       = typename TypeConverter<T>::Type;  // fp16 to half2, bf16 to bf162
    int block_size = head_num * size_per_head;
    if (is_half2) {
        while (block_size > 512) {
            if (block_size % 2 == 0) {
                block_size /= 2;
            }
            else {
                is_half2   = false;
                block_size = std::min(block_size, 1024);
                break;
            }
        }
    }
    else {
        block_size = std::min(block_size, 1024);
    }

    if (is_half2) {
        transpose_remove_padding<T2><<<valid_word_num, block_size, 0, stream>>>(
            (T2*)src, (T2*)dst, batch_size, seq_len, head_num, size_per_head / 2, mask_offset, scale, int8_mode);
    }
    else {
        transpose_remove_padding<<<valid_word_num, block_size, 0, stream>>>(
            src, dst, batch_size, seq_len, head_num, size_per_head, mask_offset, scale, int8_mode);
    }
}
// clang-format on

#define INSTANTIATETRANSPOSEATTENTIONOUTREMOVEPADDING(T)                             \
    template void invokeTransposeAttentionOutRemovePadding(T *src,                   \
                                                           T *dst,                   \
                                                           const int valid_word_num, \
                                                           const int batch_size,     \
                                                           const int seq_len,        \
                                                           const int head_num,       \
                                                           const int size_per_head,  \
                                                           const int *mask_offset,   \
                                                           const float *scale,       \
                                                           const int int8_mode,      \
                                                           cudaStream_t stream)
INSTANTIATETRANSPOSEATTENTIONOUTREMOVEPADDING(float);
INSTANTIATETRANSPOSEATTENTIONOUTREMOVEPADDING(half);
#ifdef ENABLE_BF16
INSTANTIATETRANSPOSEATTENTIONOUTREMOVEPADDING(__nv_bfloat16);
#endif
#undef INSTANTIATETRANSPOSEATTENTIONOUTREMOVEPADDING

} // namespace space_llm
