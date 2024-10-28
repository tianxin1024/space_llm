#include "kernels/vit_kernels.h"

namespace space_llm {

template <typename T>
__global__ void add_bias_concat_clstoken_add_posembed(const T *__restrict in,        // b*h*w*n
                                                      T *__restrict out,             // b*(h*w+1)*n == b*h*w*n + b*n
                                                      const T *__restrict bias,      // n
                                                      const T *__restrict cls_token, // n
                                                      const T *__restrict pos_embed, // (h*w+1)*n == h*w*n + n
                                                      const int m,                   // b*(h*w+1) == b*h*w + b
                                                      const int n,                   // n
                                                      const int s,                   // h*w+1
                                                      bool on_top = true) {
    const int concat_row_idx = on_top ? 0 : (s - 1);
    const int offset = on_top ? 1 : 0;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        int col_idx = id % n;
        int row_idx = id / n;
        int slice_row_idx = row_idx % s;
        int slice_idx = row_idx / s;
        int idx_s = slice_row_idx * n + col_idx;
        int idx_i = (slice_row_idx - offset + slice_idx * (s - 1)) * n + col_idx;

        if (slice_row_idx == concat_row_idx) {
            out[id] = __ldg(&cls_token[col_idx]) + __ldg(&pos_embed[idx_s]);
        } else {
            out[id] = __ldg(&in[idx_i]) + __ldg(&bias[col_idx]) + __ldg(&pos_embed[idx_s]);
        }
    }
}

template <>
__global__ void add_bias_concat_clstoken_add_posembed(const half *__restrict in,        // b*h*w*n
                                                      half *__restrict out,             // b*(h*w+1)*n == b*h*w*n + b*n
                                                      const half *__restrict bias,      // n
                                                      const half *__restrict cls_token, // n
                                                      const half *__restrict pos_embed, // (h*w+1)*n == h*w*n + n
                                                      const int m,                      // b*(h*w+1) == b*h*w + b
                                                      const int n,                      // n
                                                      const int s,                      // h*w+1
                                                      bool on_top) {
    const int concat_row_idx = on_top ? 0 : (s - 1);
    const int offset = on_top ? 1 : 0;
    half2 *out_ptr = (half2 *)out;
    const half2 *in_ptr = (half2 *)in;
    const half2 *bias_ptr = (half2 *)bias;
    const half2 *token_ptr = (half2 *)cls_token;
    const half2 *embed_ptr = (half2 *)pos_embed;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        int col_idx = id % n;
        int row_idx = id / n;
        int slice_row_idx = row_idx % s;
        int slice_idx = row_idx / s;
        int idx_s = slice_row_idx * n + col_idx;
        int idx_i = (slice_row_idx - offset + slice_idx * (s - 1)) * n + col_idx;

        if (slice_row_idx == concat_row_idx) {
            half2 d1 = __ldg(&token_ptr[col_idx]);
            half2 d2 = __ldg(&embed_ptr[idx_s]);
            out_ptr[id] = __hadd2(d1, d2);
        } else {
            half2 d1 = __ldg(&in_ptr[idx_i]);
            half2 d2 = __ldg(&bias_ptr[col_idx]);
            half2 d3 = __ldg(&embed_ptr[idx_s]);
            out_ptr[id] = __hadd2(d3, __hadd2(d1, d2));
        }
    }
}

template <typename T>
void invokeAddBiasConcatClsTokenAddPosEmbed(const T *in,
                                            T *out,
                                            const T *bias,
                                            const T *cls_token,
                                            const T *pos_token,
                                            const int m,
                                            const int n,
                                            const int s,
                                            cudaStream_t stream) {
    const int data_type_factor = 4 / sizeof(T); // 1 for fp32, 2 for fp16
    dim3 block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x = m;
    } else {
        block.x = 1024;
        grid.x = (m * n + 1023) / 1024;
    }
    add_bias_concat_clstoken_add_posembed<<<grid, block, 0, stream>>>(
        in, out, bias, cls_token, pos_token, m, n / data_type_factor, s);
}

template void invokeAddBiasConcatClsTokenAddPosEmbed(const float *in,
                                                     float *out,
                                                     const float *bias,
                                                     const float *cls_token,
                                                     const float *pos_token,
                                                     const int m,
                                                     const int n,
                                                     const int s,
                                                     cudaStream_t stream);

template void invokeAddBiasConcatClsTokenAddPosEmbed(const half *in,
                                                     half *out,
                                                     const half *bias,
                                                     const half *cls_token,
                                                     const half *pos_token,
                                                     const int m,
                                                     const int n,
                                                     const int s,
                                                     cudaStream_t stream);

template <typename T>
__global__ void add_bias_add_posembed(T *__restrict out,             // b * (h*w) * n
                                      const T *__restrict bias,      // n
                                      const T *__restrict pos_embed, // (h*w) * n
                                      const int m,                   // m
                                      const int n,                   // n
                                      const int s) {                 // h*w*n

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        int b_idx = id % n;
        int p_idx = id % s;
        out[id] += __ldg(&bias[b_idx]) + __ldg(&pos_embed[p_idx]);
    }
}

template <>
__global__ void add_bias_add_posembed(half *__restrict out,             // b * (h*w) * n
                                      const half *__restrict bias,      // n
                                      const half *__restrict pos_embed, // (h*w) * n
                                      const int m,                      // m
                                      const int n,                      // n
                                      const int s) {                    // h*w*n

    half2 *out_ptr = (half2 *)out;
    const half2 *bias_ptr = (half2 *)bias;
    const half2 *embed_ptr = (half2 *)pos_embed;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        int b_idx = id % n;
        int p_idx = id % s;
        half2 d1 = __ldg(&bias_ptr[b_idx]);
        half2 d2 = __ldg(&embed_ptr[p_idx]);
        out_ptr[id] = __hadd2(out_ptr[id], __hadd2(d1, d2));
    }
}

template <typename T>
void invokeAddBiasAddPosEmbed(
    T *out, const T *bias, const T *pos_embed, const int m, const int n, const int s, cudaStream_t stream) {
    const int data_type_factor = 4 / sizeof(T); // 1 for fp32, 2 for fp16
    dim3 block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x = m;
    } else {
        block.x = 1024;
        grid.x = (m * n + 1023) / 1024;
    }

    add_bias_add_posembed<<<grid, block, 0, stream>>>(out, bias, pos_embed, m, n / data_type_factor, s);
}

template void invokeAddBiasAddPosEmbed(
    float *out, const float *bias, const float *pos_embed, const int m, const int n, const int s, cudaStream_t stream);
template void invokeAddBiasAddPosEmbed(
    half *out, const half *bias, const half *pos_embed, const int m, const int n, const int s, cudaStream_t stream);

} // namespace space_llm
