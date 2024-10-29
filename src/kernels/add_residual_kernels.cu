#include "kernels/add_residual_kernels.h"
#include "utils/cuda_type_utils.cuh"

namespace space_llm {

template <typename T, int RESIDUAL_NUM, typename T2 = T>
__global__ void addBiasResidual(T *output,
                                const T2 *input,
                                const T *residual1,
                                const T *residual2,
                                const T *bias,
                                const float *scale_inter,
                                const float *scale_out,
                                const int m,
                                const int n) {
    const int col_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (col_index < n) {
        T bias_val = (bias == nullptr) ? (T)(0.0f) : bias[col_index];
        T in;
        if (std::is_same<T, T2>::value) {
            in = cuda_cast<T>(input[blockIdx.x * n + col_index]);
        } else {
            in = cuda_cast<float>(input[blockIdx.x * n + col_index]) * (*scale_inter) * (*scale_out);
        }

        if (RESIDUAL_NUM == 1) {
            output[blockIdx.x * n + col_index] = in + residual1[blockIdx.x * n + col_index] + bias_val;
        } else if (RESIDUAL_NUM == 2) {
            output[blockIdx.x * n + col_index] =
                in + residual1[blockIdx.x * n + col_index] + residual2[blockIdx.x * n + col_index] + bias_val;
        }
    }
}

template <typename T>
void invokeAddBiasResidual(T *output,
                           const T *input,
                           const T *residual1,
                           const T *residual2,
                           const T *bias,
                           const float *scale_inter,
                           const float *scale_out,
                           const int m,
                           const int n,
                           cudaStream_t stream) {
    QK_CHECK_WITH_INFO(!((scale_inter == nullptr) ^ (scale_out == nullptr)),
                       "Cannot use `scale_inter` without `scale_out`");
    const bool should_scale_input = scale_inter != nullptr;
    int blocks_per_row = ceil(float(n) / 1024);
    dim3 grid(m, blocks_per_row);
    dim3 block(std::min(n, 1024));
    if (residual2 == nullptr) {
        if (should_scale_input) {
            addBiasResidual<T, 1><<<grid, block, 0, stream>>>(output,
                                                              reinterpret_cast<const int32_t *>(input),
                                                              residual1,
                                                              residual2,
                                                              bias,
                                                              scale_inter,
                                                              scale_out,
                                                              m,
                                                              n);
        } else {
            addBiasResidual<T, 1><<<grid, block, 0, stream>>>(
                output, input, residual1, residual2, bias, nullptr, nullptr, m, n);
        }
    } else {
        if (should_scale_input) {
            addBiasResidual<T, 2><<<grid, block, 0, stream>>>(output,
                                                              reinterpret_cast<const int32_t *>(input),
                                                              residual1,
                                                              residual2,
                                                              bias,
                                                              scale_inter,
                                                              scale_out,
                                                              m,
                                                              n);
        } else {
            addBiasResidual<T, 2><<<grid, block, 0, stream>>>(
                output, input, residual1, residual2, bias, nullptr, nullptr, m, n);
        }
    }
}

template <typename T>
void invokeAddBiasResidual(
    T *output, const T *residual1, const T *residual2, const T *bias, const int m, const int n, cudaStream_t stream) {
    invokeAddBiasResidual(output, output, residual1, residual2, bias, nullptr, nullptr, m, n, stream);
}

#define INSTANTIATE_INVOKE_ADD_BIAS_RESIDUAL(T)                   \
    template void invokeAddBiasResidual(T *output,                \
                                        const T *input,           \
                                        const T *residual1,       \
                                        const T *residual2,       \
                                        const T *bias,            \
                                        const float *scale_inter, \
                                        const float *scale_out,   \
                                        const int m,              \
                                        const int n,              \
                                        cudaStream_t stream)

INSTANTIATE_INVOKE_ADD_BIAS_RESIDUAL(float);
INSTANTIATE_INVOKE_ADD_BIAS_RESIDUAL(half);

template void invokeAddBiasResidual(float *output,
                                    const float *residual1,
                                    const float *residual2,
                                    const float *bias,
                                    const int m,
                                    const int n,
                                    cudaStream_t stream);

template void invokeAddBiasResidual(half *output,
                                    const half *residual1,
                                    const half *residual2,
                                    const half *bias,
                                    const int m,
                                    const int n,
                                    cudaStream_t stream);

} // namespace space_llm
