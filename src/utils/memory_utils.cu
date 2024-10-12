#include <iostream>
#include "utils/memory_utils.h"
#include <curand_kernel.h>
#include <cuda_fp16.h>

namespace space_llm {

template <typename T>
void deviceMalloc(T **ptr, size_t size, bool is_random_initialize) {
    QK_CHECK_WITH_INFO(size >= ((size_t)0), "Ask deviceMalloc size " + std::to_string(size) + "< 0 is invalid.");
    check_cuda_error(cudaMalloc((void **)(ptr), sizeof(T) * size));
    if (is_random_initialize) {
        cudaRandomUniform(*ptr, size);
    }
}

template void deviceMalloc(float **ptr, size_t size, bool is_random_initialize);
template void deviceMalloc(half **ptr, size_t size, bool is_random_initialize);
#ifdef ENABLE_BF16
template void deviceMalloc(__nv_bfloat16 **ptr, size_t size, bool is_random_initialize);
#endif
template void deviceMalloc(uint16_t **ptr, size_t size, bool is_random_initialize);
template void deviceMalloc(int **ptr, size_t size, bool is_random_initialize);
template void deviceMalloc(bool **ptr, size_t size, bool is_random_initialize);
template void deviceMalloc(char **ptr, size_t size, bool is_random_initialize);
template void deviceMalloc(int8_t **ptr, size_t size, bool is_random_initialize);
#ifdef ENABLE_FP8
template void deviceMalloc(__nv_fp8_e4m3 **ptr, size_t size, bool is_random_initialize);
#endif

template <typename T>
void deviceFree(T *&ptr) {
    if (ptr != NULL) {
        check_cuda_error(cudaFree(ptr));
        ptr = NULL;
    }
}

template void deviceFree(float *&ptr);
template void deviceFree(half *&ptr);
template void deviceFree(unsigned short *&ptr);
template void deviceFree(int *&ptr);
template void deviceFree(bool *&ptr);
template void deviceFree(char *&ptr);
template void deviceFree(int8_t *&ptr);

template <typename T>
__global__ void cuda_random_uniform_kernel(T *buffer, const size_t size, const int seq_offset) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t local_state;
    curand_init((unsigned long long int)1337, idx + seq_offset, 0, &local_state);
    for (size_t index = idx; index < size; index += blockDim.x * gridDim.x) {
        buffer[index] = (T)(curand_uniform(&local_state) * 0.2f - 0.1f);
    }
}

template <>
__global__ void cuda_random_uniform_kernel<int>(int *buffer, const size_t size, const int seq_offset) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t local_state;
    curand_init((float)1337.f, idx + seq_offset, 0, &local_state);
    for (size_t index = idx; index < size; index += blockDim.x * gridDim.x) {
        buffer[index] = curand(&local_state);
    }
}

template <>
__global__ void cuda_random_uniform_kernel<bool>(bool *buffer, const size_t size, const int seq_offset) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t local_state;
    curand_init((float)1337.f, idx + seq_offset, 0, &local_state);
    for (size_t index = idx; index < size; index += blockDim.x * gridDim.x) {
        buffer[index] = (curand(&local_state) % 2 == 0);
    }
}

template <>
__global__ void cuda_random_uniform_kernel<char>(char *buffer, const size_t size, const int seq_offset) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t local_state;
    curand_init((float)1337.f, idx + seq_offset, 0, &local_state);
    for (size_t index = idx; index < size; index += blockDim.x * gridDim.x) {
        buffer[index] = curand(&local_state) % 0xFF;
    }
}

template <typename T>
void cudaRandomUniform(T *buffer, const size_t size) {
    static int seq_offset = 0;
    cuda_random_uniform_kernel<T><<<256, 256>>>(buffer, size, seq_offset);
    seq_offset += 256 * 256;
}

template void cudaRandomUniform<float>(float *buffer, const size_t size);
template void cudaRandomUniform<half>(half *buffer, const size_t size);
template void cudaRandomUniform(int *buffer, const size_t size);
template void cudaRandomUniform(bool *buffer, const size_t size);
template void cudaRandomUniform(char *buffer, const size_t size);

template <typename T>
void cudaD2Dcpy(T *tgt, const T *src, const size_t size) {
    check_cuda_error(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyDeviceToDevice));
}

template void cudaD2Dcpy(float *tgt, const float *src, size_t size);
template void cudaD2Dcpy(half *tgt, const half *src, size_t size);
#ifdef ENABLE_BF16
template void cudaD2Dcpy(__nv_bfloat16 *tgt, const __nv_bfloat16 *src, size_t size);
#endif
template void cudaD2Dcpy(int *tgt, const int *src, size_t size);
template void cudaD2Dcpy(bool *tgt, const bool *src, size_t size);
template void cudaD2Dcpy(int8_t *tgt, const int8_t *src, size_t size);
#ifdef ENABLE_FP8
template void cudaD2Dcpy(__nv_fp8_e4m3 *tgt, const __nv_fp8_e4m3 *src, size_t size);
#endif
template void cudaD2Dcpy(unsigned long long *tgt, const unsigned long long *src, size_t size);

} // namespace space_llm
