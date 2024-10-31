#include <iostream>
#include "utils/memory_utils.h"
#include "utils/cuda_type_utils.cuh"
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
void cudaD2Hcpy(T *tgt, const T *src, const size_t size) {
    check_cuda_error(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyDeviceToHost));
}

template void cudaD2Hcpy(float *tgt, const float *src, size_t size);
template void cudaD2Hcpy(half *tgt, const half *src, size_t size);
#ifdef ENABLE_BF16
template void cudaD2Hcpy(__nv_bfloat16 *tgt, const __nv_bfloat16 *src, size_t size);
#endif
template void cudaD2Hcpy(int *tgt, const int *src, size_t size);
template void cudaD2Hcpy(bool *tgt, const bool *src, size_t size);
#ifdef ENABLE_FP8
template void cudaD2Hcpy(__nv_fp8_e4m3 *tgt, const __nv_fp8_e4m3 *src, size_t size);
#endif
template void cudaD2Hcpy(unsigned long long *tgt, const unsigned long long *src, size_t size);
template void cudaD2Hcpy(unsigned int *tgt, const unsigned int *src, size_t size);
template void cudaD2Hcpy(int8_t *tgt, const int8_t *src, size_t size);

template <typename T>
void cudaH2Dcpy(T *tgt, const T *src, const size_t size) {
    check_cuda_error(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyHostToDevice));
}

template void cudaH2Dcpy(float *tgt, const float *src, size_t size);
template void cudaH2Dcpy(half *tgt, const half *src, size_t size);
#ifdef ENABLE_BF16
template void cudaH2Dcpy(__nv_bfloat16 *tgt, const __nv_bfloat16 *src, size_t size);
#endif
template void cudaH2Dcpy(int *tgt, const int *src, size_t size);
template void cudaH2Dcpy(bool *tgt, const bool *src, size_t size);
#ifdef ENABLE_FP8
template void cudaH2Dcpy(__nv_fp8_e4m3 *tgt, const __nv_fp8_e4m3 *src, size_t size);
#endif
template void cudaH2Dcpy(unsigned long long *tgt, const unsigned long long *src, size_t size);
template void cudaH2Dcpy(unsigned int *tgt, const unsigned int *src, size_t size);
template void cudaH2Dcpy(int8_t *tgt, const int8_t *src, size_t size);

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

template <typename T>
void cudaAutoCpy(T *tgt, const T *src, const size_t size, cudaStream_t stream) {
    if (stream != NULL) {
        check_cuda_error(cudaMemcpyAsync(tgt, src, sizeof(T) * size, cudaMemcpyDefault, stream));
    } else {
        check_cuda_error(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyDefault));
    }
}

template void cudaAutoCpy(float *tgt, const float *src, size_t size, cudaStream_t stream);
template void cudaAutoCpy(half *tgt, const half *src, size_t size, cudaStream_t stream);
#ifdef ENABLE_BF16
template void cudaAutoCpy(__nv_bfloat16 *tgt, const __nv_bfloat16 *src, size_t size, cudaStream_t stream);
#endif
template void cudaAutoCpy(int *tgt, const int *src, size_t size, cudaStream_t stream);
template void cudaAutoCpy(bool *tgt, const bool *src, size_t size, cudaStream_t stream);
template void cudaAutoCpy(int8_t *tgt, const int8_t *src, size_t size, cudaStream_t stream);
template void cudaAutoCpy(uint *tgt, const uint *src, size_t size, cudaStream_t stream);
template void cudaAutoCpy(unsigned long long *tgt, const unsigned long long *src, size_t size, cudaStream_t stream);
template void cudaAutoCpy(char *tgt, const char *src, size_t size, cudaStream_t stream);

// loads data from binary file. If it succeeds, returns a non-empty vector. If loading fails or
// the product of the elements in shape is 0, this function will return an empty vector.
template <typename T>
std::vector<T> loadWeightFromBinHelper(std::vector<size_t> shape, std::string filename) {
    if (shape.size() > 2) {
        printf("[ERROR] shape should have less than two dims \n");
        return std::vector<T>();
    }

    size_t dim0 = shape[0], dim1 = 1;
    if (shape.size() == 2) {
        dim1 = shape[1];
    }
    size_t size = dim0 * dim1;
    if (size == 0) {
        QK_LOG_WARNING("shape is zero, skip loading weight from file %s \n", filename.c_str());
        return std::vector<T>();
    }

    std::vector<T> host_array(size);
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        QK_LOG_WARNING("file %s cannot be opened, loading model fails! \n", filename.c_str());
        return std::vector<T>();
    }

    size_t loaded_data_size = sizeof(T) * size;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);

    QK_LOG_DEBUG("Read " + std::to_string(loaded_data_size) + " bytes from " + filename);
    in.read((char *)host_array.data(), loaded_data_size);

    size_t in_get_size = in.gcount();
    if (in_get_size != loaded_data_size) {
        QK_LOG_WARNING("file %s only has %ld, but request %ld, loading model fails! \n",
                       filename.c_str(),
                       in_get_size,
                       loaded_data_size);
        return std::vector<T>();
    }
    in.close();
    // If we succeed, return an array with values.
    return host_array;
}

template <typename T, typename T_IN>
int loadWeightFromBinFunc(T *ptr, std::vector<size_t> shape, std::string filename) {
    std::vector<T_IN> host_array = loadWeightFromBinHelper<T_IN>(shape, filename);

    if (host_array.empty()) {
        return 0;
    }

    if (std::is_same<T, T_IN>::value == true) {
        cudaH2Dcpy(ptr, (T *)host_array.data(), host_array.size());
    } else {
        T_IN *ptr_2 = nullptr;
        deviceMalloc(&ptr_2, host_array.size(), false);
        cudaH2Dcpy(ptr_2, host_array.data(), host_array.size());
        invokeCudaD2DcpyConvert(ptr, ptr_2, host_array.size());
        deviceFree(ptr_2);
    }
    return 0;
}

template int loadWeightFromBinFunc<float, float>(float *ptr, std::vector<size_t> shape, std::string filename);
template int loadWeightFromBinFunc<half, float>(half *ptr, std::vector<size_t> shape, std::string filename);
template int loadWeightFromBinFunc<float, half>(float *ptr, std::vector<size_t> shape, std::string filename);
template int loadWeightFromBinFunc<half, half>(half *ptr, std::vector<size_t> shape, std::string filename);
template int loadWeightFromBinFunc<int8_t, int8_t>(int8_t *ptr, std::vector<size_t> shape, std::string filename);

template <typename T>
int loadWeightFromBin(T *ptr, std::vector<size_t> shape, std::string filename, QKCudaDataType model_file_type) {
    switch (model_file_type) {
    case QKCudaDataType::FP32:
        loadWeightFromBinFunc<T, float>(ptr, shape, filename);
        break;

    case QKCudaDataType::FP16:
        loadWeightFromBinFunc<T, half>(ptr, shape, filename);
        break;
    case QKCudaDataType::INT8:
        loadWeightFromBinFunc<T, int8_t>(ptr, shape, filename);
        break;
    default:
        QK_LOG_ERROR("Does not support QKCudaDataType=%d", model_file_type);
        QK_CHECK(false);
    }
    return 0;
}

template <>
int loadWeightFromBin(int *ptr, std::vector<size_t> shape, std::string filename, QKCudaDataType model_file_type) {
    loadWeightFromBinFunc<int, int>(ptr, shape, filename);
    return 0;
}

template int loadWeightFromBin(float *ptr, std::vector<size_t> shape, std::string filename, QKCudaDataType model_file_type);
template int loadWeightFromBin(half *ptr, std::vector<size_t> shape, std::string filename, QKCudaDataType model_file_type);
template int loadWeightFromBin(int8_t *ptr, std::vector<size_t> shape, std::string filename, QKCudaDataType model_file_type);

template <typename T_IN, typename T_OUT>
__global__ void cudaD2DcpyConvert(T_OUT *dst, const T_IN *src, const size_t size) {
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x) {
        dst[tid] = cuda_cast<T_OUT>(src[tid]);
    }
}

template <typename T_IN, typename T_OUT>
void invokeCudaD2DcpyConvert(T_OUT *tgt, const T_IN *src, const size_t size, cudaStream_t stream) {
    cudaD2DcpyConvert<<<256, 256, 0, stream>>>(tgt, src, size);
}

template void invokeCudaD2DcpyConvert(int8_t *tgt, const float *src, const size_t size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(float *tgt, const int8_t *src, const size_t size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(float *tgt, const int *src, const size_t size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(half *tgt, const int *src, const size_t size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(float *tgt, const float *src, const size_t size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(half *tgt, const float *src, const size_t size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(float *tgt, const half *src, const size_t size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(uint *tgt, const int *src, const size_t size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(int *tgt, const uint *src, const size_t size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(int *tgt, const float *src, const size_t size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(int *tgt, const half *src, const size_t size, cudaStream_t stream);

} // namespace space_llm
