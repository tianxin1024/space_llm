#pragma once

#include "utils/cuda_utils.h"
#include "utils/tensor.h"
#include "utils/cuda_utils.h"
#include "utils/cuda_type_utils.cuh"

namespace space_llm {

template <typename T>
void deviceMalloc(T **ptr, size_t size, bool is_random_initialize = true);

template <typename T>
void cudaRandomUniform(T *buffer, const size_t size);

template <typename T>
void deviceFree(T *&ptr);

template <typename T>
void cudaD2Hcpy(T *tgt, const T *src, const size_t size);

template <typename T>
void cudaH2Dcpy(T *tgt, const T *src, const size_t size);

template <typename T>
void cudaD2Dcpy(T *tgt, const T *src, const size_t size);

template <typename T>
void cudaAutoCpy(T *tgt, const T *src, const size_t size, cudaStream_t stream);

template <typename T>
int loadWeightFromBin(T *ptr,
                      std::vector<size_t> shape,
                      std::string filename,
                      QKCudaDataType model_file_type = QKCudaDataType::FP32);

template <typename T_IN, typename T_OUT>
void invokeCudaD2DcpyConvert(T_OUT *tgt, const T_IN *src, const size_t size, cudaStream_t stream = 0);

} // namespace space_llm
