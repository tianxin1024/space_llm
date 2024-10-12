#pragma once

#include "utils/cuda_utils.h"
#include "utils/tensor.h"
#include "utils/cuda_utils.h"

namespace space_llm {

template <typename T>
void deviceMalloc(T **ptr, size_t size, bool is_random_initialize = true);

template <typename T>
void cudaRandomUniform(T *buffer, const size_t size);

template <typename T>
void deviceFree(T *&ptr);

template <typename T>
void cudaD2Dcpy(T *tgt, const T *src, const size_t size);

} // namespace space_llm
