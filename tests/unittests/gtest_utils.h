#pragma once

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <math.h>

#include "utils/allocator.h"
#include "utils/cuda_utils.h"
#include "utils/memory_utils.h"
#include "utils/tensor.h"

namespace qk = space_llm;

namespace {

#define EPSILON (1e-20)

bool almostEqual(float a, float b, float atol = 1e-5, float rtol = 1e-8) {
    // Params: a = value to compare and b = reference
    // This function follows implementation of numpy.isclose(), which checks
    //   abs(a - b) <= (atol + rtol * abs(b)).
    // Note that the inequality above is asymmetric where b is considered as
    // a reference value. To account into both absolute/relative errors, it
    // uses absolute tolerance and relative tolerance at the same time. The
    // default values of atol and rtol borrowed from numpy.isclose(). For the
    // case of nan value, the result will be true.
    if (isnan(a) && isnan(b)) {
        return true;
    }
    return fabs(a - b) <= (atol + rtol * fabs(b));
}

template <typename T>
bool checkResult(std::string name, T *out, T *ref, size_t size, float atol, float rtol) {
    size_t failures = 0;
    float relative_gap = 0.0f;
    ;

    for (size_t i = 0; i < size; ++i) {
        // The values for the output and the reference.
        float a = (float)out[i];
        float b = (float)ref[i];

        bool ok = almostEqual(a, b, atol, rtol);
        // Print the error.
        if (!ok && failures < 4) {
            QK_LOG_ERROR(">> invalid result for i=%lu:", i);
            QK_LOG_ERROR(">>    found......: %10.6f", a);
            QK_LOG_ERROR(">>    expected...: %10.6f", b);
            QK_LOG_ERROR(">>    error......: %.6f", fabsf(a - b));
            QK_LOG_ERROR(">>    tol........: %.6f", atol + rtol * fabs(b));
        }
        // Update the number of failures.
        failures += ok ? 0 : 1;
        // Update the relative gap.
        relative_gap += fabsf(a - b) / (fabsf(b) + EPSILON);
    }

    relative_gap /= size;

    // Allow not matched up to 1% elements.
    size_t tol_failures = (size_t)(0.01 * size);
    if (failures > tol_failures) {
        QK_LOG_ERROR("%s (failures: %.2f%% atol: %.2e rtol: %.2e rel_gap: %.2e%%)",
                     name.c_str(), 100. * failures / size, atol, rtol, 100. * relative_gap);
    }
    return failures <= tol_failures;
}

template <typename T>
bool checkResult(std::string name, T *out, T *ref, size_t size,
                 bool device_out = true, bool device_ref = false) {
    bool is_fp32 = sizeof(T) == 4;
    float atol = is_fp32 ? 1e-4f : 1e-3f;
    float rtol = is_fp32 ? 1e-2f : 1e-1f;

    T *h_out = nullptr;
    if (device_out) {
        h_out = new T[size];
        cudaMemcpy(h_out, out, sizeof(T) * size, cudaMemcpyDeviceToHost);
        out = h_out;
    }
    T *h_ref = nullptr;
    if (device_ref) {
        h_ref = new T[size];
        cudaMemcpy(h_ref, ref, sizeof(T) * size, cudaMemcpyDeviceToHost);
        ref = h_ref;
    }
    bool is_ok = checkResult(name, out, ref, size, atol, rtol);
    if (h_out != nullptr) {
        delete[] h_out;
    }
    if (h_ref != nullptr) {
        delete[] h_ref;
    }
    return is_ok;
}

template <typename T>
void initRandomInt(T *ptr, size_t size, float minval, float maxval) {
    for (size_t i = 0; i < size; ++i) {
        float val = static_cast<float>(rand() / static_cast<float>(RAND_MAX));
        val *= (maxval - minval);
        ptr[i] = static_cast<T>(minval + val);
    }
}

void initRandomInt(int *ptr, size_t size, int minval, int maxval) {
    assert(minval <= maxval);
    int mod = maxval - minval;
    for (size_t i = 0; i < size; ++i) {
        ptr[i] = minval + rand() % mod;
    }
}

namespace math {
template <typename T>
inline T add(T a, T b) {
    return static_cast<T>((float)a + (float)b);
}

template <typename T>
inline T mul(T a, T b) {
    return static_cast<T>((float)a * (float)b);
}

template <typename T>
inline T fma(T a, T b, T c) {
    return static_cast<T>((float)a * (float)b + (float)c);
}

} // namespace math

class QKTestBase : public testing::Test {
public:
    void SetUp() override {
        int device = 0;
        cudaGetDevice(&device);
        cudaStreamCreate(&stream);
        allocator = new qk::Allocator(device);
        allocator->setStream(stream);
    }

    void TearDown() override {
        // Automatically allocated CPU buffers should be released at the end of a test.
        // We don't need to care GPU buffers allocated by Allocator because they are
        // managed by the allocator.
        for (auto &buffer : allocated_cpu_buffers) {
            free(buffer);
        }
        allocated_cpu_buffers.clear();
        delete allocator;
        cudaStreamDestroy(stream);
    }

protected:
    cudaStream_t stream;
    qk::Allocator *allocator;
    std::vector<void *> allocated_cpu_buffers;

    // Utilities to easily handle tensor instances in test cases.
    qk::Tensor createTensor(const qk::MemoryType mtype,
                            const qk::DataType dtype,
                            const std::vector<size_t> shape) {
        size_t n_elmts = std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
        size_t buf_size = qk::Tensor::getTypeSize(dtype) * n_elmts;

        void *data = nullptr;
        if (mtype == qk::MEMORY_CPU || mtype == qk::MEMORY_CPU_PINNED) {
            data = malloc(buf_size);
            allocated_cpu_buffers.push_back(data);
        } else {
            data = allocator->malloc(buf_size);
        }
        return qk::Tensor(mtype, dtype, shape, data);
    }

    template <typename T>
    qk::Tensor toHost(qk::Tensor &device_tensor) {
        if (device_tensor.data == nullptr) {
            return qk::Tensor();
        }
        qk::Tensor host_tensor = createTensor(qk::MEMORY_CPU, device_tensor.type, device_tensor.shape);
        qk::cudaAutoCpy(host_tensor.getPtr<T>(), device_tensor.getPtr<T>(), host_tensor.size(), stream);
        cudaStreamSynchronize(stream);
        return host_tensor;
    }

    void copyTensor(qk::Tensor &dst, qk::Tensor &src) {
        QK_CHECK_WITH_INFO(
            src.sizeBytes() == dst.sizeBytes(),
            qk::fmtstr("src and dst has different size (%ld != %ld)", src.sizeBytes(), dst.sizeBytes()));
        qk::cudaAutoCpy(dst.getPtr<char>(), src.getPtr<char>(), src.sizeBytes(), stream);
        cudaStreamSynchronize(stream);
    }
};

typedef testing::Types<float, half> FloatAndHalfTypes;
#ifndef ENABLE_BF16
typedef FloatAndHalfTypes SupportTypes;
#else
typedef testing::Types<float, half, __nv_bfloat16> FloatHalfBf16Types;
typedef FloatHalfBf16Types SupportTypes;
#endif

} // namespace
