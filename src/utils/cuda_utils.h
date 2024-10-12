#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "utils/string_utils.h"
#include "utils/logger.h"

namespace space_llm {

#define MAX_CONFIG_NUM 20
#define COL32_ 32
// workspace for cublas gemm : 32MB
#define CUBLAS_WORKSPACE_SIZE 33554432

typedef struct __align__(4) {
    half x, y, z, w;
}
half4;

/* **************************** type definition ***************************** */

enum CublasDataType {
    FLOAT_DATATYPE = 0,
    HALF_DATATYPE = 1,
    BFLOAT16_DATATYPE = 2,
    INT8_DATATYPE = 3,
    FP8_DATATYPE = 4
};

enum FtCudaDataType {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    INT8 = 3,
    FP8 = 4
};

enum class OperationType {
    FP32,
    FP16,
    BF16,
    INT8,
    FP8
};

/* **************************** debug tools ********************************* */
static const char *_cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorString(error);
}

static const char *_cudaGetErrorEnum(cublasStatus_t error) {
    switch (error) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        throw std::runtime_error(std::string("[QK][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " "
                                 + file + ":" + std::to_string(line) + " \n");
    }
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)
#define check_cuda_error_2(val, file, line) check((val), #val, file, line)

inline void syncAndCheck(const char *const file, int const line) {
    // When QK_DEBUG_LEVEL=DEBUG, must check error
    static char *level_name = std::getenv("QK_DEBUG_LEVEL");
    if (level_name != nullptr) {
        static std::string level = std::string(level_name);
        if (level == "DEBUG") {
            cudaDeviceSynchronize();
            cudaError_t result = cudaGetLastError();
            if (result) {
                throw std::runtime_error(std::string("[QK][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result))
                                         + " " + file + ":" + std::to_string(line) + " \n");
            }
            QK_LOG_DEBUG(fmtstr("run syncAndCheck at %s:%d", file, line));
        }
    }

#ifndef NDEBUG
    cudaDeviceSynchronize();
    cudaError_t result = cudaGetLastError();
    if (result) {
        throw std::runtime_error(std::string("[QK][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " "
                                 + file + ":" + std::to_string(line) + " \n");
    }
#endif
}

#define sync_check_cuda_error() syncAndCheck(__FILE__, __LINE__)

#define checkCUDNN(expression)                                                                                       \
    {                                                                                                                \
        cudnnStatus_t status = (expression);                                                                         \
        if (status != CUDNN_STATUS_SUCCESS) {                                                                        \
            std::cerr << "Error on file " << __FILE__ << " line " << __LINE__ << ": " << cudnnGetErrorString(status) \
                      << std::endl;                                                                                  \
            std::exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                            \
    }

[[noreturn]] inline void throwRuntimeError(const char *const file, int const line, std::string const &info = "") {
    throw std::runtime_error(std::string("[QK][ERROR] ") + info + " Assertion fail: " + file + ":"
                             + std::to_string(line) + " \n");
}

inline void myAssert(bool result, const char *const file, int const line, std::string const &info = "") {
    if (!result) {
        throwRuntimeError(file, line, info);
    }
}

#define QK_CHECK(val) myAssert(val, __FILE__, __LINE__)
#define QK_CHECK_WITH_INFO(val, info)                                      \
    do {                                                                   \
        bool is_valid_val = (val);                                         \
        if (!is_valid_val) {                                               \
            space_llm::myAssert(is_valid_val, __FILE__, __LINE__, (info)); \
        }                                                                  \
    } while (0)

#define QK_THROW(info) throwRuntimeError(__FILE__, __LINE__, info)

/*************Time Handling**************/
class CudaTimer {
private:
    cudaEvent_t event_start_;
    cudaEvent_t event_stop_;
    cudaStream_t stream_;

public:
    explicit CudaTimer(cudaStream_t stream = 0) {
        stream_ = stream;
    }
    void start() {
        check_cuda_error(cudaEventCreate(&event_start_));
        check_cuda_error(cudaEventCreate(&event_stop_));
        check_cuda_error(cudaEventRecord(event_start_, stream_));
    }
    float stop() {
        float time;
        check_cuda_error(cudaEventRecord(event_stop_, stream_));
        check_cuda_error(cudaEventSynchronize(event_stop_));
        check_cuda_error(cudaEventElapsedTime(&time, event_start_, event_stop_));
        check_cuda_error(cudaEventDestroy(event_start_));
        check_cuda_error(cudaEventDestroy(event_stop_));
        return time;
    }
    ~CudaTimer() {
    }
};

static double diffTime(timeval start, timeval end) {
    return (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
}

/* ***************************** common utils ****************************** */

inline int getSMVersion() {
    int device{-1};
    check_cuda_error(cudaGetDevice(&device));
    int sm_major = 0;
    int sm_minor = 0;
    check_cuda_error(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
    check_cuda_error(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
    return sm_major * 10 + sm_minor;
}

cudaError_t getSetDevice(int i_device, int *o_device = NULL);

// clang-format off
template<typename T> struct packed_type;
template <>          struct packed_type<float>         { using type = float; }; // we don't need to pack float by default
template <>          struct packed_type<half>          { using type = half2; };

#ifdef ENABLE_BF16
template<>
struct packed_type<__nv_bfloat16> {
    using type = __nv_bfloat162;
};
#endif

template<typename T> struct num_elems;
template <>          struct num_elems<float>           { static constexpr int value = 1; };
template <>          struct num_elems<float2>          { static constexpr int value = 2; };
template <>          struct num_elems<float4>          { static constexpr int value = 4; };
template <>          struct num_elems<half>            { static constexpr int value = 1; };
template <>          struct num_elems<half2>           { static constexpr int value = 2; };
#ifdef ENABLE_BF16
template <>          struct num_elems<__nv_bfloat16>   { static constexpr int value = 1; };
template <>          struct num_elems<__nv_bfloat162>  { static constexpr int value = 2; };
#endif

template<typename T, int num> struct packed_as;
template<typename T>          struct packed_as<T, 1>              { using type = T; };
template<>                    struct packed_as<half,  2>          { using type = half2; };
template<>                    struct packed_as<float,  2>         { using type = float2; };
template<>                    struct packed_as<int8_t, 2>         { using type = int16_t; };
template<>                    struct packed_as<int32_t, 2>        { using type = int2; };
template<>                    struct packed_as<half2, 1>          { using type = half; };
#ifdef ENABLE_BF16
template<> struct packed_as<__nv_bfloat16,  2> { using type = __nv_bfloat162; };
template<> struct packed_as<__nv_bfloat162, 1> { using type = __nv_bfloat16;  };
#endif

inline __device__ float2 operator*(float2 a, float2 b) { return make_float2(a.x * b.x, a.y * b.y); }
inline __device__ float2 operator*(float2 a, float  b) { return make_float2(a.x * b, a.y * b); }
// clang-format on

} // namespace space_llm
