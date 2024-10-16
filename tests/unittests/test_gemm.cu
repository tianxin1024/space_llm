#include <iostream>
#include <assert.h>
#include <math.h>
#include <vector>
#include <tuple>
#include <string>

#include "utils/logger.h"
#include "utils/memory_utils.h"
#include "utils/allocator.h"
#include "utils/cuda_utils.h"
#include "utils/cublasMMWrapper.h"
#include "utils/gemm.h"

using namespace space_llm;

class TestFailureError : public std::exception {
private:
    std::string msg_;

public:
    explicit TestFailureError() = default;
    explicit TestFailureError(std::string name, std::string msg = "") {
        msg_ = fmtstr("TEST FAIL [%s] %s", name.c_str(), msg.c_str());
    }
    const char *what() const throw() {
        return msg_.c_str();
    }
};

#define EXPECT_ALMOST_EQUAL(name, dtype, ctype, out, ref)       \
    do {                                                        \
        bool is_ok = checkResult<dtype, ctype>(name, out, ref); \
        if (!is_ok) {                                           \
            QK_LOG_ERROR("TEST FAIL [%s] at %s:%d",             \
                         __func__, __FILE__, __LINE__);         \
            throw TestFailureError(__func__);                   \
        }                                                       \
    } while (false)

// TensorWrapper is to handle a tensor object as well as its memory buffer,
// because tensor.data is const we cannot set values.
class TensorWrapper {
private:
    IAllocator *allocator;

public:
    std::vector<size_t> shape;
    DataType type;
    Tensor *tensor;
    void *data;

    TensorWrapper(IAllocator *allocator, DataType dtype, std::vector<size_t> shape, bool zero_init = false) {
        this->allocator = allocator;
        this->type = dtype;
        this->shape = shape;

        size_t tensor_memsize = this->memsize();
        this->data = this->allocator->malloc(tensor_memsize, false);

        if (zero_init) {
            check_cuda_error(cudaMemset(data, 0x0, tensor_memsize));
        } else {
            setRandomValues();
        }
        this->tensor = new Tensor(MEMORY_GPU, dtype, shape, data);
    }

    TensorWrapper(TensorWrapper const &other) :
        allocator(other.allocator), shape(other.shape), type(other.type), data(other.data), tensor(other.tensor) {
        QK_LOG_DEBUG("TensorWrapper copy: this=%p other=%p", data, other.data);
    }

    ~TensorWrapper() {
        delete tensor;
        allocator->free((void **)&data);
    }

    void setInvalidValues() {
        size_t type_size = (tensor->type == TYPE_FP16) ? sizeof(float) : sizeof(half);
        size_t tensor_size = type_size * tensor->size();
        // Fill by a random number to guarantee invalid values
        check_cuda_error(cudaMemset(data, 0xdc, tensor_size));
    }

    void setRandomValues() {
        // random initialization
        size_t num_elements = this->size();
        switch (this->type) {
        case TYPE_FP32:
            cudaRandomUniform((float *)data, num_elements);
            break;
        case TYPE_FP16:
            cudaRandomUniform((half *)data, num_elements);
            break;
        default:
            throw std::runtime_error("Not supported data type");
        }
    }

    size_t size() {
        size_t n_elements = 1;
        for (size_t s : this->shape) {
            n_elements *= s;
        }
        return n_elements;
    }

    size_t memsize() {
        size_t type_size = 0;
        switch (this->type) {
        case TYPE_FP32:
            type_size = sizeof(float);
            break;
        case TYPE_FP16:
            type_size = sizeof(half);
            break;
        default:
            throw std::runtime_error("Not supported data type");
        }
        return type_size * this->size();
    }
};

template <DataType computeType>
void computeReference(GemmOp transa, GemmOp transb,
                      TensorWrapper &C, TensorWrapper &A, TensorWrapper &B,
                      float alpha = 1.0f, float beta = 0.0f) {
    size_t m = C.shape[0];
    size_t n = C.shape[1];
    size_t k = A.shape[1];

    size_t lda = (transa == GEMM_OP_N) ? k : m;
    size_t ldb = (transb == GEMM_OP_N) ? n : k;
    size_t ldc = n;

    cudaDataType_t atype = (A.type == TYPE_FP16) ? CUDA_R_16F : CUDA_R_32F;
    cudaDataType_t btype = (B.type == TYPE_FP16) ? CUDA_R_16F : CUDA_R_32F;
    cudaDataType_t ctype = (C.type == TYPE_FP16) ? CUDA_R_16F : CUDA_R_32F;
    cudaDataType_t compute_type = (computeType == TYPE_FP16) ? CUDA_R_16F : CUDA_R_32F;

    cublasHandle_t cublas_handle;
    check_cuda_error(cublasCreate(&cublas_handle));

    half h_alpha = (half)alpha;
    half h_beta = (half)beta;
    const void *_alpha = (computeType == TYPE_FP16) ? (const void *)&h_alpha : (const void *)&alpha;
    const void *_beta = (computeType == TYPE_FP16) ? (const void *)&h_beta : (const void *)&beta;

    check_cuda_error(cublasGemmEx(cublas_handle,
                                  getCublasOperation(transa),
                                  getCublasOperation(transb),
                                  n, m, k,
                                  _alpha,
                                  (const void *)B.data, btype, ldb,
                                  (const void *)A.data, atype, lda,
                                  _beta,
                                  (void *)C.data, ctype, ldc,
                                  compute_type,
                                  CUBLAS_GEMM_DEFAULT));
    check_cuda_error(cublasDestroy(cublas_handle));
    cudaDeviceSynchronize();
}

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
bool _checkResult(std::string name, TensorWrapper &out, TensorWrapper &ref, float atol, float rtol) {
    assert(out.type == ref.type);

    size_t out_size = out.size();
    size_t ref_size = ref.size();
    T *h_out = reinterpret_cast<T *>(malloc(sizeof(T) * out_size));
    T *h_ref = reinterpret_cast<T *>(malloc(sizeof(T) * ref_size));

    cudaMemcpy(h_out, out.data, sizeof(T) * out_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref, ref.data, sizeof(T) * ref_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    size_t failures = 0;
    for (size_t i = 0; i < out_size; ++i) {
        // The values for the output and the reference.
        float a = (float)h_out[i];
        float b = (float)h_ref[i];

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
    }

    // Allow not matched up to 1% elements.
    size_t tol_failures = (size_t)(0.01 * out_size);
    QK_LOG_INFO("check....... %30s : %s (failures: %.2f%% atol: %.2e rtol: %.2e)",
                name.c_str(), failures <= tol_failures ? "OK" : "FAILED",
                100. * failures / out_size, atol, rtol);
    return failures <= tol_failures;
}

template <typename T, DataType computeType>
bool checkResult(std::string name, TensorWrapper &out, TensorWrapper &ref) {
    float atol = (computeType == TYPE_FP32) ? 1e-6f : 1e-3f;
    float rtol = (computeType == TYPE_FP32) ? 1e-4f : 1e-1f;
    bool is_ok = false;
    if (sizeof(T) == 4) {
        is_ok = _checkResult<float>(name, out, ref, atol, rtol);
    } else {
        is_ok = _checkResult<half>(name, out, ref, atol, rtol);
    }
    return is_ok;
}

template <typename T, DataType computeType>
bool checkResult(TensorWrapper &out, TensorWrapper &ref) {
    return checkResult<T, computeType>("", out, ref);
}

template <typename T>
std::string toString() {
    std::string str = "dtype=";
    str += std::is_same<T, float>::value ? "FP32" : "FP16";
    return str;
}

template <typename T, DataType ctype>
std::string toString() {
    std::string str = "dtype=";
    str += std::is_same<T, float>::value ? "FP32" : "FP16";
    str += ", compute_type=";
    str += (ctype == TYPE_FP32) ? "FP32" : "FP16";
    return str;
}

struct GemmOpPair {
    GemmOp transa;
    GemmOp transb;
};

static const std::vector<GemmOpPair> op_pairs{{GEMM_OP_N, GEMM_OP_N},
                                              {GEMM_OP_N, GEMM_OP_T},
                                              {GEMM_OP_T, GEMM_OP_N},
                                              {GEMM_OP_T, GEMM_OP_T}};

static inline std::string getTestName(const char *func_name, GemmOp transa, GemmOp transb,
                                      size_t m, size_t n, size_t k) {
    return fmtstr("%s [opA=%s, opB=%s, m=%ld, n=%ld, k=%ld]",
                  func_name, getGemmOpString(transa).c_str(), getGemmOpString(transb).c_str(),
                  m, n, k);
}

static inline std::string getTestName(const char *func_name, GemmOpPair op_pairs, size_t m, size_t n, size_t k) {
    return getTestName(func_name, op_pairs.transa, op_pairs.transb, m, n, k);
}

/////////////////////////////////// Unittests //////////////////////////////////////////

template <typename T, DataType computeType>
void testGemmCorrectnessMatmul(size_t m, size_t n, size_t k) {
    QK_LOG_INFO("Matmul function correctness test [m=%ld, n=%ld, k=%ld, %s]",
                m, n, k, toString<T, computeType>().c_str());
    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream));

    Allocator allocator(getDevice());

    DataType dtype = getTensorType<T>();
    TensorWrapper a_tensor(&allocator, dtype, {m, k}, false);
    TensorWrapper b_tensor(&allocator, dtype, {k, n}, false);
    TensorWrapper c_tensor(&allocator, dtype, {m, n}, true);
    TensorWrapper expected(&allocator, dtype, {m, n}, true);

    std::shared_ptr<Gemm> gemm = createGemm(&allocator, stream, false, false);
    gemm->setTypes(a_tensor.type, b_tensor.type, c_tensor.type, computeType);

    for (auto &op_pair : op_pairs) {
        std::string tc_name = getTestName(__func__, op_pair, m, n, k);
        QK_LOG_DEBUG(tc_name);
        computeReference<computeType>(op_pair.transa, op_pair.transb,
                                      expected, a_tensor, b_tensor);

        size_t lda = (op_pair.transa == GEMM_OP_N) ? k : m;
        size_t ldb = (op_pair.transb == GEMM_OP_N) ? n : k;
        size_t ldc = n;

        c_tensor.setInvalidValues(); // to guarantee C has invalid data
        gemm->gemm(op_pair.transa, op_pair.transb, m, n, k,
                   a_tensor.data, a_tensor.type, lda,
                   b_tensor.data, b_tensor.type, ldb,
                   c_tensor.data, c_tensor.type, ldc);
        EXPECT_ALMOST_EQUAL(tc_name + " api1", T, computeType, c_tensor, expected);
    }
}

int main(int argc, char *argv[]) {
    // testGemmCreate();
    using testcase_t = std::tuple<size_t, size_t, size_t>;

    std::vector<testcase_t> testcases = {{16, 32, 64},
                                         {255, 255, 255},
                                         {1041, 2047, 9999},
                                         {1041, 1, 9999},
                                         {1041, 999, 1}};

    // Computation correctness tests
    for (testcase_t &tc : testcases) {
        size_t m = std::get<0>(tc);
        size_t n = std::get<1>(tc);
        size_t k = std::get<2>(tc);

        testGemmCorrectnessMatmul<float, TYPE_FP32>(m, n, k);
    }

    return 0;
}
