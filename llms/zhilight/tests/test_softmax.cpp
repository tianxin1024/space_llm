#include "core/core.h"
#include "functions/all.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cuda_fp16.h>
#include <algorithm>

using namespace bmengine;

template <typename T>
T t_random();

template <>
double t_random() {
    return (double)rand() / ((double)RAND_MAX / 10);
}

template <>
float t_random() {
    return (float)rand() / ((float)RAND_MAX / 10);
}

template <>
half t_random() {
    return __float2half(t_random<float>());
}

template <>
int32_t t_random() {
    return rand();
}

template <>
int16_t t_random() {
    return rand() % 32768;
}

template <>
int8_t t_random() {
    return rand() % 128;
}

template <typename T>
core::DataType t_type() {
    if (std::is_same<T, int8_t>::value)
        return core::DataType::kInt8;
}

template <typename T>
bool run_one_test(const core::Context &ctx, size_t batch, size_t n, bool print = false) {
    core::DataType dtype = t_type<T>();
    T *inp = new T[batch * n];
    T *out = new T[batch * n];
    T *ans = new T[batch * n];
    for (int i = 0; i < batch * n; i++) {
        inp[i] = t_random<T>();
        ans[i] = inp[i];
    }
    core::Tensor tensor_inp = ctx.tensor({batch, n}, dtype);
    core::Tensor tensor_out = ctx.tensor({batch, n}, dtype);
    tensor_inp.from_buffer(inp);

    functions::softmax(ctx, tensor_inp, tensor_out);

    for (int i = 0; i < batch; ++i) {
        float local_max = -INFINITY;
        for (int j = 0; j < n; ++j) {
            local_max = std::max<float>(local_max, (float)ans[i * n + j]);
        }
        float local_sum = 0;
        for (int j = 0; j < n; ++j) {
            ans[i * n + j] = std::exp((float)ans[i * n + j] - local_max);
            local_sum += (float)ans[i * n + j];
        }
        for (int j = 0; j < n; ++j) {
            ans[i * n + j] = (float)ans[i * n + j] / local_sum;
        }
    }
    tensor_out.to_buffer(out);

    bool ret = true;
    for (int i = 0; i < batch; i++) {
        if (print) {
            std::cout << "input" << std::endl;
            for (int j = 0; j < n; j++) {
                std::cout << std::setw(8) << std::setprecision(4) << (float)inp[i * n + j] << " ";
            }
            std::cout << std::endl;
            std::cout << "output" << std::endl;
            for (int j = 0; j < n; j++) {
                std::cout << std::setw(8) << std::setprecision(4) << (float)out[i * n + j] << " ";
            }
            std::cout << std::endl;
            std::cout << "ans" << std::endl;
            for (int j = 0; j < n; j++) {
                std::cout << std::setw(8) << std::setprecision(4) << (float)ans[i * n + j] << " ";
            }
            std::cout << std::endl;
        }
        bool correct = true;
        for (int j = 0; j < n; ++j) {
            if ((T)std::abs(out[i * n + j] - ans[i * n + j]) > Eps<T>()) {
                correct = false;
                break;
            }
        }
        if (print) {
            if (correct) {
                if (correct) {
                    std::cout << "correct" << std::endl;
                } else {
                    std::cout << "wrong" << std::endl;
                }
                std::cout << std::endl;
            }
        }
        ret &= correct;
    }

    delete[] inp;
    delete[] out;
    delete[] ans;
    return ret;
}

bool all_pass = true;

#define RUN_ONE_TEST_CASE(type, batch, n, idx, total)                                      \
    if (run_one_test<type>(ctx, batch, n)) {                                               \
        std::cout << "[" #type "] Test " << idx << "/" << total << " passed" << std::endl; \
    } else {                                                                               \
        std::cout << "[" #type "] Test " << idx << "/" << total << " failed" << std::endl; \
        all_pass = false;                                                                  \
    }

#define RUN_TEST_CASE(type, total)                       \
    for (int i = 0; i < total; i++) {                    \
        int batch = 128;                                 \
        int n = t_random<int>() % 100000 + 10;           \
        RUN_ONE_TEST_CASE(type, batch, n, i + 1, total); \
    }

int main() {
    srand(0);

    bmengine::core::Engine engine({
        {0, 1ll * 1024 * 1024 * 1024},
    });
    auto ctx = engine.create_context({0});
    bmengine::core::WithDevice device(ctx, 0);

    RUN_TEST_CASE(half, 10);
    return 0;
}
