#include <iostream>
#include <string>
#include <vector>

#include "utils/cuda_utils.h"
#include "utils/memory_utils.h"
#include "kernels/activation_kernels.h"

#include "unittest_utils.h"

using namespace space_llm;

struct TestCase {
    std::string name;
    size_t m;
    size_t n;
    size_t ite;

    std::string toString() {
        char buf[100];
        snprintf(buf, sizeof(buf), "TestCase[name=%s, m=%ld, n=%ld]", name.c_str(), m, n);
        return buf;
    }

    void print() {
        QK_LOG_INFO(toString());
    }
};

template <typename T>
void testActivationKernel(TestCase tc) {
    const int m = tc.m;
    const int n = tc.n;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    T *output_baseline, *output_opt1, *bias;
    deviceMalloc(&output_baseline, m * n);
    deviceMalloc(&output_opt1, m * n);
    deviceMalloc(&bias, n);
    cudaD2Dcpy(output_opt1, output_baseline, m * n);
    invokeGenericActivation<GeluActivation>(output_baseline,
                                            (const T *)bias,
                                            (const T *)nullptr,
                                            (const T *)nullptr,
                                            (const int *)nullptr,
                                            (const T *)nullptr,
                                            m,
                                            n,
                                            0,
                                            (const float *)nullptr,
                                            (const float *)nullptr,
                                            stream);
    invokeAddBiasGeluV2(output_opt1, bias, (const int *)nullptr, (const T *)nullptr, m, n, stream);
    bool passed = checkResult(tc.name, output_baseline, output_opt1, m * n, true, true);
    QK_CHECK(passed);

    deviceFree(output_baseline);
    deviceFree(output_opt1);
    deviceFree(bias);
}

int main() {
    printf("[INFO] Device: %s \n", getDeviceName().c_str());

    std::vector<TestCase> test_cases = {
        // TC : name / m / n
        TestCase{"addBiasGelu", 32, 1024, 1000},
    };

    for (auto &tc : test_cases) {
        testActivationKernel<float>(tc);
    }

    QK_LOG_INFO("testActivationKernel done!!!");
    return 0;
}
