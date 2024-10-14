#pragma once

#include <algorithm> // min, max
#include <assert.h>  // assert
#include <float.h>   // FLT_MAX
#include <iostream>  // snprintf
#include <math.h>    // expf, log
#include <limits>    // numeric_limits
#include <stdlib.h>  // rand
#include <string>    // string
#include <vector>    // vector

#include "utils/cuda_utils.h"

#define PRINT_LIMIT 16
#define EPSILON (1e-20)
#define EPSILON_FP16 (1e-10)

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
}
