#include "core/core.h"
#include "functions/gemm.h"
#include <thread>
#include <iostream>
#include <cuda_fp16.h>

int main() {
    bmengine::core::Engine engine({
        {0, 1ll * 1024 * 1024 * 1024},
    });
    auto ctx = engine.create_context({0});
    bmengine::core::WithDevice device(ctx, 0);

    bmengine::functions::Gemm gemm(ctx, bmengine::core::DataType::kHalf, false, false);
    auto A = ctx.tensor({2, 4}, bmengine::core::DataType::kHalf);
    auto B = ctx.tensor({4, 4}, bmengine::core::DataType::kHalf);

    half buf_A[2][4] = {
        {1.6, 0.0, 1.1, -1.0},
        {0.0, 1.0, 0.0, 0.0},
    };
    half buf_B[4][4] = {
        {1.0, 0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 1.5, 0.0, -1.3},
        {0.0, 1.0, -1.2, 0.0},
    };
    A.from_buffer(buf_A);
    B.from_buffer(buf_B);
    std::cout << "A: " << A << std::endl;
    std::cout << "B: " << B << std::endl;

    {
        auto C = gemm(ctx, A, B);
        std::cout << "Half output:" << std::endl;
        std::cout << "C: " << C << std::endl;
    }
    {
        gemm.set_output_type(bmengine::core::DataType::kFloat);
        auto C = gemm(ctx, A, B);
        std::cout << "Float output:" << std::endl;
        std::cout << "C: " << C << std::endl;
    }
    return 0;
}
