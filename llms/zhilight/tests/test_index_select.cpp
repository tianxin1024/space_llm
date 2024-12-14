#include "core/core.h"
#include "functions/index_select.h"
#include <iostream>

int main() {
    bmengine::core::Engine engine({
        {0, 1 * 1024 * 1024 * 1024},
    });
    auto ctx = engine.create_context();
    bmengine::core::WithDevice device(ctx, 0);

    {
        auto inp = ctx.tensor({4, 10, 10}, bmengine::core::DataType::kInt32);
        int buffer[4][10][10];
        for (int k = 0; k < 4; k++) {
            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 10; j++) {
                    buffer[k][i][j] = k * 10000 + i * 100 + j;
                }
            }
        }
        inp.from_buffer(buffer);
        std::cout << "Origin tensor: " << std::endl;
        std::cout << inp << std::endl;
        std::cout << "Select dim 0: " << std::endl;
        auto t = bmengine::functions::index_select(ctx, inp, 0, ctx.tensor_of(std::vector<int>({1, 3})));
        std::cout << t << std::endl;

        std::cout << "Select dim 1: " << std::endl;
        auto t1 = bmengine::functions::index_select(ctx, inp, -2, ctx.tensor_of(std::vector<int>({1, 3, 5, 9})));
        std::cout << t1 << std::endl;

        std::cout << "Select dim -1: " << std::endl;
        auto t2 = bmengine::functions::index_select(ctx, inp, -1, ctx.tensor_of(std::vector<int>({1, 3, 5, 9})));
        std::cout << t2 << std::endl;

        std::cout << "Select dim 2: " << std::endl;
        auto t3 = bmengine::functions::index_select(ctx, inp, 2, ctx.tensor_of(std::vector<int>({2, 4, 6, 8})));
        std::cout << t3 << std::endl;
    }
}
