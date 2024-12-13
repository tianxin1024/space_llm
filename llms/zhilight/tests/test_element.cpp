#include "core/core.h"
#include "functions/element.h"
#include "functions/typecast.h"
#include <iostream>
#include <numeric>
#include <vector>

using namespace bmengine;
using bmengine::core::DataType;
using bmengine::core::Tensor;
using BinaryOp = bmengine::functions::BinaryElementwiseOp;

int main() {
    core::Engine engine({
        {0, 1 * 1024 * 1024 * 1024},
    });
    auto ctx = engine.create_context();
    core::WithDevice device(ctx, 0);

    {
        std::cout << "test pow" << std::endl;
        std::vector<DataType> types{DataType::kFloat, DataType::kHalf};
        for (DataType type : types) {
            std::vector<float> vec(10);
            std::iota(vec.begin(), vec.end(), 0);
            Tensor float_t = ctx.tensor_of(vec);
            Tensor t = functions::typecast(ctx, float_t, type);
            Tensor ret = functions::pow(ctx, t, 2);
            std::cout << ret << std::endl;
        }
    }

    {
        std::cout << "test broadcast" << std::endl;
        std::vector<DataType> types{DataType::kFloat, DataType::kHalf};
        std::vector<float> vec(20);
        std::iota(vec.begin(), vec.end(), 0);
        Tensor x = ctx.tensor_of(vec, {2, 10});
        {
            std::vector<float> vec_y{-1, 2};
            Tensor y = ctx.tensor_of(vec_y, {2, 1});

            BinaryOp op(ctx, BinaryOp::Mul);
            Tensor ret = op.broadcast_y(ctx, x, y);
            std::cout << ret << std::endl;
        }
        {
            x = x.view({10, 2});
            std::vector<float> vec_y{-1, 2};
            Tensor y = ctx.tensor_of(vec_y);

            BinaryOp op(ctx, BinaryOp::Mul);
            Tensor ret = op.broadcast_y(ctx, x, y);
            std::cout << ret << std::endl;
        }
    }
}
