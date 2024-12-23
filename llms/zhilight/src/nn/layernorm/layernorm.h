#pragma once
#include "core/core.h"

namespace nn {
using namespace bmengine;

class LayerNorm : public core::Layer {
    BM_LAYER_DEF(LayerNorm);

    LayerNorm(const core::Context &ctx,
              int dim_model, bool quant = false, float eps = 1e-6, float scale = 1.0,
              core::DataType dtype = core::DataType::kHalf,
              int num_head = 1);

    core::Tensor forward(const core::Context &ctx, const core::Tensor &x);

    // c = a + b (element-wise)
    core::Tensor fuse_add(const core::Context &ctx, const core::Tensor &a, const core::Tensor &b, core::Tensor &c);

    void inplace(const core::Context &ctx, core::Tensor &x);

    static void forward_2(const core::Context &ctx,
                          core::Tensor &x, core::Tensor &y,
                          core::Tensor &x_out, core::Tensor &y_out,
                          LayerNorm *la, LayerNorm *lb);

    void set_rms(bool b); // If false, use standard LayerNorm

    void load_state_dict(
        const core::Context &ctx,
        const std::map<std::string, const core::Tensor> &state_dict,
        const std::string &prefix,
        bool allow_missing) override;
};

} // namespace nn
