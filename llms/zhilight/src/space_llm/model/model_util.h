#pragma once
#include "core/core.h"

namespace model {

namespace core = bmengine::core;

core::Tensor convert_fp32(const core::Context &ctx, const core::Tensor &logits);

core::Tensor convert_fp16(const core::Context &ctx, const core::Tensor &logits);

core::Tensor concat_logits(
    const core::Context &ctx, const std::tuple<core::Tensor, core::Tensor> &logits_tuple);

static void multiply(
    const core::Context &ctx, const core::Tensor &a, const core::Tensor &c, float b);

} // namespace model
