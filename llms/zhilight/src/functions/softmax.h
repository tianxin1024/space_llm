#pragma once
#include "core/core.h"

namespace bmengine {

namespace functions {

void softmax(
    const core::Context &ctx,
    const core::Tensor &logits,
    const core::Tensor &output,
    float temperature = 1.0f);

}

} // namespace bmengine::functions
