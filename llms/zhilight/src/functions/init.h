#pragma once
#include "core/core.h"

namespace bmengine {

namespace functions {

void zeros_(const core::Context& ctx, const core::Tensor& x);
void ones_(const core::Context& ctx, const core::Tensor& x);
void normal_(const core::Context& ctx, curandGenerator_t& gen, const core::Tensor& x);
}

}
