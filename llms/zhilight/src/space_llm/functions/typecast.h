#pragma once
#include "core/core.h"

namespace bmengine {
namespace functions {

core::Tensor typecast(const core::Context &ctx, const core::Tensor &in, core::DataType out_type);

}
} // namespace bmengine::functions
