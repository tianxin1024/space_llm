#pragma once
#include <stddef.h>
#include <stdint.h>
#include "core/core.h"
#include <map>
#include <string>
#include "model/model.h"

using namespace bmengine;

namespace utils {

void load_state_dict(
    bmengine::core::Context &ctx,
    const std::map<std::string, bmengine::core::Tensor> &state_dict,
    std::map<const std::string, bmengine::core::Tensor *> named_params,
    bool parallel = false);

} // namespace utils
