#pragma once

#include "utils/cuda_utils.h"

namespace space_llm {

enum class ActivationType {
    Gelu,
    Relu,
    Silu,
    GeGLU,
    ReGLU,
    SiGLU,
    Identity,
    InvalidType
};

} //  namespace space_llm
