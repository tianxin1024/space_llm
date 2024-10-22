#pragma once

#include "utils/denseWeight.h"

namespace space_llm {

template <typename T1, typename T2 = T1>
struct ffnWeight {
    DenseWeight<T1, T2> gating_weight;
    DenseWeight<T1, T2> intermediate_weight;
    DenseWeight<T1, T2> intermediate_weight2;
    DenseWeight<T1, T2> output_weight;
    DenseWeight<T1, T2> ia3_weight;
};

} // namespace space_llm
