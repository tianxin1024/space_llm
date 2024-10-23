#pragma once

#include "utils/denseWeight.h"

namespace space_llm {

template <typename T1, typename T2 = T1>
struct AttentionWeight {
    DenseWeight<T1, T2> query_weight;
    DenseWeight<T1, T2> key_weight;
    DenseWeight<T1, T2> value_weight;
    DenseWeight<T1, T2> attention_output_weight;
    DenseWeight<T1, T2> ia3_key_weight;
    DenseWeight<T1, T2> ia3_value_weight;
};

} // namespace space_llm
