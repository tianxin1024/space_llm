#pragma once

#include <string>
#include <unordered_map>

#include "utils/string_utils.h"

namespace space_llm {

enum class RepetitionPenaltyType {
    Additive,       // the presence penalty
    Multiplicative, // the repetition penalty
    None            // No repetition penalty.
};

inline float getDefaultPenaltyValue(RepetitionPenaltyType penalty_type) {
    switch (penalty_type) {
    case RepetitionPenaltyType::Additive:
        return 0.0f;
    case RepetitionPenaltyType::Multiplicative:
        return 1.0f;
    default:
        break;
    }
    return 0.0f;
}

} // namespace space_llm
