#pragma once

#include "utils/tensor.h"

namespace space_llm {

void invokeLengthCriterion(bool *finished,
                           bool *should_stop,
                           int *finished_sum,
                           const uint32_t *sequence_limit_length,
                           int batch_size,
                           int beam_width,
                           int step,
                           cudaStream_t stream);

} // namespace space_llm
