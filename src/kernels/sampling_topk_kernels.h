#pragma once

#include "utils/logger.h"
#include <curand_kernel.h>

namespace space_llm {

void invokeCurandInitialize(curandState_t *state,
                            const size_t batch_size,
                            unsigned long long random_seed,
                            cudaStream_t stream);

} // namespace space_llm
