#include "kernels/sampling_topk_kernels.h"

namespace space_llm {

__global__ void curandInitialize(curandState_t *state, const int size, const unsigned long long random_seed) {
    if (blockIdx.x * blockDim.x + threadIdx.x < size) {
        curand_init(random_seed, 0, 0, &state[blockIdx.x * blockDim.x + threadIdx.x]);
    }
}

void invokeCurandInitialize(curandState_t *state,
                            const size_t batch_size,
                            unsigned long long random_seed,
                            cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((int)(ceil(batch_size * 1.0 / 256)));
    curandInitialize<<<grid, block, 0, stream>>>(state, batch_size, random_seed);
}

} // namespace space_llm
