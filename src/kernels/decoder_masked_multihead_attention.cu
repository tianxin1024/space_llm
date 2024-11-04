#include "kernels/decoder_masked_multihead_attention.h"
#include <assert.h>
#include <float.h>

template <typename T, typename KERNEL_PARAMS_TYPE>
void multihead_attention_(const KERNEL_PARAMS_TYPE &params, const cudaStream_t &stream) {
    // switch (params.hidden_size_per_head) {
    //     case 32:
    //         mmha_launch_kernel<T, 32, 32, KERNEL_PARAMS_TYPE>(params, stream);
    //         break;
    //     case 48:
    //         mmha_launch_kernel<T, 48, 64, KERNEL_PARAMS_TYPE>(params, stream);
    //         break;
    //     case 64:
    //         mmha_launch_kernel<T, 64, 64, KERNEL_PARAMS_TYPE>(params, stream);
    //         break;
    //     case 80:
    //         mmha_launch_kernel<T, 80, 128, KERNEL_PARAMS_TYPE>(params, stream);
    //         break;
    //     case 96:
    //         mmha_launch_kernel<T, 96, 128, KERNEL_PARAMS_TYPE>(params, stream);
    //         break;
    //     case 112:
    //         mmha_launch_kernel<T, 112, 128, KERNEL_PARAMS_TYPE>(params, stream);
    //         break;
    //     case 128:
    //         mmha_launch_kernel<T, 128, 128, KERNEL_PARAMS_TYPE>(params, stream);
    //         break;
    //     case 144:
    //         mmha_launch_kernel<T, 144, 256, KERNEL_PARAMS_TYPE>(params, stream);
    //         break;
    //     case 160:
    //         mmha_launch_kernel<T, 160, 256, KERNEL_PARAMS_TYPE>(params, stream);
    //         break;
    //     case 192:
    //         mmha_launch_kernel<T, 192, 256, KERNEL_PARAMS_TYPE>(params, stream);
    //         break;
    //     case 224:
    //         mmha_launch_kernel<T, 224, 256, KERNEL_PARAMS_TYPE>(params, stream);
    //         break;
    //     case 256:
    //         mmha_launch_kernel<T, 256, 256, KERNEL_PARAMS_TYPE>(params, stream);
    //         break;
    //     default:
    //         assert(false);
    // }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void masked_multihead_attention(const Masked_multihead_attention_params<float> &params, const cudaStream_t &stream) {
    multihead_attention_<float, Masked_multihead_attention_params<float>>(params, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void masked_multihead_attention(const Masked_multihead_attention_params<uint16_t> &params, const cudaStream_t &stream) {
    multihead_attention_<uint16_t, Masked_multihead_attention_params<uint16_t>>(params, stream);
}
