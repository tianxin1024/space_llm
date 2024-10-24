#pragma once

#include "stdlib.h"

namespace space_llm {

// Note that the int8 mode of BERT and GPT are different.
// For int8 mode = 2 on GPT:
// scale (gemm input scale): quantize input of GEMM (float/half) in the int8 range. Namely, int8_x = scale * x
// scale_inter: (gemm output scale) / (gemm input scale * gemm weight scale)
// scale_out: 1 / (gemm output scale), dequantize activation from int8 range to float/half.
template <typename T1, typename T2 = T1>
struct DenseWeight {
    const T1 *kernel = nullptr;
    const T2 *bias = nullptr;
    const T1 *fp8_bias = nullptr;
    const T1 *sp_kernel = nullptr;
    // for int8 kernel
    const int8_t *int8_kernel = nullptr;
    const float *scale = nullptr;
    const T2 *weight_only_quant_scale = nullptr;
    const T2 *moe_scale = nullptr;
    const float *scale_inter = nullptr;
    const float *scale_out = nullptr;

    // FP8 scales
    // scale = AMAX(tensor) / FP8_MAX
    // During GEMM, A (original) = A_scaled (fp8) * "scale of A"
    const float *input_scale = nullptr;      // a scalar
    const float *input_scale_inv = nullptr;  // a scalar
    const float *weight_scale = nullptr;     // a scalar or a vector
    const float *weight_scale_inv = nullptr; // a scalar or a vector
    const float *output_scale = nullptr;     // a scalar
    const float *output_scale_inv = nullptr; // a scalar
    // host pointer of scales, all are scalars
    const float *input_h_scale = nullptr;
    const float *input_h_scale_inv = nullptr;
    const float *weight_h_scale = nullptr;
    const float *weight_h_scale_inv = nullptr;
    const float *output_h_scale = nullptr;
    const float *output_h_scale_inv = nullptr;

    // TODO(bhsueh) check do we need this param
    const float *per_channel_scale_min = nullptr; // = min(weight_scale), used to adjust the scaling of per channel scaling

    bool fuse_gemm_bias = false;
};

} // namespace space_llm
