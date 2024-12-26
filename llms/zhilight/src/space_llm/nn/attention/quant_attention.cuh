#pragma once
#include <assert.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <vector_types.h>
#include "functions/reduce.cuh"

static __inline__ __device__ void DEV_dequant_int8(
    uint32_t const i8s, half2 (&result)[2]
) {
    uint32_t* h = reinterpret_cast<uint32_t*>(&result);

    static constexpr uint32_t mask_for_elt_01     = 0x5150;
    static constexpr uint32_t mask_for_elt_23     = 0x5352;
    static constexpr uint32_t start_byte_for_fp16 = 0x64646464;
    asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[0]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
    asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[1]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));

    // Lastly, we subtract 1152 from our constructed number using fp16 math to get our signed integer as fp16.
    static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[1]) : "r"(h[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
}

template<typename T>
struct Half2Type {
    typedef int T_HALF2;
};

template<>
struct Half2Type<half> {
    typedef half2 T_HALF2;
};

#define WARP_SIZE 32
template<typename T, typename T2 = T>
static __forceinline__ __device__ void DEV_mul_qk_quant_h128(
    const half* __restrict__ g_q, // (dim_head)
    const uint8_t* __restrict__ g_k_quant, // (len_buf, num_kv_heads, dim_head)
    // const float* __restrict__ g_k_scale, // (len_buf, num_kv_heads)
    T2* __restrict__ logit,    // (len_buf)
    int len_buf,
    int stride, // num_kv_heads * dim_head
    int stride_scale = 0 // num_kv_heads
    ) {
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpNum = blockDim.x / WARP_SIZE;

    // typedef typename Half2Type<T>::T_HALF2 T_HALF2;
    typedef half2 T_HALF2;
    __align__(8) T_HALF2 q_h22[2];
    __align__(8) T_HALF2 k_h22[2];

    *reinterpret_cast<short4*>(&q_h22) = *reinterpret_cast<const short4*>(g_q + laneId * 4);
    for (int col = warpId; col < len_buf; col += warpNum) {
        uint32_t b4 = *reinterpret_cast<const uint32_t*>(g_k_quant + col * stride + laneId * 4);
        DEV_dequant_int8(b4, k_h22);
        T_HALF2 r0 = __hmul2(q_h22[0], k_h22[0]);
        T_HALF2 r1 = __hmul2(q_h22[1], k_h22[1]);
        T_HALF2 r = __hadd2(r0, r1);
        float res = float(__low2float(r)) + float(__high2float(r));
        res = bmengine::functions::warpReduceSum<float>(res);
        if (laneId == 0) {
            // logit[col] = T2(res * g_k_scale[col * stride_scale]);
            logit[col] = res;
        }
    }
}

template<typename T, typename T2 = T>
static __forceinline__ __device__ void DEV_mul_logit_scale(
    T* logit,           // (len_buf)
    const float* scale, // (len_buf, num_kv_heads)
    int len_buf,
    int stride
) {
    assert(len_buf < blockDim.x);
    if (threadIdx.x < len_buf) {
        logit[threadIdx.x] = T(logit[threadIdx.x] * scale[threadIdx.x * stride]);
    }
}

template<typename T, int DIM_HEAD = 128, typename T2 = T, int NUM_SPLIT = 1024 / DIM_HEAD>
static __device__ void DEV_mul_score_v_v1(
    const float* score,        // (len_buf)
    const uint8_t* __restrict__ g_v, // (len_buf, DIM_HEAD)
    T2* __restrict__ output,   // (DIM_HEAD)
    int len_buf,
    int stride_v = DIM_HEAD) {
    static_assert((DIM_HEAD % WARP_SIZE) == 0);
    static_assert((1024 % DIM_HEAD) == 0);
    assert(blockDim.x == 1024);
    const int len_buf_s = (len_buf + NUM_SPLIT - 1) / NUM_SPLIT;

    const int shard = threadIdx.x / DIM_HEAD;
    const int d = threadIdx.x % DIM_HEAD;

    float res = 0;
    int start = shard * len_buf_s;
    int end = min(start + len_buf_s, len_buf);
    for (int i = start; i < end; ++i) {
        float dq = float(g_v[i * stride_v + d]) - 128.f;
        res += float(score[i]) * dq;
    }
    // reduce sum of all shards
    static __shared__ float tmp[1024];
    tmp[d * NUM_SPLIT + shard] = res;
    __syncthreads();
    float x = tmp[threadIdx.x];
    if (NUM_SPLIT == 16)
        x += __shfl_down_sync(0xFFFFFFFF, x, 8);
    if (NUM_SPLIT >= 8)
        x += __shfl_down_sync(0xFFFFFFFF, x, 4);
    if (NUM_SPLIT >= 4)
        x += __shfl_down_sync(0xFFFFFFFF, x, 2);
    x += __shfl_down_sync(0xFFFFFFFF, x, 1);
    if ((threadIdx.x % NUM_SPLIT) == 0) {
        output[threadIdx.x / NUM_SPLIT] = T2(x);
    }
}
