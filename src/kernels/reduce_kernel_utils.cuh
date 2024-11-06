#pragma once

#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>
#include <type_traits>
#include "utils/cuda_type_utils.cuh"

#if ((__CUDACC_VER_MAJOR__ > 11) || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MAJOR__ >= 0))
#include <cooperative_groups/reduce.h>
#else
#include <cooperative_groups.h>
#endif

namespace cg = cooperative_groups;

namespace space_llm {

template <int VPT>
struct BytesToType;

template <>
struct BytesToType<2> {
    using type = uint16_t;
};

template <>
struct BytesToType<4> {
    using type = uint32_t;
};

template <>
struct BytesToType<8> {
    using type = uint64_t;
};

template <>
struct BytesToType<16> {
    using type = float4;
};

static const float HALF_FLT_MAX = 65504.F;
#define FINAL_MASK 0xffffffff

// warp级别的shuffle操作进行规约求和
template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = add(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32)); // __shfl_sync bf16 return float when sm < 80
    }
    return val;
}

// block级别的规约求和
/* Calculate the sum of all elements in a block */
template <typename T>
__inline__ __device__ T blockReduceSum(T val) {
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum<T>(val);

    if (lane == 0) {
        shared[wid] = val;
    }

    __syncthreads();

    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
    val = warpReduceSum<T>(val);

    return val;
}

template <typename T>
__inline__ __device__ T warpReduceMax(T val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    }
    return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
__inline__ __device__ T blockReduceMax(T val) {
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f; // in-warp idx
    int wid = threadIdx.x >> 5;    // warp idx

    val = warpReduceMax(val); // get maxx in each warp

    if (lane == 0) // record in-warp maxx by warp Idx
        shared[wid] = val;

    __syncthreads();

    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
    val = warpReduceMax(val);

    return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
__inline__ __device__ T blockAllReduceMax(T val) {
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f; // in-warp idx
    int wid = threadIdx.x >> 5;    // warp idx

    val = warpReduceMax(val); // get maxx in each warp

    if (lane == 0) { // record in-warp maxx by warp idx
        shared[wid] = val;
    }

    __syncthreads();

    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (lane < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
    val = warpReduceMax(val);

    return val;
}

template <typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T *val) {
#pragma unroll
    for (int i = 0; i < NUM; i++) {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
        }
    }
    return (T)(0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceSumV2(T *val) {
    static __shared__ T shared[NUM][33];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    warpReduceSumV2<T, NUM>(val);

    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < NUM; i++) {
            shared[i][wid] = val[i];
        }
    }

    __syncthreads();

    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
    for (int i = 0; i < NUM; i++) {
        val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
    }
    warpReduceSumV2<T, NUM>(val);

    return (T)0.0f;
}

template <typename T, int NUM>
__inline__ __device__ T warpReduceMaxV2(T *val) {
#pragma unroll
    for (int i = 0; i < NUM; i++) {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            val[i] = max(val[i], __shfl_xor_sync(FINAL_MASK, val[i], mask, 32));
        }
    }
    return (T)(0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceMaxV2(T *val) {
    static __shared__ T shared[32][NUM];
    int lane = threadIdx.x & 0x1f; // in-warp idx
    int wid = threadIdx.x >> 5;    // warp idx

    warpReduceMaxV2<T, NUM>(val); // get maxx in each warp

    if (lane == 0) { // record in-warp maxx by warp Idx
#pragma unroll
        for (int i = 0; i < NUM; i++) {
            shared[wid][i] = val[i];
        }
    }
    __syncthreads();

    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
    for (int i = 0; i < NUM; i++) {
        val[i] = is_mask ? shared[lane][i] : (T)-1e20f;
    }
    warpReduceMaxV2<T, NUM>(val);

    return (T)0.0f;
}

} // namespace space_llm
