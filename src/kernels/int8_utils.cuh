#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

static inline __device__ int8_t float_to_int8_rn(float x) {
    uint32_t dst;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
    return reinterpret_cast<const int8_t &>(dst);
}

static inline __device__ uint32_t float4_to_char4(float x,
                                                  float y,
                                                  float z,
                                                  float w) {
    uint32_t dst;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 720
    uint32_t a;
    asm volatile("cvt.rni.sat.s32.f32 %0, %1;\n" : "=r"(a) : "f"(x));
    uint32_t b;
    asm volatile("cvt.rni.sat.s32.f32 %0, %1;\n" : "=r"(b) : "f"(y));
    uint32_t c;
    asm volatile("cvt.rni.sat.s32.f32 %0, %1;\n" : "=r"(c) : "f"(z));
    uint32_t d;
    asm volatile("cvt.rni.sat.s32.f32 %0, %1;\n" : "=r"(d) : "f"(w));

    asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2,  0;\n" : "=r"(dst) : "r"(d), "r"(c));
    asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, %0;\n" : "+r"(dst) : "r"(b), "r"(a));
#else
    char4 tmp;
    tmp.x = x;
    tmp.y = y;
    tmp.z = z;
    tmp.w = w;
    dst = reinterpret_cast<const uint32_t &>(tmp);
#endif
    return dst;
}
