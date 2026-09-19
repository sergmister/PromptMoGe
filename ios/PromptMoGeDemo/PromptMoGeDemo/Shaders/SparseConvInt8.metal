// Submanifold sparse 3-D convolution, int8.
//
//   out[m] = bias + s_a * s_w[n] * sum_v  W8[:, v, :] . feats8[ nmap[v][m] ]        (index < 0: no neighbour)
//
// Activations are quantised per tensor and weights per output channel, which is exactly what the refiner was
// trained with, so the int32 accumulation below is the model's function, not an approximation of it. A per-row
// activation scale is not usable: the sum runs over neighbours gathered from different rows.
//
// The kernel is a tiled GEMM (MetalPerformancePrimitives matmul2d, multiply-accumulate) and everything else about
// it is memory flow:
//   * a threadgroup owns TM output rows x TN output channels; the input tile is gathered once per tap, in 16-byte
//     units, from a TAP-MAJOR neighbour map [V][M] whose indices are hoisted into threadgroup memory first;
//   * which of the 27 taps a tile must issue comes from a per-tile bitmask written by the structure build, so
//     empty taps cost nothing;
//   * bias, activation and the next layer's quantisation (conv1) or the residual add (conv2) are folded into the
//     epilogue, so a residual block is three dispatches and nothing between them touches memory twice:
//
//         ln_silu_i8(x) -> int8      conv1 [SPCONV_I8Q] -> int8      conv2 [SPCONV_I8R] -> fp16 + x

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct ConvParams {
    int M;      // output voxels
    int Ci;     // input channels
    int Co;     // output channels
    int V;      // kernel volume (27)
    int pad0, pad1, pad2, pad3;
};

// Gather + int8 GEMM for one (row tile, channel tile) into the cooperative int32 accumulator `acc`.
#define SPCONV_I8_ACCUMULATE(TM, TN, TK, SG)                                                 \
    const uint tid = tid2.x, tcount = tcount2.x;                                            \
    const int m0 = int(tgid.y) * TM;                                                        \
    const int n0 = int(tgid.x) * TN;                                                        \
    if (m0 >= P.M) return;                                                                  \
    threadgroup int srcIdx[TM];                                                             \
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, false, false,               \
                              matmul2d_descriptor::mode::multiply_accumulate);              \
    matmul2d<desc, execution_simdgroups<SG>> op;                                            \
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>                         \
        A(Atile, dextents<int32_t, 2>(TK, TM));                                             \
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>                              \
        Bfull(wB8, dextents<int32_t, 2>(P.Co, P.V * P.Ci));                                 \
    auto acc = op.get_destination_cooperative_tensor<                                       \
        metal::remove_addrspace_t<decltype(A)>,                                             \
        metal::remove_addrspace_t<decltype(Bfull)>, int32_t>();                             \
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)                                       \
        if (acc.is_valid_element(i)) acc[i] = 0;                                            \
    constexpr int VEC = TK / 16;                                                             \
    uint mask = tileTaps[tgid.y];                                                            \
    while (mask != 0u) {                                                                     \
        const int v = int(ctz(mask));                                                        \
        mask &= (mask - 1u);                                                                 \
        /* This must loop: a tile can have more rows than the threadgroup has threads. */    \
        for (uint q = tid; q < uint(TM); q += tcount) {                                      \
            const int r = m0 + int(q);                                                        \
            srcIdx[q] = (r < P.M) ? nmapT[v * P.M + r] : -1;                                  \
        }                                                                                     \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                      \
        for (int k0 = 0; k0 < P.Ci; k0 += TK) {                                               \
            for (uint e = tid; e < uint(TM * VEC); e += tcount) {                              \
                const int r = int(e) / VEC, kv = int(e) % VEC;                                 \
                const int src = srcIdx[r];                                                     \
                threadgroup uint4* dst = (threadgroup uint4*)(Atile + r * TK) + kv;            \
                *dst = (src >= 0)                                                              \
                     ? *((device const uint4*)(feats8 + src * P.Ci + k0) + kv)                 \
                     : uint4(0);                                                                \
            }                                                                                   \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
            auto tA = A.slice(0, 0);                                                            \
            auto tB = Bfull.slice(n0, v * P.Ci + k0);                                           \
            op.run(tA, tB, acc);                                                                \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
        }                                                                                        \
    }

// conv1: bias, SiLU, requantise for conv2 -> int8
#define SPCONV_I8Q(NAME, TM, TN, TK, SG)                                                     \
kernel void NAME(                                                                           \
    device int8_t*      feats8   [[buffer(0)]],                                             \
    device const int*   nmapT    [[buffer(1)]],                                             \
    device int8_t*      wB8      [[buffer(2)]],                                             \
    device const half*  bias     [[buffer(3)]],                                             \
    device int8_t*      out8     [[buffer(4)]],                                             \
    constant ConvParams& P       [[buffer(5)]],                                             \
    device const uint*  tileTaps [[buffer(6)]],                                             \
    device const float* wScale   [[buffer(7)]],  /* [Co] */                                 \
    constant float&     aScale   [[buffer(8)]],                                             \
    constant float&     oInvScale[[buffer(9)]],  /* 127 / clip of the next layer's input */ \
    threadgroup int8_t* Atile    [[threadgroup(0)]],                                        \
    uint2 tgid    [[threadgroup_position_in_grid]],                                         \
    uint2 tid2    [[thread_position_in_threadgroup]],                                       \
    uint2 tcount2 [[threads_per_threadgroup]])                                              \
{                                                                                           \
    SPCONV_I8_ACCUMULATE(TM, TN, TK, SG)                                                     \
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)                                        \
        if (acc.is_valid_element(i)) {                                                       \
            auto ix = acc.get_multidimensional_index(i);                                     \
            const int n = n0 + int(ix[0]);                                                   \
            const int m = m0 + int(ix[1]);                                                   \
            if (n < P.Co && m < P.M) {                                                       \
                float y = float(acc[i]) * aScale * wScale[n] + float(bias[n]);                \
                y = y / (1.0f + exp(-y));                                                     \
                out8[m * P.Co + n] = int8_t(clamp(rint(y * oInvScale), -127.0f, 127.0f));     \
            }                                                                                 \
        }                                                                                     \
}

// conv2: bias + block residual -> fp16
#define SPCONV_I8R(NAME, TM, TN, TK, SG)                                                     \
kernel void NAME(                                                                           \
    device int8_t*      feats8   [[buffer(0)]],                                             \
    device const int*   nmapT    [[buffer(1)]],                                             \
    device int8_t*      wB8      [[buffer(2)]],                                             \
    device const half*  bias     [[buffer(3)]],                                             \
    device half*        out16    [[buffer(4)]],                                             \
    constant ConvParams& P       [[buffer(5)]],                                             \
    device const uint*  tileTaps [[buffer(6)]],                                             \
    device const float* wScale   [[buffer(7)]],  /* [Co] */                                 \
    constant float&     aScale   [[buffer(8)]],                                             \
    device const half*  skip     [[buffer(9)]],                                             \
    threadgroup int8_t* Atile    [[threadgroup(0)]],                                        \
    uint2 tgid    [[threadgroup_position_in_grid]],                                         \
    uint2 tid2    [[thread_position_in_threadgroup]],                                       \
    uint2 tcount2 [[threads_per_threadgroup]])                                              \
{                                                                                           \
    SPCONV_I8_ACCUMULATE(TM, TN, TK, SG)                                                     \
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)                                        \
        if (acc.is_valid_element(i)) {                                                       \
            auto ix = acc.get_multidimensional_index(i);                                     \
            const int n = n0 + int(ix[0]);                                                   \
            const int m = m0 + int(ix[1]);                                                   \
            if (n < P.Co && m < P.M)                                                         \
                out16[m * P.Co + n] = half(float(acc[i]) * aScale * wScale[n] + float(bias[n]) \
                                           + float(skip[m * P.Co + n]));                     \
        }                                                                                     \
}

// One tile shape per channel width, tuned on the M5 GPU: a 32-row tile throughout (smaller tiles issue fewer
// wasted taps, larger ones re-read the weights less often), TN = Co where the threadgroup budget allows.
//            name                          TM   TN   TK  SG
SPCONV_I8Q(spconv_i8q_m32_n32_k32_sg1,      32,  32,  32, 1)
SPCONV_I8R(spconv_i8r_m32_n32_k32_sg1,      32,  32,  32, 1)
SPCONV_I8Q(spconv_i8q_m32_n64_k64_sg1,      32,  64,  64, 1)
SPCONV_I8R(spconv_i8r_m32_n64_k64_sg1,      32,  64,  64, 1)
SPCONV_I8Q(spconv_i8q_m32_n128_k128_sg2,    32, 128, 128, 2)
SPCONV_I8R(spconv_i8r_m32_n128_k128_sg2,    32, 128, 128, 2)
SPCONV_I8Q(spconv_i8q_m32_n256_k128_sg4,    32, 256, 128, 4)
SPCONV_I8R(spconv_i8r_m32_n256_k128_sg4,    32, 256, 128, 4)
