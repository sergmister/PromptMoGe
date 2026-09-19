// The sparse refiner's graph around its convolutions (SparseConvInt8.metal), written for persistent buffers and
// as few full passes over a level as possible: the input projection reads the planar point map directly, the
// LiDAR residual is computed on the GPU from the fit's state buffer, every dense layer the model quantises runs
// as an int8 GEMM, and pooling / upsampling are fused with the quantisation or skip add that follows them.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

// ---------------------------------------------------------------------------------------------------------- //
// prompt_feats: MoGeModel._lidar_in_affine_frame with soft_residual_gate, per voxel of level 0.
//   d = nearest(lidar), c = nearest(conf)/2 clamped, valid = finite & d > 0, d_aff = (d - t) / max(s, 1e-6),
//   ok = valid & d_aff > 1e-4, log_d_aff = ok ? log(max(d_aff, 1e-4)) : logz,
//   gate = (0.15 + 0.85 c) * ok, residual = (log_d_aff - logz) * gate
// Nearest is F.interpolate's floor(dst * in / out), taken in integer arithmetic.
struct PromptParams { uint H; uint W; uint dh; uint dw; float s; float t; uint useFit; uint pad1; };
struct FitStateP { float s; float t; float lo; float hi; uint count; uint k; uint ok; uint iter;
                   uint n; uint chunk; uint nchunks; uint bins; };

kernel void rf_prompt_feats(device const float*   logz   [[buffer(0)]],
                            device const float*   lidar  [[buffer(1)]],   // [dh*dw] metres
                            device const uchar*   conf   [[buffer(2)]],   // [dh*dw] {0,1,2}
                            device float*         out    [[buffer(3)]],   // [N][2]
                            constant PromptParams& P     [[buffer(4)]],
                            constant FitStateP&   F      [[buffer(5)]],
                            uint m [[thread_position_in_grid]])
{
    const uint N = P.H * P.W;
    const float fs = P.useFit ? F.s : P.s, ft = P.useFit ? F.t : P.t;
    if (m >= N) return;
    const uint y = m / P.W, x = m % P.W;
    const uint sy = min(P.dh - 1, (y * P.dh) / P.H);
    const uint sx = min(P.dw - 1, (x * P.dw) / P.W);
    const float d = lidar[sy * P.dw + sx];
    const float c = clamp(float(conf[sy * P.dw + sx]) / 2.0f, 0.0f, 1.0f);
    const bool valid = isfinite(d) && d > 0.0f;
    const float dAff = (d - ft) / max(fs, 1e-6f);
    const bool ok = valid && dAff > 1e-4f;
    const float lz = logz[m];
    const float logDAff = ok ? log(max(dAff, 1e-4f)) : lz;
    const float gate = ok ? (0.15f + 0.85f * c) : 0.0f;
    out[m * 2] = (logDAff - lz) * gate;
    out[m * 2 + 1] = gate;
}

// input_proj(x/z, y/z, logz) + prompt_proj(residual, gate), straight from the planar point map: no packing.
// Weights in the GEMM layout [K][N]: w[k*N + n].
struct InputParams { uint N; uint C; uint pad0; uint pad1; };

kernel void rf_input(device const float* xy     [[buffer(0)]],   // [2][N] planar
                     device const float* logz   [[buffer(1)]],   // [N]
                     device const float* prompt [[buffer(2)]],   // [N][2]
                     device const half*  wi     [[buffer(3)]],   // [3][C]
                     device const half*  bi     [[buffer(4)]],
                     device const half*  wp     [[buffer(5)]],   // [2][C]
                     device const half*  bp     [[buffer(6)]],
                     device half*        out    [[buffer(7)]],   // [N][C]
                     constant InputParams& P    [[buffer(8)]],
                     uint m [[thread_position_in_grid]])
{
    if (m >= P.N) return;
    const float x = xy[m], y = xy[P.N + m], z = logz[m];
    const float r = prompt[m * 2], g = prompt[m * 2 + 1];
    const uint C = P.C;
    for (uint n = 0; n < C; ++n) {
        const float v = float(bi[n]) + float(wi[n]) * x + float(wi[C + n]) * y + float(wi[2 * C + n]) * z
                      + float(bp[n]) + float(wp[n]) * r + float(wp[C + n]) * g;
        out[m * C + n] = half(v);
    }
}

// LayerNorm + SiLU + quantise for conv1, block-staged: the threadgroup block-copies R contiguous rows in 16-byte
// units, reduces out of threadgroup memory (one simdgroup per row, each lane walking its own channel stride) and
// block-writes the int8 result. R is chosen so that R * C = 4096.
struct RowOpParams { int M; int C; float eps; float invScale; };

#define LN_SILU_BLK(NAME, OUT_T, QUANT, R)                                           \
kernel void NAME(device const half* in     [[buffer(0)]],                            \
                 device OUT_T*      out    [[buffer(1)]],                            \
                 device const half* gamma  [[buffer(2)]],                            \
                 device const half* beta   [[buffer(3)]],                            \
                 constant RowOpParams& P   [[buffer(4)]],                            \
                 threadgroup half*  tile   [[threadgroup(0)]],                       \
                 threadgroup OUT_T* otile  [[threadgroup(1)]],                       \
                 uint  tgi  [[threadgroup_position_in_grid]],                        \
                 uint  tid  [[thread_index_in_threadgroup]],                         \
                 uint  sgid [[simdgroup_index_in_threadgroup]],                      \
                 uint  lane [[thread_index_in_simdgroup]],                           \
                 uint  nsg  [[simdgroups_per_threadgroup]],                          \
                 uint  tcount [[threads_per_threadgroup]])                           \
{                                                                                    \
    const int row0 = int(tgi) * R;                                                   \
    const int rows = min(R, P.M - row0);                                             \
    if (rows <= 0) return;                                                           \
    const int n16 = (rows * P.C) / 8;                                                \
    device const uint4* src = (device const uint4*)(in + row0 * P.C);                \
    threadgroup uint4* dst = (threadgroup uint4*)tile;                               \
    for (uint e = tid; e < uint(n16); e += tcount) dst[e] = src[e];                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                 \
    for (int rr = int(sgid); rr < rows; rr += int(nsg)) {                            \
        threadgroup const half* r = tile + rr * P.C;                                 \
        float s = 0.0f, ss = 0.0f;                                                   \
        for (int c = int(lane); c < P.C; c += 32) {                                  \
            float x = float(r[c]); s += x; ss += x * x;                              \
        }                                                                             \
        s = simd_sum(s); ss = simd_sum(ss);                                           \
        const float nn = float(P.C);                                                  \
        const float mean = s / nn;                                                    \
        const float rstd = rsqrt(max(ss / nn - mean * mean, 0.0f) + P.eps);            \
        threadgroup OUT_T* o = otile + rr * P.C;                                      \
        for (int c = int(lane); c < P.C; c += 32) {                                   \
            float y = (float(r[c]) - mean) * rstd * float(gamma[c]) + float(beta[c]);   \
            y = y / (1.0f + exp(-y));                                                   \
            o[c] = QUANT ? OUT_T(clamp(rint(y * P.invScale), -127.0f, 127.0f))          \
                         : OUT_T(y);                                                     \
        }                                                                                \
    }                                                                                     \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                      \
    const int obytes = rows * P.C * int(sizeof(OUT_T));                                   \
    device uint4* od = (device uint4*)(out + row0 * P.C);                                 \
    threadgroup const uint4* os = (threadgroup const uint4*)otile;                        \
    for (uint e = tid; e < uint(obytes / 16); e += tcount) od[e] = os[e];                 \
}

LN_SILU_BLK(ln_silu_i8_r128, int8_t, 1, 128)
LN_SILU_BLK(ln_silu_i8_r64, int8_t, 1, 64)
LN_SILU_BLK(ln_silu_i8_r32, int8_t, 1, 32)
LN_SILU_BLK(ln_silu_i8_r16, int8_t, 1, 16)

// Pool mean, quantised in the same pass for the int8 linear that consumes it. Each child row is read once, four
// halves at a time, into a stack accumulator.
struct PoolQParams { int Min; int Mout; int C; float invScale; };

kernel void pool_mean_q8(device const half* in     [[buffer(0)]],
                          device const int*  segIdx [[buffer(1)]],
                          device const int*  segOff [[buffer(2)]],
                          device int8_t*     out    [[buffer(3)]],
                          constant PoolQParams& P  [[buffer(4)]],
                          uint o [[thread_position_in_grid]])
{
    if (int(o) >= P.Mout) return;
    const int lo = segOff[o], hi = segOff[o + 1];
    const uint C = uint(P.C), V = C / 4;            // C is a multiple of 32
    float4 acc[64];
    for (uint v = 0; v < V; ++v) acc[v] = float4(0.0f);
    for (int k = lo; k < hi; ++k) {
        device const half4* row = (device const half4*)(in + uint(segIdx[k]) * C);
        for (uint v = 0; v < V; ++v) acc[v] += float4(row[v]);
    }
    const float sc = P.invScale / float(max(hi - lo, 1));
    device int8_t* d = out + o * C;
    for (uint v = 0; v < V; ++v) {
        const float4 q = clamp(rint(acc[v] * sc), -127.0f, 127.0f);
        d[v * 4] = int8_t(q.x); d[v * 4 + 1] = int8_t(q.y); d[v * 4 + 2] = int8_t(q.z); d[v * 4 + 3] = int8_t(q.w);
    }
}

// fp16 -> int8, per tensor.
struct QuantParams { uint n; float invScale; uint pad0; uint pad1; };

kernel void quant_h_i8(device const half* in  [[buffer(0)]],
                       device int8_t*     out [[buffer(1)]],
                       constant QuantParams& P [[buffer(2)]],
                       uint i [[thread_position_in_grid]])
{
    if (i >= P.n) return;
    out[i] = int8_t(clamp(rint(float(in[i]) * P.invScale), -127.0f, 127.0f));
}

// Int8 dense GEMM:  out[m][n] = float(sum_k a8[m][k] * w8[k][n]) * aScale * wScale[n] + bias[n]
// MODE 0: fp16 out.   MODE 1: SiLU, then requantise to int8 with oInvScale (fuse_proj.0 -> fuse_proj.2).
// Tiled like the convolution, minus the gather: TN = N reads the A tile once per row tile, staged in 16-byte moves.
struct DG8Params { int M; int N; int K; float aScale; float oInvScale; int pad0; int pad1; int pad2; };

#define DGEMM8(NAME, TM, TN, TK, SG, MODE)                                                    \
kernel void NAME(                                                                             \
    device int8_t*       A      [[buffer(0)]],                                                \
    device int8_t*       B      [[buffer(1)]],                                                \
    device const float*  wScale [[buffer(2)]],                                                \
    device const half*   bias   [[buffer(3)]],                                                \
    device half*         C16    [[buffer(4)]],                                                \
    device int8_t*       C8     [[buffer(5)]],                                                \
    constant DG8Params&  P      [[buffer(6)]],                                                \
    threadgroup int8_t*  Atile  [[threadgroup(0)]],                                           \
    uint2 tgid    [[threadgroup_position_in_grid]],                                           \
    uint2 tid2    [[thread_position_in_threadgroup]],                                         \
    uint2 tcount2 [[threads_per_threadgroup]])                                                \
{                                                                                             \
    const uint tid = tid2.x, tcount = tcount2.x;                                              \
    const int m0 = int(tgid.y) * TM;                                                          \
    const int n0 = int(tgid.x) * TN;                                                          \
    if (m0 >= P.M) return;                                                                    \
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, false, false,                \
                              matmul2d_descriptor::mode::multiply_accumulate);                \
    matmul2d<desc, execution_simdgroups<SG>> op;                                              \
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>                           \
        At(Atile, dextents<int32_t, 2>(TK, TM));                                              \
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>                                \
        Bfull(B, dextents<int32_t, 2>(P.N, P.K));                                             \
    auto acc = op.get_destination_cooperative_tensor<                                         \
        metal::remove_addrspace_t<decltype(At)>,                                              \
        metal::remove_addrspace_t<decltype(Bfull)>, int32_t>();                               \
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)                                         \
        if (acc.is_valid_element(i)) acc[i] = 0;                                              \
    constexpr int VEC = TK / 16;                                                              \
    for (int k0 = 0; k0 < P.K; k0 += TK) {                                                    \
        for (uint e = tid; e < uint(TM * VEC); e += tcount) {                                  \
            const int r = int(e) / VEC, kv = int(e) % VEC;                                     \
            const int row = m0 + r;                                                             \
            threadgroup uint4* dst = (threadgroup uint4*)(Atile + r * TK) + kv;                 \
            *dst = (row < P.M) ? *((device const uint4*)(A + row * P.K + k0) + kv) : uint4(0);  \
        }                                                                                       \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                        \
        auto tA = At.slice(0, 0);                                                               \
        auto tB = Bfull.slice(n0, k0);                                                          \
        op.run(tA, tB, acc);                                                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                        \
    }                                                                                           \
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)                                           \
        if (acc.is_valid_element(i)) {                                                          \
            auto ix = acc.get_multidimensional_index(i);                                        \
            const int n = n0 + int(ix[0]);                                                      \
            const int m = m0 + int(ix[1]);                                                      \
            if (n < P.N && m < P.M) {                                                           \
                float y = float(acc[i]) * P.aScale * wScale[n] + float(bias[n]);                \
                if (MODE == 1) {                                                                \
                    y = y / (1.0f + exp(-y));                                                   \
                    C8[m * P.N + n] = int8_t(clamp(rint(y * P.oInvScale), -127.0f, 127.0f));    \
                } else {                                                                        \
                    C16[m * P.N + n] = half(y);                                                 \
                }                                                                               \
            }                                                                                   \
        }                                                                                       \
}

// The (N, K) shapes the refiners use. Tile rule: TM * TN / SG >= 2048, TK | K, 16 | TK.
DGEMM8(dg8_n64_k32,     64,  64,  32, 2, 0)   // pool 32 -> 64
DGEMM8(dg8_n128_k64,    32, 128,  64, 2, 0)   // pool 64 -> 128
DGEMM8(dg8_n256_k128,   32, 256, 128, 4, 0)   // pool 128 -> 256
DGEMM8(dg8_n256_k256,   32, 256, 128, 4, 0)   // pool / up 256 -> 256, fuse_proj.2
DGEMM8(dg8_n128_k256,   32, 128, 128, 2, 0)   // up 256 -> 128
DGEMM8(dg8_n64_k128,    64,  64, 128, 2, 0)   // up 128 -> 64
DGEMM8(dg8_n32_k64,     64,  32,  64, 1, 0)   // up 64 -> 32
DGEMM8(dg8_n256_k1024,  32, 256, 128, 4, 0)   // encoder_fuse, once per frame at the token grid
DGEMM8(dg8q_n256_k512,  32, 256, 128, 4, 1)   // fuse_proj.0 + SiLU + requantise

// Encoder conditioning: rows gathered from the per-frame token-grid result by the voxel's column id.
struct GatherParams { uint M; uint C; uint pad0; uint pad1; };

kernel void gather_rows_h(device const half* tok [[buffer(0)]],   // [T][C]
                          device const uint* cid [[buffer(1)]],   // [M]
                          device half*       out [[buffer(2)]],   // [M][C]
                          constant GatherParams& P [[buffer(3)]],
                          uint o [[thread_position_in_grid]])
{
    if (o >= P.M) return;
    device const half* s = tok + cid[o] * P.C;
    device half* d = out + o * P.C;
    for (uint c = 0; c < P.C; ++c) d[c] = s[c];
}

// The head emits the encoder feature planar, [K][T]; the int8 GEMM wants rows, [T][K]. Quantise and transpose
// in one pass instead of reordering 1.2 M values on the host every frame.
struct PlanarQParams { uint T; uint K; float invScale; uint pad; };

kernel void quant_planar_f_i8(device const float*    in  [[buffer(0)]],   // [K][T]
                              device int8_t*         out [[buffer(1)]],   // [T][K]
                              constant PlanarQParams& P  [[buffer(2)]],
                              uint2 g [[thread_position_in_grid]])        // x = k, y = t
{
    if (g.x >= P.K || g.y >= P.T) return;
    out[g.y * P.K + g.x] = int8_t(clamp(rint(in[g.x * P.T + g.y] * P.invScale), -127.0f, 127.0f));
}

// Token rows += a planar [C][T] map (the UV half of the encoder conditioning, baked at export).
kernel void add_planar_f_to_h(device half*           io  [[buffer(0)]],   // [T][C]
                              device const float*    add [[buffer(1)]],   // [C][T]
                              constant PlanarQParams& P  [[buffer(2)]],   // K = C
                              uint2 g [[thread_position_in_grid]])
{
    if (g.x >= P.K || g.y >= P.T) return;
    const uint i = g.y * P.K + g.x;
    io[i] = half(float(io[i]) + add[g.x * P.T + g.y]);
}

// cat([a, b]) quantised with one per-tensor scale -- fuse_proj.0's input.
struct CatQParams { uint M; uint C1; uint C2; float invScale; };

kernel void concat_q8(device const half* a   [[buffer(0)]],
                      device const half* b   [[buffer(1)]],
                      device int8_t*     out [[buffer(2)]],
                      constant CatQParams& P [[buffer(3)]],
                      uint o [[thread_position_in_grid]])
{
    if (o >= P.M) return;
    const uint C = P.C1 + P.C2;
    for (uint c = 0; c < P.C1; ++c)
        out[o * C + c] = int8_t(clamp(rint(float(a[o * P.C1 + c]) * P.invScale), -127.0f, 127.0f));
    for (uint c = 0; c < P.C2; ++c)
        out[o * C + P.C1 + c] = int8_t(clamp(rint(float(b[o * P.C2 + c]) * P.invScale), -127.0f, 127.0f));
}

// Nearest upsample and skip add in one pass, four halves at a time: out[m] = in[parent[m]] + skip[m].
struct UpParams { uint Mout; uint C; uint pad0; uint pad1; };

kernel void up_add_h(device const half* in     [[buffer(0)]],
                      device const int*  parent [[buffer(1)]],
                      device const half* skip   [[buffer(2)]],
                      device half*       out    [[buffer(3)]],
                      constant UpParams& P     [[buffer(4)]],
                      uint m [[thread_position_in_grid]])
{
    if (m >= P.Mout) return;
    const uint V = P.C / 4;
    device const half4* s = (device const half4*)(in + uint(parent[m]) * P.C);
    device const half4* k = (device const half4*)(skip + m * P.C);
    device half4* d = (device half4*)(out + m * P.C);
    for (uint v = 0; v < V; ++v) d[v] = s[v] + k[v];
}

// out_proj (C -> 1) straight into a float32 delta, and the log-depth update for the next step.
kernel void rf_out(device const half* feats [[buffer(0)]],
                   device const half* w     [[buffer(1)]],   // [C][1]
                   device const half* b     [[buffer(2)]],
                   device float*      delta [[buffer(3)]],
                   constant InputParams& P  [[buffer(4)]],
                   uint m [[thread_position_in_grid]])
{
    if (m >= P.N) return;
    float acc = float(b[0]);
    device const half* f = feats + m * P.C;
    for (uint c = 0; c < P.C; ++c) acc += float(f[c]) * float(w[c]);
    delta[m] = acc;
}

kernel void rf_apply(device float*       logz  [[buffer(0)]],
                     device const float* delta [[buffer(1)]],
                     constant InputParams& P   [[buffer(2)]],
                     uint m [[thread_position_in_grid]])
{
    if (m >= P.N) return;
    logz[m] += delta[m];
}
