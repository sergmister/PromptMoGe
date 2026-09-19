// Robust metric attachment on the GPU: trimmed least squares  target ~= s * z + t  over a mask -- four trim
// iterations, then a final solve -- reproducing the reference `fit_scale_shift` step for step.
//
// The only non-local operation is the trim threshold, the k-th smallest relative residual. It is found with a
// two-level histogram on u = r / (1 + r), which maps [0, inf) onto [0, 1) monotonically (so no max pass is needed):
// 4096 bins over [0, 1), then 4096 bins inside the bin that holds the k-th value. Every step, including the solve,
// is a kernel, and the state -- s, t, the histogram window, k -- lives in one small buffer, so neither the per-step
// refit inside the refiner nor the final fit needs a host round trip.
//
// Serial loops inside a GPU thread are what cost time here, not the atomic histograms: the sums therefore run one
// simdgroup per 4096-element chunk with each lane walking its own stride, and the k-th scan exits early.

#include <metal_stdlib>
using namespace metal;

struct FitState {
    float s; float t; float lo; float hi;
    uint count; uint k; uint ok; uint iter;
    uint n; uint chunk; uint nchunks; uint bins;
};

struct FitPrompt { uint H; uint W; uint dh; uint dw; };

// The refiner's prompt fit (_lidar_in_affine_frame): z = exp(logz) at the point map, target = nearest LiDAR,
// mask = valid & conf == 2.
kernel void fit_prep_prompt(device const float* logz  [[buffer(0)]],
                            device const float* lidar [[buffer(1)]],
                            device const uchar* conf  [[buffer(2)]],
                            device float*       z     [[buffer(3)]],
                            device float*       tg    [[buffer(4)]],
                            device uchar*       mask  [[buffer(5)]],
                            device uchar*       w     [[buffer(6)]],
                            constant FitPrompt& P     [[buffer(7)]],
                            uint m [[thread_position_in_grid]])
{
    const uint N = P.H * P.W;
    if (m >= N) return;
    const uint y = m / P.W, x = m % P.W;
    const uint sy = min(P.dh - 1, (y * P.dh) / P.H);
    const uint sx = min(P.dw - 1, (x * P.dw) / P.W);
    const float d = lidar[sy * P.dw + sx];
    const float c = clamp(float(conf[sy * P.dw + sx]) / 2.0f, 0.0f, 1.0f);
    const bool ok = isfinite(d) && d > 0.0f && c >= 0.999f;
    z[m] = exp(logz[m]);
    tg[m] = ok ? d : 0.0f;
    mask[m] = ok ? 1 : 0;
    w[m] = mask[m];
}

// The final metric fit (infer, metric_from='lidar_ls'): z = area mean of exp(logz) over the point-map window of
// each LiDAR pixel (adaptive_avg_pool2d windows, so a non-integer ratio is exact), mask = conf >= 2 & LiDAR > 0 &
// z > 0 & model-mask area > 0.99.
kernel void fit_prep_final(device const float* logz      [[buffer(0)]],
                           device const half*  maskLogit [[buffer(1)]],
                           device const float* lidar     [[buffer(2)]],
                           device const uchar* conf      [[buffer(3)]],
                           device float*       z         [[buffer(4)]],
                           device float*       tg        [[buffer(5)]],
                           device uchar*       mask      [[buffer(6)]],
                           device uchar*       w         [[buffer(7)]],
                           constant FitPrompt& P         [[buffer(8)]],
                           uint2 g [[thread_position_in_grid]])
{
    const uint px = g.x, py = g.y;
    if (px >= P.dw || py >= P.dh) return;
    const uint ys = (py * P.H) / P.dh, ye = min(P.H, ((py + 1) * P.H + P.dh - 1) / P.dh);
    const uint xs = (px * P.W) / P.dw, xe = min(P.W, ((px + 1) * P.W + P.dw - 1) / P.dw);
    float acc = 0.0f, macc = 0.0f;
    uint cnt = 0;
    for (uint yy = ys; yy < ye; ++yy)
        for (uint xx = xs; xx < xe; ++xx) {
            acc += exp(logz[yy * P.W + xx]);
            macc += (maskLogit[yy * P.W + xx] > 0.0h) ? 1.0f : 0.0f;
            cnt++;
        }
    const uint p = py * P.dw + px;
    const float zl = acc / float(max(cnt, 1u));
    const float ma = macc / float(max(cnt, 1u));
    const float d = lidar[p];
    const bool ok = conf[p] >= 2 && d > 0.0f && zl > 0.0f && ma > 0.99f;
    z[p] = zl;
    tg[p] = ok ? d : 0.0f;
    mask[p] = ok ? 1 : 0;
    w[p] = mask[p];
}

// Masked (first solve) or trimmed (later solves) sums per chunk: n, sx, sy, sxx, sxy.
kernel void fit_sums(device const float* z      [[buffer(0)]],
                      device const float* tg     [[buffer(1)]],
                      device const float* u      [[buffer(2)]],
                      device const uchar* mask   [[buffer(3)]],
                      device float*       sums   [[buffer(4)]],
                      constant FitState& S      [[buffer(5)]],
                      constant uint&      onMask [[buffer(6)]],
                      uint c      [[threadgroup_position_in_grid]],
                      uint lane   [[thread_index_in_simdgroup]],
                      uint nlanes [[threads_per_simdgroup]])
{
    const uint lo = c * S.chunk, hi = min(lo + S.chunk, S.n);
    float n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    if (c < S.nchunks) {
        for (uint i = lo + lane; i < hi; i += nlanes) {
            const bool keep = onMask ? (mask[i] != 0) : (u[i] <= S.hi);
            if (!keep) continue;
            const float a = z[i], b = tg[i];
            n += 1; sx += a; sy += b; sxx += a * a; sxy += a * b;
        }
    }
    n = simd_sum(n); sx = simd_sum(sx); sy = simd_sum(sy); sxx = simd_sum(sxx); sxy = simd_sum(sxy);
    if (lane == 0 && c < S.nchunks) {
        sums[c * 5] = n; sums[c * 5 + 1] = sx; sums[c * 5 + 2] = sy; sums[c * 5 + 3] = sxx; sums[c * 5 + 4] = sxy;
    }
}

// Solve, and on the first call record the mask count, `ok`, and k = floor(0.8 * count) clamped to 32.
kernel void fit_solve(device const float* sums [[buffer(0)]],
                      device FitState&    S    [[buffer(1)]],
                      uint g [[thread_position_in_grid]])
{
    if (g != 0) return;
    float n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (uint c = 0; c < S.nchunks; ++c) {
        n += sums[c * 5]; sx += sums[c * 5 + 1]; sy += sums[c * 5 + 2];
        sxx += sums[c * 5 + 3]; sxy += sums[c * 5 + 4];
    }
    if (S.iter == 0) {
        S.count = uint(n);
        S.ok = (n >= 32.0f) ? 1u : 0u;
        S.k = max(uint(0.8f * float(S.count)), 32u);
        S.s = 1.0f; S.t = 0.0f;
    }
    const float nn = max(n, 1.0f);
    const float den = nn * sxx - sx * sx;
    const float sNew = (den > 1e-12f) ? (nn * sxy - sx * sy) / max(den, 1e-12f) : 1.0f;
    const float tNew = (sy - sNew * sx) / nn;
    if (S.ok != 0u) { S.s = sNew; S.t = tNew; }
    S.iter += 1;
}

// residual + level-1 histogram over u in [0, 1)
kernel void fit_resid_hist(device const float* z    [[buffer(0)]],
                            device const float* tg   [[buffer(1)]],
                            device const uchar* mask [[buffer(2)]],
                            device float*       u    [[buffer(3)]],
                            device atomic_uint* hist [[buffer(4)]],
                            constant FitState& S    [[buffer(5)]],
                            uint i [[thread_position_in_grid]])
{
    if (i >= S.n) return;
    if (mask[i] == 0) { u[i] = 2.0f; return; }
    const float r = abs(S.s * z[i] + S.t - tg[i]) / max(tg[i], 1e-3f);
    const float v = r / (1.0f + r);
    u[i] = v;
    const uint b = min(S.bins - 1, uint(v * float(S.bins)));
    atomic_fetch_add_explicit(&hist[b], 1u, memory_order_relaxed);
}

// level-2 histogram inside [lo, hi]; hist[bins] counts those below lo
kernel void fit_hist2(device const float* u    [[buffer(0)]],
                       device atomic_uint* hist [[buffer(1)]],
                       constant FitState& S    [[buffer(2)]],
                       uint i [[thread_position_in_grid]])
{
    if (i >= S.n) return;
    const float v = u[i];
    if (v > 1.5f) return;                                   // unmasked
    if (v < S.lo) { atomic_fetch_add_explicit(&hist[S.bins], 1u, memory_order_relaxed); return; }
    if (v > S.hi) return;
    const uint b = min(S.bins - 1, uint((v - S.lo) / (S.hi - S.lo) * float(S.bins)));
    atomic_fetch_add_explicit(&hist[b], 1u, memory_order_relaxed);
}

// Narrow [lo, hi] to the bin that holds the k-th smallest value. `level1` selects the [0, 1) window.
kernel void fit_kth(device const uint* hist   [[buffer(0)]],
                     device FitState&  S      [[buffer(1)]],
                     constant uint&     level1 [[buffer(2)]],
                     uint g [[thread_position_in_grid]])
{
    if (g != 0) return;
    const uint k = min(S.k, S.n);
    const float lo = level1 ? 0.0f : S.lo, hi = level1 ? 1.0f : S.hi;
    uint cum = level1 ? 0u : hist[S.bins];
    const float width = (hi - lo) / float(S.bins);
    for (uint b = 0; b < S.bins; ++b) {
        cum += hist[b];
        if (cum >= k) { S.lo = lo + width * float(b); S.hi = lo + width * float(b + 1); return; }
    }
    S.lo = lo; S.hi = hi;
}

// hist[bins] counts residuals in [lo, hi]; hist[bins] (one past the end) counts those below lo.
kernel void fit_hist_clear(device uint*       hist [[buffer(0)]],
                           constant FitState& S    [[buffer(1)]],
                           uint g [[thread_position_in_grid]])
{
    if (g > S.bins) return;
    hist[g] = 0u;
}

// Metric depth from the refined log-depth, the final (s, t) and the model's mask; +inf where invalid.
kernel void post_depth(device const float*  logz      [[buffer(0)]],
                       device const half*   maskLogit [[buffer(1)]],
                       device float*        depth     [[buffer(2)]],
                       constant FitState& F       [[buffer(3)]],
                       constant uint&       N         [[buffer(4)]],
                       uint m [[thread_position_in_grid]])
{
    if (m >= N) return;
    const float d = F.s * exp(logz[m]) + F.t;
    depth[m] = (maskLogit[m] > 0.0h && d > 0.0f) ? d : INFINITY;
}
