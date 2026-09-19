// The refiner's sparse structure, built entirely on the GPU from the column layout.
//
// Every level of the SSR U-Net is a set of voxels (i, j, z) kept in lexicographic order -- that is the
// order `torch.unique(sorted=True)` produces and the order the parent maps index. Lexicographic order is
// exactly "raster over (i, j) columns, z ascending within a column", so a level is a CSR:
//
//     colStart[i*W + j] .. colStart[i*W + j + 1]   the voxels of column (i, j), z strictly ascending
//
// and everything the refiner needs follows locally from it, with no global sort and no hash:
//
//   pool k -> k+1   coarse column (a, b) owns fine columns (2a,2b) (2a,2b+1) (2a+1,2b) (2a+1,2b+1); its z set is
//                   the de-duplicated merge of their z >> 1, which is a 4-way merge of short sorted lists.
//   rulebook        a voxel's 27 neighbours live in 9 columns; each is found by a short in-column scan.
//
// A CPU build spends 86 % of its time in the neighbour maps and 12 % in pooling (a global sort of 307 200
// keys). Both are replaced here; the per-tile tap bitmask the int8 convolutions branch on is
// written by the rulebook itself, which removes the separate tile_or_mask pass.
//
// Level 0 is the same CSR with colStart = identity: `_voxelize` emits one voxel per pixel.
// Dispatch sizes that depend on a level's voxel count are indirect, so the whole build is one command
// buffer with no host round trip.

#include <metal_stdlib>
using namespace metal;

struct StQuant { uint N; float res; uint chunk; uint nchunks; };

struct StLevel {
    uint H; uint W; uint Nc;      // column grid of this level
    uint M;                       // voxel count -- written on the GPU for levels >= 1
    uint TM; uint tiles;          // int8 conv row tile, and the tile count it implies
    uint cap;                     // allocated voxel capacity
    uint overflow;                // set when M > cap; the host grows the buffers and rebuilds
};

struct StTrans { uint Hf; uint Wf; uint Hc; uint Wc; uint capC; uint pad0; uint pad1; uint pad2; };

constant constexpr int ZMAX = 2147483647;

// z = rint(logz * res). `rint` is round-half-to-even, which is torch.round; `round` would disagree on the
// exact .5 cases, and at 1/256 bins over a float32 logz those occur a handful of times per frame.
kernel void st_zq(device const float* logz [[buffer(0)]],
                  device int*         zq   [[buffer(1)]],
                  device int*         cmin [[buffer(2)]],
                  constant StQuant&   P    [[buffer(3)]],
                  uint c [[thread_position_in_grid]])
{
    if (c >= P.nchunks) return;
    const uint lo = c * P.chunk, hi = min(lo + P.chunk, P.N);
    int mn = ZMAX;
    for (uint m = lo; m < hi; ++m) {
        const int q = int(rint(logz[m] * P.res));
        zq[m] = q;
        mn = min(mn, q);
    }
    cmin[c] = mn;
}

kernel void st_zmin(device const int* cmin [[buffer(0)]],
                    device int*       zmin [[buffer(1)]],
                    constant StQuant& P    [[buffer(2)]],
                    uint g [[thread_position_in_grid]])
{
    if (g != 0) return;
    int mn = ZMAX;
    for (uint c = 0; c < P.nchunks; ++c) mn = min(mn, cmin[c]);
    zmin[0] = mn;
}

kernel void st_z0(device int*       zq   [[buffer(0)]],
                  device const int* zmin [[buffer(1)]],
                  constant StQuant& P    [[buffer(2)]],
                  uint m [[thread_position_in_grid]])
{
    if (m >= P.N) return;
    zq[m] -= zmin[0];
}

kernel void st_clear_tiles(device uint*     tiles [[buffer(0)]],
                           constant StLevel& P    [[buffer(1)]],
                           uint g [[thread_position_in_grid]])
{
    if (g >= P.tiles || P.overflow != 0u) return;
    tiles[g] = 0u;
}

// Coarse column (a, b): how many distinct z >> 1 its four fine columns hold, and how many fine voxels.
kernel void st_pool_count(device const uint* colF [[buffer(0)]],
                          device const int*  zF   [[buffer(1)]],
                          device uint*       cnt  [[buffer(2)]],
                          device uint*       fcnt [[buffer(3)]],
                          constant StTrans&  T    [[buffer(4)]],
                          uint2 g [[thread_position_in_grid]])
{
    const uint b = g.x, a = g.y;
    if (a >= T.Hc || b >= T.Wc) return;
    uint p[4], e[4];
    uint nf = 0, tot = 0;
    for (uint di = 0; di < 2; ++di)
        for (uint dj = 0; dj < 2; ++dj) {
            const uint i = 2 * a + di, j = 2 * b + dj;
            if (i < T.Hf && j < T.Wf) {
                const uint f = i * T.Wf + j;
                p[nf] = colF[f]; e[nf] = colF[f + 1]; tot += e[nf] - p[nf]; nf++;
            }
        }
    uint n = 0;
    while (true) {
        int best = ZMAX;
        for (uint s = 0; s < nf; ++s) if (p[s] < e[s]) best = min(best, zF[p[s]] >> 1);
        if (best == ZMAX) break;
        n++;
        for (uint s = 0; s < nf; ++s) while (p[s] < e[s] && (zF[p[s]] >> 1) == best) p[s]++;
    }
    const uint c = a * T.Wc + b;
    cnt[c] = n;
    fcnt[c] = tot;
}

// In-place exclusive prefix sum over a[0..n), total written to a[n]. One threadgroup of a power-of-two
// thread count: per-thread chunk sums, a pointer-jumping scan of those sums in threadgroup memory (the same
// construction pool_scan_flags validated), then each thread fills its own chunk.
struct StPrefix { uint n; };

kernel void st_prefix(device uint*       a [[buffer(0)]],
                      constant StPrefix& P [[buffer(1)]],
                      uint tid    [[thread_index_in_threadgroup]],
                      uint tcount [[threads_per_threadgroup]])
{
    threadgroup uint part[256];
    const uint per = (P.n + tcount - 1) / tcount;
    const uint lo = min(tid * per, P.n), hi = min(lo + per, P.n);
    uint s = 0u;
    for (uint i = lo; i < hi; ++i) s += a[i];
    part[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup uint scan[256];
    scan[tid] = (tid == 0) ? 0u : part[tid - 1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = 1; off < tcount; off <<= 1) {
        const uint add = (tid >= off) ? scan[tid - off] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scan[tid] += add;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint run = scan[tid];
    for (uint i = lo; i < hi; ++i) { const uint v = a[i]; a[i] = run; run += v; }
    if (tid == tcount - 1) a[P.n] = run;
}

// After the coarse prefix: the level's voxel count, its capacity check, the pool segment sentinel, and the
// indirect dispatch sizes for everything that runs over the coarse voxels.
kernel void st_finalize(device const uint* colC   [[buffer(0)]],
                        device const uint* fstart [[buffer(1)]],
                        device StLevel&    Lc     [[buffer(2)]],
                        device uint*       args   [[buffer(3)]],
                        device int*        segOff [[buffer(4)]],
                        constant StTrans&  T      [[buffer(5)]],
                        uint g [[thread_position_in_grid]])
{
    if (g != 0) return;
    const uint M = colC[Lc.Nc];
    Lc.M = M;
    Lc.tiles = (M + Lc.TM - 1) / Lc.TM;
    Lc.overflow = (M > Lc.cap) ? 1u : 0u;
    if (Lc.overflow == 0u) segOff[M] = int(fstart[T.Hc * T.Wc]);
    args[0] = max(1u, (M + 255) / 256);          args[1] = 1u; args[2] = 1u;
    args[3] = max(1u, (Lc.tiles + 255) / 256);   args[4] = 1u; args[5] = 1u;
}

// Emit the coarse level: z and column per coarse voxel, the fine -> coarse parent, and the pool's children
// grouped by parent in fine-index order (fine columns in raster order, z ascending within each) -- which is
// exactly `torch.argsort(parent, stable=True)`.
kernel void st_pool_emit(device const uint* colF   [[buffer(0)]],
                         device const int*  zF     [[buffer(1)]],
                         device const uint* colC   [[buffer(2)]],
                         device const uint* fstart [[buffer(3)]],
                         device int*        zC     [[buffer(4)]],
                         device uint*       cidC   [[buffer(5)]],
                         device int*        parent [[buffer(6)]],
                         device int*        segIdx [[buffer(7)]],
                         device int*        segOff [[buffer(8)]],
                         constant StTrans&  T      [[buffer(9)]],
                         uint2 g [[thread_position_in_grid]])
{
    const uint b = g.x, a = g.y;
    if (a >= T.Hc || b >= T.Wc) return;
    uint p[4], e[4];
    uint nf = 0;
    for (uint di = 0; di < 2; ++di)
        for (uint dj = 0; dj < 2; ++dj) {
            const uint i = 2 * a + di, j = 2 * b + dj;
            if (i < T.Hf && j < T.Wf) {
                const uint f = i * T.Wf + j;
                p[nf] = colF[f]; e[nf] = colF[f + 1]; nf++;
            }
        }
    const uint c = a * T.Wc + b;
    uint o = colC[c];
    uint t = fstart[c];
    while (true) {
        int best = ZMAX;
        for (uint s = 0; s < nf; ++s) if (p[s] < e[s]) best = min(best, zF[p[s]] >> 1);
        if (best == ZMAX) break;
        const bool ok = o < T.capC;
        if (ok) { zC[o] = best; cidC[o] = c; segOff[o] = int(t); }
        for (uint s = 0; s < nf; ++s)
            while (p[s] < e[s] && (zF[p[s]] >> 1) == best) {
                segIdx[t++] = int(p[s]);
                parent[p[s]] = int(o);
                p[s]++;
            }
        o++;
    }
}

// Neighbour map for one level, plus the per-tile tap bitmask. Tap order is make_conv_kernel_delta's:
// v = (di+1)*9 + (dj+1)*3 + (dz+1). Levels >= 1 are de-duplicated and level 0 has one voxel per column, so
// each (column, z) holds at most one voxel and a single forward scan finds z-1, z, z+1 in turn.
kernel void st_rulebook(device const uint* col   [[buffer(0)]],
                        device const int*  z     [[buffer(1)]],
                        device const uint* cid   [[buffer(2)]],
                        device int*        nmapT [[buffer(3)]],
                        device uint*       tiles [[buffer(4)]],
                        constant StLevel&  P     [[buffer(5)]],
                        uint o [[thread_position_in_grid]])
{
    if (o >= P.M || P.overflow != 0u) return;
    const uint c = cid[o];
    const int i = int(c / P.W), j = int(c % P.W);
    const int zz = z[o];
    uint mask = 0u;
    for (int di = -1; di <= 1; ++di)
        for (int dj = -1; dj <= 1; ++dj) {
            const uint vb = uint((di + 1) * 9 + (dj + 1) * 3);
            int h0 = -1, h1 = -1, h2 = -1;
            const int ni = i + di, nj = j + dj;
            if (ni >= 0 && ni < int(P.H) && nj >= 0 && nj < int(P.W)) {
                const uint nc = uint(ni) * P.W + uint(nj);
                uint q = col[nc];
                const uint qe = col[nc + 1];
                while (q < qe && z[q] < zz - 1) q++;
                if (q < qe && z[q] == zz - 1) { h0 = int(q); q++; }
                if (q < qe && z[q] == zz)     { h1 = int(q); q++; }
                if (q < qe && z[q] == zz + 1) { h2 = int(q); }
            }
            nmapT[vb * P.M + o]       = h0;
            nmapT[(vb + 1) * P.M + o] = h1;
            nmapT[(vb + 2) * P.M + o] = h2;
            if (h0 >= 0) mask |= 1u << vb;
            if (h1 >= 0) mask |= 1u << (vb + 1);
            if (h2 >= 0) mask |= 1u << (vb + 2);
        }
    atomic_fetch_or_explicit((device atomic_uint*)(tiles + o / P.TM), mask, memory_order_relaxed);
}
