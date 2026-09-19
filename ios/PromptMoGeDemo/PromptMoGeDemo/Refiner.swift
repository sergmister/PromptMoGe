import Foundation
import Metal

/// The sparse 3-D refiner of Model A / Model B on Metal, int8 end to end, on persistent buffers.
///
/// One refinement step lifts the point map to voxels in (x/z, y/z, log z), runs a sparse U-Net over them and adds
/// its output to log z. Per step:
///  * the voxel structure (columns, pooling maps, neighbour maps, per-tile tap masks) is rebuilt on the GPU in one
///    command buffer (`VoxelStructure`);
///  * the LiDAR is re-fitted to the current depth and its gated residual fed in as two extra channels, all on the GPU;
///  * the graph is one compute encoder: activations, skips and int8 staging are allocated once, the down block
///    writes straight into the skip buffer, so there are no blits.
/// The encoder conditioning (quantise, int8 `encoder_fuse`, + UV bias) runs once per frame at the 30x40 token grid;
/// a step only gathers rows from it by column id.
///
/// The checkpoint is quantisation-aware: every convolution and linear layer except the input / prompt / output
/// projections is quantised on its input with a learned clip. The kernels reproduce exactly that arithmetic:
/// scale = clip / 127 per tensor, int8 weights per output channel, int32 accumulation.
final class Refiner {
    struct Block: Decodable { let offset: Int; let count: Int; let dtype: String }
    struct Manifest: Decodable {
        let H: Int, W: Int, R: Int, C: Int, levels: Int, channels: [Int]
        let enc_channels: Int, enc_out: Int, depth_resolution: Float
        let weights: [String: Block]
    }
    /// An int8 kernel and the tile it was instantiated for: TM rows x TN output channels, TK input channels per
    /// GEMM, SG simdgroups per threadgroup.
    struct Tile { let name: String; let tm: Int; let tn: Int; let tk: Int; let sg: Int }

    private struct PromptParams { var H: UInt32; var W: UInt32; var dh: UInt32; var dw: UInt32; var s: Float = 0; var t: Float = 0; var useFit: UInt32 = 1; var pad: UInt32 = 0 }
    private struct InputParams { var N: UInt32; var C: UInt32; var p0: UInt32 = 0; var p1: UInt32 = 0 }
    private struct PoolParams { var Min: Int32; var Mout: Int32; var C: Int32; var invScale: Float }
    private struct QuantParams { var n: UInt32; var invScale: Float; var p0: UInt32 = 0; var p1: UInt32 = 0 }
    private struct GemmParams { var M: Int32; var N: Int32; var K: Int32; var aScale: Float; var oInvScale: Float; var p0: Int32 = 0; var p1: Int32 = 0; var p2: Int32 = 0 }
    private struct GatherParams { var M: UInt32; var C: UInt32; var p0: UInt32 = 0; var p1: UInt32 = 0 }
    private struct PlanarParams { var T: UInt32; var K: UInt32; var invScale: Float; var p0: UInt32 = 0 }
    private struct CatParams { var M: UInt32; var C1: UInt32; var C2: UInt32; var invScale: Float }
    private struct RowParams { var M: Int32; var C: Int32; var eps: Float; var invScale: Float }
    private struct ConvParams { var M: Int32; var Ci: Int32; var Co: Int32; var V: Int32 = 27; var a: Int32 = 0; var b: Int32 = 0; var c: Int32 = 0; var d: Int32 = 0 }

    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary
    let structure: VoxelStructure
    let fit: MetricFit
    let L: Int, ch: [Int], H: Int, W: Int, N: Int, T: Int, encChannels: Int, fused: Int
    private var pipes: [String: MTLComputePipelineState] = [:]
    private var weights: [String: MTLBuffer] = [:]
    private var clip: [String: Float] = [:]
    private let conv: [Tile]

    /// Written by the pipeline every frame: point-map x/z, y/z planes, the sensor planes, the token-grid feature.
    let xy: MTLBuffer, lidar: MTLBuffer, conf: MTLBuffer, encoderFeature: MTLBuffer
    private let prompt: MTLBuffer, delta: MTLBuffer, encQ: MTLBuffer, encFused: MTLBuffer, unused: MTLBuffer
    private var x: [MTLBuffer] = [], skip: [MTLBuffer] = [], y: [MTLBuffer] = []
    private var poolQ: [MTLBuffer] = [], upLin: [MTLBuffer] = []
    private var q1: MTLBuffer!, q2: MTLBuffer!, encG: MTLBuffer!, catQ: MTLBuffer!, fuseQ: MTLBuffer!
    private var allocated: [Int] = []
    private(set) var dh = 0, dw = 0
    private var M: [Int] = []

    static let maxSensorPixels = 320 * 256

    /// `dir` holds `refiner.json` and `refiner.bin` (`python -m promptmoge.export.refiner`).
    init?(dir: URL) {
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue(), let lib = d.makeDefaultLibrary(),
              let md = try? Data(contentsOf: dir.appendingPathComponent("refiner.json")),
              let m = try? JSONDecoder().decode(Manifest.self, from: md),
              let wd = try? Data(contentsOf: dir.appendingPathComponent("refiner.bin"), options: .alwaysMapped) else { return nil }
        device = d; queue = q; library = lib
        L = m.levels; ch = m.channels; H = m.H; W = m.W; N = m.H * m.W; T = m.R * m.C
        encChannels = m.enc_channels; fused = m.enc_out
        // One tile per channel width (see SparseConvInt8.metal). A 256-channel level that is not the last, smallest
        // one keeps TN = 128: with more rows, two half-width GEMMs per tile beat one full-width one.
        conv = (0..<m.levels).map { k in
            switch m.channels[k] {
            case 32: return Tile(name: "m32_n32_k32_sg1", tm: 32, tn: 32, tk: 32, sg: 1)
            case 64: return Tile(name: "m32_n64_k64_sg1", tm: 32, tn: 64, tk: 64, sg: 1)
            case 256 where k == m.levels - 1: return Tile(name: "m32_n256_k128_sg4", tm: 32, tn: 256, tk: 128, sg: 4)
            default: return Tile(name: "m32_n128_k128_sg2", tm: 32, tn: 128, tk: 128, sg: 2)
            }
        }
        guard let st = VoxelStructure(device: d, queue: q, library: lib, H: m.H, W: m.W, levels: m.levels,
                                      tileRows: conv.map { $0.tm }, resolution: m.depth_resolution),
              let ft = MetricFit(device: d, library: lib, capacity: max(m.H * m.W, Self.maxSensorPixels)),
              // the last refiner level must be the token grid: the conditioning is gathered by column id
              st.dims(m.levels - 1) == (m.R, m.C) else { return nil }
        structure = st; fit = ft

        var names = ["rf_prompt_feats", "rf_input", "pool_mean_q8", "quant_h_i8", "quant_planar_f_i8", "add_planar_f_to_h",
                     "gather_rows_h", "concat_q8", "up_add_h", "rf_out", "rf_apply",
                     "dg8_n64_k32", "dg8_n128_k64", "dg8_n256_k128", "dg8_n256_k256", "dg8_n128_k256", "dg8_n64_k128",
                     "dg8_n32_k64", "dg8_n256_k1024", "dg8q_n256_k512"]
        for k in 0..<m.levels { names += ["spconv_i8q_" + conv[k].name, "spconv_i8r_" + conv[k].name, "ln_silu_i8_r\(4096 / m.channels[k])"] }
        for n in Set(names) {
            guard let f = lib.makeFunction(name: n), let s = try? d.makeComputePipelineState(function: f) else { return nil }
            pipes[n] = s
        }
        let shared = MTLResourceOptions.storageModeShared
        for (name, b) in m.weights {
            let bytes = b.count * (b.dtype == "float16" ? 2 : (b.dtype == "int8" ? 1 : 4))
            guard let buf = wd.withUnsafeBytes({ d.makeBuffer(bytes: $0.baseAddress! + b.offset, length: max(bytes, 4), options: shared) }) else { return nil }
            weights[name] = buf
            if name.hasSuffix("clip") { clip[name] = buf.contents().load(as: Float.self) }
        }
        guard weights["uv_bias"] != nil else { return nil }
        xy = d.makeBuffer(length: 2 * N * 4, options: shared)!
        prompt = d.makeBuffer(length: 2 * N * 4, options: shared)!
        delta = d.makeBuffer(length: N * 4, options: shared)!
        lidar = d.makeBuffer(length: Self.maxSensorPixels * 4, options: shared)!
        conf = d.makeBuffer(length: Self.maxSensorPixels, options: shared)!
        encoderFeature = d.makeBuffer(length: m.enc_channels * T * 4, options: shared)!
        encQ = d.makeBuffer(length: m.enc_channels * T, options: shared)!
        encFused = d.makeBuffer(length: m.enc_out * T * 2, options: shared)!
        unused = d.makeBuffer(length: 16, options: shared)!
    }

    private func ensureActivations() {
        let caps = structure.caps
        guard caps != allocated else { return }
        allocated = caps
        let o = MTLResourceOptions.storageModeShared
        func f16(_ n: Int) -> MTLBuffer { device.makeBuffer(length: max(n, 8) * 2, options: o)! }
        func i8(_ n: Int) -> MTLBuffer { device.makeBuffer(length: max(n, 16) + 16, options: o)! }
        x = (0..<L).map { f16(caps[$0] * ch[$0]) }
        skip = (0..<L).map { f16(caps[$0] * ch[$0]) }
        y = (0..<L).map { f16(caps[$0] * ch[$0]) }
        poolQ = (0..<(L - 1)).map { i8(caps[$0 + 1] * ch[$0]) }
        upLin = (0..<(L - 1)).map { f16(caps[$0 + 1] * ch[$0]) }
        let widest = (0..<L).map { caps[$0] * ch[$0] }.max()!
        q1 = i8(widest); q2 = i8(widest)
        encG = f16(caps[L - 1] * fused); catQ = i8(caps[L - 1] * (ch[L - 1] + fused)); fuseQ = i8(caps[L - 1] * fused)
    }

    /// Sensor dimensions of the planes the caller has written into `lidar` and `conf`.
    func setSensor(dw: Int, dh: Int) { self.dw = dw; self.dh = dh }

    // ------------------------------------------------------------------------------------------------ encode helpers
    private func pso(_ e: MTLComputeCommandEncoder, _ n: String) { e.setComputePipelineState(pipes[n]!) }
    private func rows(_ e: MTLComputeCommandEncoder, _ n: Int, _ w: Int = 256) {
        e.dispatchThreads(MTLSize(width: max(n, 1), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
    }
    private func bytes<P: BitwiseCopyable>(_ e: MTLComputeCommandEncoder, _ p: P, _ i: Int) {
        withUnsafeBytes(of: p) { e.setBytes($0.baseAddress!, length: MemoryLayout<P>.stride, index: i) }
    }
    private func w(_ n: String) -> MTLBuffer { weights[n]! }
    private func scale(_ tag: String) -> Float { clip[tag]! / 127 }
    private func inv(_ tag: String) -> Float { 127 / clip[tag]! }

    private func gemmTile(_ N: Int, _ K: Int, requantise: Bool) -> Tile {
        if requantise { return Tile(name: "dg8q_n256_k512", tm: 32, tn: 256, tk: 128, sg: 4) }
        switch (N, K) {
        case (64, 32):    return Tile(name: "dg8_n64_k32", tm: 64, tn: 64, tk: 32, sg: 2)
        case (128, 64):   return Tile(name: "dg8_n128_k64", tm: 32, tn: 128, tk: 64, sg: 2)
        case (256, 128):  return Tile(name: "dg8_n256_k128", tm: 32, tn: 256, tk: 128, sg: 4)
        case (256, 256):  return Tile(name: "dg8_n256_k256", tm: 32, tn: 256, tk: 128, sg: 4)
        case (128, 256):  return Tile(name: "dg8_n128_k256", tm: 32, tn: 128, tk: 128, sg: 2)
        case (64, 128):   return Tile(name: "dg8_n64_k128", tm: 64, tn: 64, tk: 128, sg: 2)
        case (32, 64):    return Tile(name: "dg8_n32_k64", tm: 64, tn: 32, tk: 64, sg: 1)
        case (256, 1024): return Tile(name: "dg8_n256_k1024", tm: 32, tn: 256, tk: 128, sg: 4)
        default: fatalError("no int8 GEMM for N=\(N) K=\(K)")
        }
    }

    /// Int8 linear layer `tag` on `r` rows: fp16 out, or (fuse_proj.0) SiLU + requantise to int8 with `oInv`.
    private func gemm(_ e: MTLComputeCommandEncoder, _ tag: String, _ A: MTLBuffer, rows r: Int, K: Int, N n: Int,
                      out16: MTLBuffer? = nil, out8: MTLBuffer? = nil, oInv: Float = 0) {
        let c = gemmTile(n, K, requantise: out8 != nil)
        pso(e, c.name)
        e.setBuffer(A, offset: 0, index: 0); e.setBuffer(w("\(tag).w8"), offset: 0, index: 1)
        e.setBuffer(w("\(tag).ws"), offset: 0, index: 2); e.setBuffer(w("\(tag).b"), offset: 0, index: 3)
        e.setBuffer(out16 ?? unused, offset: 0, index: 4); e.setBuffer(out8 ?? unused, offset: 0, index: 5)
        bytes(e, GemmParams(M: Int32(r), N: Int32(n), K: Int32(K), aScale: scale("\(tag).clip"), oInvScale: oInv), 6)
        e.setThreadgroupMemoryLength(c.tm * c.tk, index: 0)
        e.dispatchThreadgroups(MTLSize(width: (n + c.tn - 1) / c.tn, height: (r + c.tm - 1) / c.tm, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: c.sg * 32, height: 1, depth: 1))
    }

    /// LayerNorm -> SiLU -> conv1 -> SiLU -> conv2 -> + input, as three dispatches.
    private func resblock(_ e: MTLComputeCommandEncoder, _ k: Int, _ tag: String, _ input: MTLBuffer, _ output: MTLBuffer) {
        let C = ch[k], R = 4096 / C, c = conv[k], Mk = M[k]
        pso(e, "ln_silu_i8_r\(R)")
        e.setBuffer(input, offset: 0, index: 0); e.setBuffer(q1, offset: 0, index: 1)
        e.setBuffer(w("\(tag).n1w"), offset: 0, index: 2); e.setBuffer(w("\(tag).n1b"), offset: 0, index: 3)
        bytes(e, RowParams(M: Int32(Mk), C: Int32(C), eps: 1e-6, invScale: inv("\(tag).c1clip")), 4)
        e.setThreadgroupMemoryLength(R * C * 2, index: 0); e.setThreadgroupMemoryLength(R * C, index: 1)
        e.dispatchThreadgroups(MTLSize(width: (Mk + R - 1) / R, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        let params = ConvParams(M: Int32(Mk), Ci: Int32(C), Co: Int32(C))
        let grid = MTLSize(width: (C + c.tn - 1) / c.tn, height: (Mk + c.tm - 1) / c.tm, depth: 1)
        let threads = MTLSize(width: c.sg * 32, height: 1, depth: 1)
        for (i, src, dst) in [(1, q1!, q2!), (2, q2!, output)] {
            pso(e, (i == 1 ? "spconv_i8q_" : "spconv_i8r_") + c.name)
            e.setBuffer(src, offset: 0, index: 0); e.setBuffer(structure.neighbours[k], offset: 0, index: 1)
            e.setBuffer(w("\(tag).c\(i)w8"), offset: 0, index: 2); e.setBuffer(w("\(tag).c\(i)b"), offset: 0, index: 3)
            e.setBuffer(dst, offset: 0, index: 4); bytes(e, params, 5)
            e.setBuffer(structure.tiles[k], offset: 0, index: 6); e.setBuffer(w("\(tag).c\(i)ws"), offset: 0, index: 7)
            bytes(e, scale("\(tag).c\(i)clip"), 8)
            if i == 1 { bytes(e, inv("\(tag).c2clip"), 9) } else { e.setBuffer(input, offset: 0, index: 9) }
            e.setThreadgroupMemoryLength(c.tm * c.tk, index: 0)
            e.dispatchThreadgroups(grid, threadsPerThreadgroup: threads)
        }
    }

    // ------------------------------------------------------------------------------------------------ frame
    /// Once per frame, after `encoderFeature` is written: quantise it, int8 `encoder_fuse`, + UV bias.
    func setFrame() {
        let cb = queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
        let grid = MTLSize(width: 64, height: 16, depth: 1)
        pso(e, "quant_planar_f_i8")
        e.setBuffer(encoderFeature, offset: 0, index: 0); e.setBuffer(encQ, offset: 0, index: 1)
        bytes(e, PlanarParams(T: UInt32(T), K: UInt32(encChannels), invScale: inv("encoder_fuse.clip")), 2)
        e.dispatchThreads(MTLSize(width: encChannels, height: T, depth: 1), threadsPerThreadgroup: grid)
        gemm(e, "encoder_fuse", encQ, rows: T, K: encChannels, N: fused, out16: encFused)
        pso(e, "add_planar_f_to_h")
        e.setBuffer(encFused, offset: 0, index: 0); e.setBuffer(w("uv_bias"), offset: 0, index: 1)
        bytes(e, PlanarParams(T: UInt32(T), K: UInt32(fused), invScale: 0), 2)
        e.dispatchThreads(MTLSize(width: fused, height: T, depth: 1), threadsPerThreadgroup: grid)
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    }

    // ------------------------------------------------------------------------------------------------ step
    private func encodeGraph(_ e: MTLComputeCommandEncoder) {
        // the sensor in the model's affine frame: refit (s, t) on the current depth, then the gated log residual
        fit.encodePrompt(e, logz: structure.logz, lidar: lidar, conf: conf, H: H, W: W, dh: dh, dw: dw)
        pso(e, "rf_prompt_feats")
        e.setBuffer(structure.logz, offset: 0, index: 0); e.setBuffer(lidar, offset: 0, index: 1)
        e.setBuffer(conf, offset: 0, index: 2); e.setBuffer(prompt, offset: 0, index: 3)
        bytes(e, PromptParams(H: UInt32(H), W: UInt32(W), dh: UInt32(dh), dw: UInt32(dw)), 4)
        e.setBuffer(fit.state, offset: 0, index: 5)
        rows(e, N)

        pso(e, "rf_input")
        e.setBuffer(xy, offset: 0, index: 0); e.setBuffer(structure.logz, offset: 0, index: 1); e.setBuffer(prompt, offset: 0, index: 2)
        e.setBuffer(w("input_proj.w"), offset: 0, index: 3); e.setBuffer(w("input_proj.b"), offset: 0, index: 4)
        e.setBuffer(w("prompt_proj.w"), offset: 0, index: 5); e.setBuffer(w("prompt_proj.b"), offset: 0, index: 6)
        e.setBuffer(x[0], offset: 0, index: 7)
        bytes(e, InputParams(N: UInt32(N), C: UInt32(ch[0])), 8)
        rows(e, N)

        for k in 0..<L {
            resblock(e, k, "down\(k)", x[k], skip[k])
            guard k < L - 1 else { continue }
            pso(e, "pool_mean_q8")
            e.setBuffer(skip[k], offset: 0, index: 0); e.setBuffer(structure.children[k], offset: 0, index: 1)
            e.setBuffer(structure.childOffsets[k], offset: 0, index: 2); e.setBuffer(poolQ[k], offset: 0, index: 3)
            bytes(e, PoolParams(Min: Int32(M[k]), Mout: Int32(M[k + 1]), C: Int32(ch[k]), invScale: inv("pool\(k).clip")), 4)
            rows(e, M[k + 1], 64)
            gemm(e, "pool\(k)", poolQ[k], rows: M[k + 1], K: ch[k], N: ch[k + 1], out16: x[k + 1])
        }

        // bottleneck: fuse the encoder conditioning, gathered by column id
        let ML = M[L - 1], CL = ch[L - 1]
        pso(e, "gather_rows_h")
        e.setBuffer(encFused, offset: 0, index: 0); e.setBuffer(structure.column[L - 1], offset: 0, index: 1)
        e.setBuffer(encG, offset: 0, index: 2)
        bytes(e, GatherParams(M: UInt32(ML), C: UInt32(fused)), 3)
        rows(e, ML)
        pso(e, "concat_q8")
        e.setBuffer(skip[L - 1], offset: 0, index: 0); e.setBuffer(encG, offset: 0, index: 1); e.setBuffer(catQ, offset: 0, index: 2)
        bytes(e, CatParams(M: UInt32(ML), C1: UInt32(CL), C2: UInt32(fused), invScale: inv("fuse0.clip")), 3)
        rows(e, ML)
        gemm(e, "fuse0", catQ, rows: ML, K: CL + fused, N: fused, out8: fuseQ, oInv: inv("fuse2.clip"))
        gemm(e, "fuse2", fuseQ, rows: ML, K: fused, N: fused, out16: y[L - 1])
        resblock(e, L - 1, "bott", y[L - 1], x[L - 1])

        for i in 0..<(L - 1) {
            let t = L - 2 - i
            pso(e, "quant_h_i8")
            e.setBuffer(x[t + 1], offset: 0, index: 0); e.setBuffer(q1, offset: 0, index: 1)
            bytes(e, QuantParams(n: UInt32(M[t + 1] * ch[t + 1]), invScale: inv("up\(i).clip")), 2)
            rows(e, M[t + 1] * ch[t + 1])
            gemm(e, "up\(i)", q1, rows: M[t + 1], K: ch[t + 1], N: ch[t], out16: upLin[t])
            pso(e, "up_add_h")
            e.setBuffer(upLin[t], offset: 0, index: 0); e.setBuffer(structure.parent[t], offset: 0, index: 1)
            e.setBuffer(skip[t], offset: 0, index: 2); e.setBuffer(y[t], offset: 0, index: 3)
            bytes(e, InputParams(N: UInt32(M[t]), C: UInt32(ch[t])), 4)
            rows(e, M[t])
            resblock(e, t, "dec\(i)", y[t], x[t])
        }

        pso(e, "rf_out")
        e.setBuffer(x[0], offset: 0, index: 0); e.setBuffer(w("out_proj.w"), offset: 0, index: 1)
        e.setBuffer(w("out_proj.b"), offset: 0, index: 2); e.setBuffer(delta, offset: 0, index: 3)
        bytes(e, InputParams(N: UInt32(N), C: UInt32(ch[0])), 4)
        rows(e, N)
        pso(e, "rf_apply")
        e.setBuffer(structure.logz, offset: 0, index: 0); e.setBuffer(delta, offset: 0, index: 1)
        bytes(e, InputParams(N: UInt32(N), C: 1), 2)
        rows(e, N)
    }

    /// One refinement step, in place on `structure.logz`.
    @discardableResult
    func step() -> Bool {
        guard structure.build() else { return false }
        M = structure.M
        ensureActivations()
        guard let cb = queue.makeCommandBuffer(), let e = cb.makeComputeCommandEncoder() else { return false }
        encodeGraph(e)
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        return cb.status == .completed
    }
}
