import Foundation
import Metal

/// Trimmed least-squares scale and shift on the GPU (`Shaders/MetricFit.metal`): encoded into the caller's compute
/// encoder, no host round trip, result left in `state` where the refiner's residual kernel reads it.
final class MetricFit {
    struct State {
        var s: Float = 1; var t: Float = 0; var lo: Float = 0; var hi: Float = 0
        var count: UInt32 = 0; var k: UInt32 = 0; var ok: UInt32 = 0; var iter: UInt32 = 0
        var n: UInt32 = 0; var chunk: UInt32 = 4096; var nchunks: UInt32 = 0; var bins: UInt32 = 4096
    }
    struct Geometry { var H: UInt32; var W: UInt32; var dh: UInt32; var dw: UInt32 }

    let state: MTLBuffer
    private let z, target, mask, weight, u, sums, hist: MTLBuffer
    private var p: [String: MTLComputePipelineState] = [:]

    init?(device: MTLDevice, library: MTLLibrary, capacity n: Int) {
        for name in ["fit_prep_prompt", "fit_prep_final", "fit_sums", "fit_solve", "fit_resid_hist", "fit_hist2",
                     "fit_kth", "fit_hist_clear"] {
            guard let f = library.makeFunction(name: name), let s = try? device.makeComputePipelineState(function: f) else { return nil }
            p[name] = s
        }
        let o = MTLResourceOptions.storageModeShared
        var st = State()
        state = device.makeBuffer(bytes: &st, length: MemoryLayout<State>.stride, options: o)!
        z = device.makeBuffer(length: n * 4, options: o)!; target = device.makeBuffer(length: n * 4, options: o)!
        mask = device.makeBuffer(length: n, options: o)!; weight = device.makeBuffer(length: n, options: o)!
        u = device.makeBuffer(length: n * 4, options: o)!
        sums = device.makeBuffer(length: ((n + 4095) / 4096) * 5 * 4, options: o)!
        hist = device.makeBuffer(length: (4096 + 1) * 4, options: o)!
    }

    var s: Float { state.contents().load(as: State.self).s }
    var t: Float { state.contents().load(as: State.self).t }

    /// State is reset on the host, so encode before committing.
    private func reset(_ n: Int) {
        var st = State()
        st.n = UInt32(n); st.nchunks = UInt32((n + Int(st.chunk) - 1) / Int(st.chunk))
        state.contents().storeBytes(of: st, as: State.self)
    }

    /// The refiner's per-step fit: exp(logz) at the point map against the nearest confident LiDAR pixel.
    func encodePrompt(_ e: MTLComputeCommandEncoder, logz: MTLBuffer, lidar: MTLBuffer, conf: MTLBuffer,
                      H: Int, W: Int, dh: Int, dw: Int) {
        let n = H * W
        reset(n)
        e.setComputePipelineState(p["fit_prep_prompt"]!)
        e.setBuffer(logz, offset: 0, index: 0); e.setBuffer(lidar, offset: 0, index: 1); e.setBuffer(conf, offset: 0, index: 2)
        e.setBuffer(z, offset: 0, index: 3); e.setBuffer(target, offset: 0, index: 4)
        e.setBuffer(mask, offset: 0, index: 5); e.setBuffer(weight, offset: 0, index: 6)
        var g = Geometry(H: UInt32(H), W: UInt32(W), dh: UInt32(dh), dw: UInt32(dw))
        e.setBytes(&g, length: MemoryLayout<Geometry>.stride, index: 7)
        e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encodeSolve(e, n)
    }

    /// The final metric fit, at the LiDAR resolution: area-mean model depth per sensor pixel, confident and unmasked.
    func encodeFinal(_ e: MTLComputeCommandEncoder, logz: MTLBuffer, maskLogit: MTLBuffer, lidar: MTLBuffer,
                     conf: MTLBuffer, H: Int, W: Int, dh: Int, dw: Int) {
        let n = dh * dw
        reset(n)
        e.setComputePipelineState(p["fit_prep_final"]!)
        e.setBuffer(logz, offset: 0, index: 0); e.setBuffer(maskLogit, offset: 0, index: 1)
        e.setBuffer(lidar, offset: 0, index: 2); e.setBuffer(conf, offset: 0, index: 3)
        e.setBuffer(z, offset: 0, index: 4); e.setBuffer(target, offset: 0, index: 5)
        e.setBuffer(mask, offset: 0, index: 6); e.setBuffer(weight, offset: 0, index: 7)
        var g = Geometry(H: UInt32(H), W: UInt32(W), dh: UInt32(dh), dw: UInt32(dw))
        e.setBytes(&g, length: MemoryLayout<Geometry>.stride, index: 8)
        e.dispatchThreads(MTLSize(width: dw, height: dh, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        encodeSolve(e, n)
    }

    private func encodeSolve(_ e: MTLComputeCommandEncoder, _ n: Int) {
        let one = MTLSize(width: 1, height: 1, depth: 1)
        let chunks = MTLSize(width: (n + 4095) / 4096, height: 1, depth: 1), lanes = MTLSize(width: 32, height: 1, depth: 1)
        let elems = MTLSize(width: n, height: 1, depth: 1), tg = MTLSize(width: 256, height: 1, depth: 1)
        func clear() {
            e.setComputePipelineState(p["fit_hist_clear"]!)
            e.setBuffer(hist, offset: 0, index: 0); e.setBuffer(state, offset: 0, index: 1)
            e.dispatchThreads(MTLSize(width: 4097, height: 1, depth: 1), threadsPerThreadgroup: tg)
        }
        func solve(onMask: Bool) {
            var m: UInt32 = onMask ? 1 : 0
            e.setComputePipelineState(p["fit_sums"]!)
            e.setBuffer(z, offset: 0, index: 0); e.setBuffer(target, offset: 0, index: 1); e.setBuffer(u, offset: 0, index: 2)
            e.setBuffer(mask, offset: 0, index: 3); e.setBuffer(sums, offset: 0, index: 4); e.setBuffer(state, offset: 0, index: 5)
            e.setBytes(&m, length: 4, index: 6)
            e.dispatchThreadgroups(chunks, threadsPerThreadgroup: lanes)
            e.setComputePipelineState(p["fit_solve"]!)
            e.setBuffer(sums, offset: 0, index: 0); e.setBuffer(state, offset: 0, index: 1)
            e.dispatchThreads(one, threadsPerThreadgroup: one)
        }
        func kth(level1: Bool) {
            var lv: UInt32 = level1 ? 1 : 0
            e.setComputePipelineState(p["fit_kth"]!)
            e.setBuffer(hist, offset: 0, index: 0); e.setBuffer(state, offset: 0, index: 1)
            e.setBytes(&lv, length: 4, index: 2)
            e.dispatchThreads(one, threadsPerThreadgroup: one)
        }
        clear()
        solve(onMask: true)
        for _ in 0..<4 {
            e.setComputePipelineState(p["fit_resid_hist"]!)
            e.setBuffer(z, offset: 0, index: 0); e.setBuffer(target, offset: 0, index: 1); e.setBuffer(mask, offset: 0, index: 2)
            e.setBuffer(u, offset: 0, index: 3); e.setBuffer(hist, offset: 0, index: 4); e.setBuffer(state, offset: 0, index: 5)
            e.dispatchThreads(elems, threadsPerThreadgroup: tg)
            kth(level1: true); clear()
            e.setComputePipelineState(p["fit_hist2"]!)
            e.setBuffer(u, offset: 0, index: 0); e.setBuffer(hist, offset: 0, index: 1); e.setBuffer(state, offset: 0, index: 2)
            e.dispatchThreads(elems, threadsPerThreadgroup: tg)
            kth(level1: false); clear()
            solve(onMask: false)
        }
    }
}
