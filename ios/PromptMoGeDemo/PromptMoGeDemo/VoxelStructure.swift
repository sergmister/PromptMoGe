import Foundation
import Metal

/// The whole refiner structure on the GPU, in one command buffer, from persistent buffers.
///
/// `Structure.metal` explains the construction: every level is a CSR over (i, j) columns with z ascending,
/// which is exactly the lexicographic order the reference uses, so pooling is a per-column merge and the
/// neighbour search a short in-column scan. Nothing here allocates per call: buffers are sized once for the
/// point map and grown only if a level ever exceeds its capacity, which the GPU reports rather than
/// overrunning.
///
/// Level-dependent dispatch sizes are indirect. The single host round trip is the final
/// `waitUntilCompleted`, which the caller needs anyway to learn the voxel counts it encodes the graph with.
final class VoxelStructure {
    struct StQuant { var N: UInt32; var res: Float; var chunk: UInt32; var nchunks: UInt32 }
    struct StLevel {
        var H: UInt32; var W: UInt32; var Nc: UInt32; var M: UInt32
        var TM: UInt32; var tiles: UInt32; var cap: UInt32; var overflow: UInt32
    }
    struct StTrans {
        var Hf: UInt32; var Wf: UInt32; var Hc: UInt32; var Wc: UInt32
        var capC: UInt32; var pad0: UInt32 = 0; var pad1: UInt32 = 0; var pad2: UInt32 = 0
    }
    struct StPrefix { var n: UInt32 }

    let dev: MTLDevice
    let queue: MTLCommandQueue
    let H: Int, W: Int, L: Int
    let res: Float
    private let tm: [Int]
    private var p: [String: MTLComputePipelineState] = [:]

    // level 0 input
    let logz: MTLBuffer                       // [N] float32, written by the caller
    private let chunk = 64
    private let cmin: MTLBuffer, zmin: MTLBuffer, quant: MTLBuffer

    // per level
    private(set) var colStart: [MTLBuffer] = []     // [Nc+1]
    private(set) var z: [MTLBuffer] = []            // [cap]
    private(set) var column: [MTLBuffer] = []          // [cap]
    private(set) var neighbours: [MTLBuffer] = []        // [27*cap]
    private(set) var tiles: [MTLBuffer] = []        // [cap/TM + 1]
    private(set) var level: [MTLBuffer] = []        // StLevel
    private(set) var args: [MTLBuffer] = []         // indirect args
    // per transition k -> k+1
    private(set) var parent: [MTLBuffer] = []       // [cap_k]
    private(set) var children: [MTLBuffer] = []       // [cap_k]
    private(set) var childOffsets: [MTLBuffer] = []       // [cap_{k+1}+1]
    private var cnt: [MTLBuffer] = [], fcnt: [MTLBuffer] = [], trans: [MTLBuffer] = []
    private(set) var caps: [Int] = []

    private(set) var M: [Int] = []

    func dims(_ k: Int) -> (Int, Int) {
        var h = H, w = W
        for _ in 0..<k { h = (h + 1) / 2; w = (w + 1) / 2 }
        return (h, w)
    }

    /// `tileRows`: the conv row tile per level (the int8 kernels' TM), so the rulebook can write the tap mask
    /// those kernels read.
    init?(device: MTLDevice, queue: MTLCommandQueue, library lib: MTLLibrary, H: Int, W: Int, levels: Int,
          tileRows: [Int], resolution: Float = 256, capFractions: [Double] = [1, 0.5, 0.2, 0.07, 0.025, 0.01]) {
        dev = device; self.queue = queue; self.H = H; self.W = W; L = levels; res = resolution
        tm = tileRows
        for n in ["st_zq", "st_zmin", "st_z0", "st_clear_tiles", "st_pool_count", "st_prefix",
                  "st_finalize", "st_pool_emit", "st_rulebook"] {
            guard let f = lib.makeFunction(name: n), let s = try? device.makeComputePipelineState(function: f)
            else { return nil }
            p[n] = s
        }
        let N = H * W
        logz = device.makeBuffer(length: N * 4, options: .storageModeShared)!
        let nch = (N + chunk - 1) / chunk
        cmin = device.makeBuffer(length: nch * 4, options: .storageModeShared)!
        zmin = device.makeBuffer(length: 4, options: .storageModeShared)!
        var q = StQuant(N: UInt32(N), res: resolution, chunk: UInt32(chunk), nchunks: UInt32(nch))
        quant = device.makeBuffer(bytes: &q, length: MemoryLayout<StQuant>.stride, options: .storageModeShared)!
        for k in 0..<L {
            caps.append(k == 0 ? N : max(4096, Int(Double(N) * capFractions[min(k, capFractions.count - 1)])))
        }
        allocate()
    }

    private func mk(_ bytes: Int) -> MTLBuffer {
        dev.makeBuffer(length: max(bytes, 16), options: .storageModeShared)!
    }

    private func allocate() {
        colStart = []; z = []; column = []; neighbours = []; tiles = []; level = []; args = []
        parent = []; children = []; childOffsets = []; cnt = []; fcnt = []; trans = []
        let N = H * W
        for k in 0..<L {
            let (h, w) = dims(k), nc = h * w, cap = caps[k]
            let cs = mk((nc + 1) * 4), zz = mk(cap * 4), ci = mk(cap * 4)
            if k == 0 {
                // level 0 is the identity CSR: one voxel per pixel, in raster order
                let a = cs.contents().bindMemory(to: UInt32.self, capacity: nc + 1)
                for i in 0...nc { a[i] = UInt32(i) }
                let c = ci.contents().bindMemory(to: UInt32.self, capacity: nc)
                for i in 0..<nc { c[i] = UInt32(i) }
            }
            colStart.append(cs); z.append(zz); column.append(ci)
            neighbours.append(mk(27 * cap * 4))
            tiles.append(mk((cap / tm[k] + 2) * 4))
            var lv = StLevel(H: UInt32(h), W: UInt32(w), Nc: UInt32(nc), M: UInt32(k == 0 ? N : 0),
                             TM: UInt32(tm[k]), tiles: UInt32(k == 0 ? (N + tm[k] - 1) / tm[k] : 0),
                             cap: UInt32(cap), overflow: 0)
            level.append(dev.makeBuffer(bytes: &lv, length: MemoryLayout<StLevel>.stride, options: .storageModeShared)!)
            args.append(mk(6 * 4))
        }
        for k in 0..<(L - 1) {
            let (hf, wf) = dims(k), (hc, wc) = dims(k + 1)
            parent.append(mk(caps[k] * 4)); children.append(mk(caps[k] * 4)); childOffsets.append(mk((caps[k + 1] + 1) * 4))
            cnt.append(mk((hc * wc + 1) * 4)); fcnt.append(mk((hc * wc + 1) * 4))
            var t = StTrans(Hf: UInt32(hf), Wf: UInt32(wf), Hc: UInt32(hc), Wc: UInt32(wc), capC: UInt32(caps[k + 1]))
            trans.append(dev.makeBuffer(bytes: &t, length: MemoryLayout<StTrans>.stride, options: .storageModeShared)!)
        }
    }

    /// Encode the whole build into `enc`. Level voxel counts are valid once the command buffer completes.
    func encode(_ e: MTLComputeCommandEncoder) {
        let N = H * W, nch = (N + chunk - 1) / chunk
        e.setComputePipelineState(p["st_zq"]!)
        e.setBuffer(logz, offset: 0, index: 0); e.setBuffer(z[0], offset: 0, index: 1)
        e.setBuffer(cmin, offset: 0, index: 2); e.setBuffer(quant, offset: 0, index: 3)
        e.dispatchThreads(MTLSize(width: nch, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        e.setComputePipelineState(p["st_zmin"]!)
        e.setBuffer(cmin, offset: 0, index: 0); e.setBuffer(zmin, offset: 0, index: 1); e.setBuffer(quant, offset: 0, index: 2)
        e.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        e.setComputePipelineState(p["st_z0"]!)
        e.setBuffer(z[0], offset: 0, index: 0); e.setBuffer(zmin, offset: 0, index: 1); e.setBuffer(quant, offset: 0, index: 2)
        e.dispatchThreads(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        func clearTiles(_ k: Int) {
            e.setComputePipelineState(p["st_clear_tiles"]!)
            e.setBuffer(tiles[k], offset: 0, index: 0); e.setBuffer(level[k], offset: 0, index: 1)
            if k == 0 {
                e.dispatchThreads(MTLSize(width: (N + tm[0] - 1) / tm[0], height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            } else {
                e.dispatchThreadgroups(indirectBuffer: args[k], indirectBufferOffset: 12,
                                       threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
        }
        func rulebook(_ k: Int) {
            e.setComputePipelineState(p["st_rulebook"]!)
            e.setBuffer(colStart[k], offset: 0, index: 0); e.setBuffer(z[k], offset: 0, index: 1)
            e.setBuffer(column[k], offset: 0, index: 2); e.setBuffer(neighbours[k], offset: 0, index: 3)
            e.setBuffer(tiles[k], offset: 0, index: 4); e.setBuffer(level[k], offset: 0, index: 5)
            if k == 0 {
                e.dispatchThreads(MTLSize(width: N, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            } else {
                e.dispatchThreadgroups(indirectBuffer: args[k], indirectBufferOffset: 0,
                                       threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
        }
        clearTiles(0); rulebook(0)
        for k in 0..<(L - 1) {
            let (hc, wc) = dims(k + 1), ncc = hc * wc
            e.setComputePipelineState(p["st_pool_count"]!)
            e.setBuffer(colStart[k], offset: 0, index: 0); e.setBuffer(z[k], offset: 0, index: 1)
            e.setBuffer(cnt[k], offset: 0, index: 2); e.setBuffer(fcnt[k], offset: 0, index: 3)
            e.setBuffer(trans[k], offset: 0, index: 4)
            e.dispatchThreads(MTLSize(width: wc, height: hc, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            var pn = StPrefix(n: UInt32(ncc))
            for b in [cnt[k], fcnt[k]] {
                e.setComputePipelineState(p["st_prefix"]!)
                e.setBuffer(b, offset: 0, index: 0)
                e.setBytes(&pn, length: MemoryLayout<StPrefix>.stride, index: 1)
                e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            // the coarse colStart IS the prefixed count
            e.setComputePipelineState(p["st_finalize"]!)
            e.setBuffer(cnt[k], offset: 0, index: 0); e.setBuffer(fcnt[k], offset: 0, index: 1)
            e.setBuffer(level[k + 1], offset: 0, index: 2); e.setBuffer(args[k + 1], offset: 0, index: 3)
            e.setBuffer(childOffsets[k], offset: 0, index: 4); e.setBuffer(trans[k], offset: 0, index: 5)
            e.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            e.setComputePipelineState(p["st_pool_emit"]!)
            e.setBuffer(colStart[k], offset: 0, index: 0); e.setBuffer(z[k], offset: 0, index: 1)
            e.setBuffer(cnt[k], offset: 0, index: 2); e.setBuffer(fcnt[k], offset: 0, index: 3)
            e.setBuffer(z[k + 1], offset: 0, index: 4); e.setBuffer(column[k + 1], offset: 0, index: 5)
            e.setBuffer(parent[k], offset: 0, index: 6); e.setBuffer(children[k], offset: 0, index: 7)
            e.setBuffer(childOffsets[k], offset: 0, index: 8); e.setBuffer(trans[k], offset: 0, index: 9)
            e.dispatchThreads(MTLSize(width: wc, height: hc, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            // the coarse level's column CSR is the prefixed count buffer
            colStart[k + 1] = cnt[k]
            clearTiles(k + 1); rulebook(k + 1)
        }
    }

    /// Build from `logz` (already written by the caller). Returns false only if the GPU failed.
    @discardableResult
    func build() -> Bool {
        while true {
            guard let cb = queue.makeCommandBuffer(), let e = cb.makeComputeCommandEncoder() else { return false }
            encode(e)
            e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            guard cb.status == .completed else { return false }
            var grow = false
            M = []
            for k in 0..<L {
                let lv = level[k].contents().bindMemory(to: StLevel.self, capacity: 1).pointee
                M.append(Int(lv.M))
                if lv.overflow != 0 { grow = true; caps[k] = Int(Double(lv.M) * 1.3) + 1024 }
            }
            if !grow { return true }
            allocate()
        }
    }
}
