import Foundation
import CoreVideo
import Accelerate

/// The LiDAR prompt map, built exactly as `moge/model/modules/prompt_stem.py: build_prompt`.
///
/// Channels, in order: normalised log depth (0 where invalid), valid, conf/levels, uncertainty.
/// The normalisation is self-referential -- centred on the median log-depth of the CONFIDENT
/// pixels (falling back to all valid pixels when there are none) -- so no external calibration
/// is involved.
enum PromptBuilder {
    static let logClamp: Float = 4.0
    static let rangeKnee: Float = 3.0
    static let rangeFull: Float = 5.0

    /// depth: metres, <=0 or non-finite means missing. conf in {0,1,2}.
    /// Returns a planar [4, H, W] buffer, channel-major, ready to resize into the two shapes
    /// the prompt models take.
    static func build(depth: [Float], conf: [UInt8], w: Int, h: Int) -> [Float] {
        let n = w * h
        var out = [Float](repeating: 0, count: 4 * n)
        var logd = [Float](repeating: 0, count: n)
        var nHi = 0, nValid = 0
        out.withUnsafeMutableBufferPointer { o in
            logd.withUnsafeMutableBufferPointer { ld in
                depth.withUnsafeBufferPointer { dp in
                    conf.withUnsafeBufferPointer { cp in
                        for i in 0..<n {
                            let d = dp[i]
                            guard d.isFinite, d > 0 else { o[3 * n + i] = 1; continue }
                            ld[i] = log(d)
                            let c = min(max(Float(cp[i]) / 2.0, 0), 1)
                            o[n + i] = 1                      // valid
                            o[2 * n + i] = c                  // conf * valid
                            nValid += 1
                            if c >= 0.999 { nHi += 1 }
                        }
                    }
                }
            }
        }
        // mu: median log-depth of the confident pixels, else of all valid ones. Subsampled --
        // the centre is stable to far better than the fp16 datapath's resolution.
        let usingHi = nHi > 0
        let poolCount = usingHi ? nHi : nValid
        var mu: Float = 0
        if poolCount > 0 {
            let step = max(1, poolCount / 12000)
            var vals = [Float](); vals.reserveCapacity(poolCount / step + 1)
            var seen = 0
            for i in 0..<n {
                guard out[n + i] > 0 else { continue }
                if usingHi && out[2 * n + i] < 0.999 { continue }
                if seen % step == 0 { vals.append(logd[i]) }
                seen += 1
            }
            if !vals.isEmpty { vals.sort(); mu = vals[vals.count / 2] }
        }
        out.withUnsafeMutableBufferPointer { o in
            depth.withUnsafeBufferPointer { dp in
                for i in 0..<n {
                    guard o[n + i] > 0 else { continue }
                    o[i] = min(max(logd[i] - mu, -logClamp), logClamp)
                    let range = min(max((dp[i] - rangeKnee) / (rangeFull - rangeKnee), 0), 1)
                    o[3 * n + i] = 1 - o[2 * n + i] * (1 - range)
                }
            }
        }
        return out
    }

    /// Nearest-fill the sensor planes: every invalid pixel takes the depth of its EUCLIDEAN-nearest valid pixel and
    /// confidence 1; valid pixels keep theirs (`promptmoge.nearest_fill`). With holes in the prompt Model A's AbsRel
    /// goes 0.0138 -> 0.0269 at 50 % of pixels dropped, and filling restores 0.0139. The ARKit `sceneDepth` prompt
    /// the models were trained on is dense; the AVFoundation sensor return this app captures is not.
    ///
    /// Exact, not approximate: the squared Euclidean distance transform in two phases (nearest valid row per column,
    /// then the lower envelope of parabolas along each row -- Felzenszwalb & Huttenlocher), carrying the source pixel
    /// so the depth comes from the true nearest neighbour rather than a Manhattan one.
    /// Returns the filled planes and how many pixels were filled.
    static func nearestFill(depth: [Float], conf: [UInt8], w: Int, h: Int) -> (depth: [Float], conf: [UInt8], filled: Int) {
        let n = w * h
        var valid = [Bool](repeating: false, count: n)
        var invalidCount = 0
        for i in 0..<n {
            valid[i] = depth[i].isFinite && depth[i] > 0
            if !valid[i] { invalidCount += 1 }
        }
        if invalidCount == 0 || invalidCount == n { return (depth, conf, 0) }

        let INF = Float(1e20)
        // phase 1: per column, the squared distance to the nearest valid pixel in that column, and its row
        var g = [Float](repeating: INF, count: n)
        var srcY = [Int32](repeating: -1, count: n)
        for x in 0..<w {
            var last = -1
            for y in 0..<h {
                let i = y * w + x
                if valid[i] { last = y }
                if last >= 0 { let d = Float(y - last); g[i] = d * d; srcY[i] = Int32(last) }
            }
            last = -1
            for y in stride(from: h - 1, through: 0, by: -1) {
                let i = y * w + x
                if valid[i] { last = y }
                if last >= 0 {
                    let d = Float(last - y)
                    if d * d < g[i] { g[i] = d * d; srcY[i] = Int32(last) }
                }
            }
        }
        // phase 2: per row, the lower envelope of the parabolas f(x') = g[x'] + (x - x')^2
        var outD = depth, outC = conf
        var vtx = [Int](repeating: 0, count: w)
        var bnd = [Float](repeating: 0, count: w + 1)
        for y in 0..<h {
            let row = y * w
            var k = 0
            vtx[0] = 0; bnd[0] = -INF; bnd[1] = INF
            for q in 1..<w {
                if g[row + q] >= INF { continue }
                var s: Float = 0
                while true {
                    let p = vtx[k]
                    if g[row + p] >= INF { k -= 1; if k < 0 { break }; continue }
                    s = ((g[row + q] + Float(q * q)) - (g[row + p] + Float(p * p))) / Float(2 * q - 2 * p)
                    if s <= bnd[k] { k -= 1; if k < 0 { break } } else { break }
                }
                k += 1
                if k < 0 { k = 0 }
                // the first parabola on the row owns everything to its left
                vtx[k] = q; bnd[k] = (k == 0 ? -INF : s); bnd[k + 1] = INF
            }
            guard k >= 0, g[row + vtx[0]] < INF else { continue }
            var kk = 0
            for x in 0..<w {
                while bnd[kk + 1] < Float(x) { kk += 1 }
                let i = row + x
                if valid[i] { continue }
                let sx = vtx[kk]
                let sy = Int(srcY[row + sx])
                guard sy >= 0 else { continue }
                outD[i] = depth[sy * w + sx]
                outC[i] = 1
            }
        }
        return (outD, outC, invalidCount)
    }
}
