import Foundation
import simd

/// A depth map with everything needed to unproject and re-filter it without another forward pass.
struct PointCloud {
    var points: [SIMD3<Float>]
    var colors: [SIMD3<Float>]
    var width: Int, height: Int
    var depth: [Float]                 // metres, row-major
    var logz: [Float] = []             // the model's log-depth (empty for the raw sensor)
    var maskLogit: [Float] = []
    var intrinsics: SIMD4<Float> = .zero   // fx, fy, cx, cy at width x height

    /// Unproject, dropping pixels the model masks out and flying pixels: a pixel whose log-depth differs from the
    /// median of its 3x3 neighbourhood by more than `edgeReject` sits on a depth discontinuity.
    func filtered(edgeReject: Float, useMask: Bool) -> PointCloud {
        let W = width, H = height, n = W * H
        var keep = [Bool](repeating: true, count: n)
        if useMask, maskLogit.count >= n {
            for i in 0..<n where maskLogit[i] <= 0 { keep[i] = false }
        }
        if edgeReject > 0, logz.count >= n {
            var win = [Float](repeating: 0, count: 9)
            for y in 1..<(H - 1) {
                for x in 1..<(W - 1) {
                    var k = 0
                    for dy in -1...1 { for dx in -1...1 { win[k] = logz[(y + dy) * W + (x + dx)]; k += 1 } }
                    win.sort()
                    if abs(logz[y * W + x] - win[4]) > edgeReject { keep[y * W + x] = false }
                }
            }
            for x in 0..<W { keep[x] = false; keep[(H - 1) * W + x] = false }
            for y in 0..<H { keep[y * W] = false; keep[y * W + W - 1] = false }
        }
        var c = self
        let fx = intrinsics.x, fy = intrinsics.y, cx = intrinsics.z, cy = intrinsics.w
        for y in 0..<H {
            for x in 0..<W {
                let i = y * W + x, z = depth[i]
                c.points[i] = keep[i] && z.isFinite && z > 0.05 && z < 12
                    ? SIMD3<Float>((Float(x) - cx) / fx * z, (Float(y) - cy) / fy * z, z) : SIMD3<Float>(0, 0, .nan)
            }
        }
        return c
    }
}
