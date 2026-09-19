import Foundation

/// On-device check of an export against the PyTorch reference, and warm latency.
///
/// `python -m promptmoge.export.selftest` writes, next to each model, one frame and the reference depth after every
/// refinement step. With Documents/SELFTEST containing "1" the app runs the full pipeline on that frame instead of
/// opening the camera, reports the relative depth error against the reference per step, then times K = 1.
/// The report is shown on screen and written to Documents/selftest.txt.
enum SelfTest {
    struct Meta: Decodable { let image_shape: [Int]; let lidar_shape: [Int]; let steps: Int }

    private static let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    /// Content-based flags: `devicectl` can overwrite a file in the app container but not delete one.
    static func flag(_ name: String) -> Bool {
        (try? String(contentsOf: docs.appendingPathComponent(name), encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines) == "1"
    }
    static var requested: Bool { flag("SELFTEST") }

    private static func floats(_ url: URL) -> [Float]? {
        (try? Data(contentsOf: url, options: .alwaysMapped))?.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// F.interpolate(mode="bilinear", align_corners=False): how the reference brings the point map to image size.
    private static func bilinear(_ src: [Float], h: Int, w: Int, oh: Int, ow: Int) -> [Float] {
        var out = [Float](repeating: 0, count: oh * ow)
        let sy = Float(h) / Float(oh), sx = Float(w) / Float(ow)
        for y in 0..<oh {
            let fy = max((Float(y) + 0.5) * sy - 0.5, 0), y0 = min(Int(fy), h - 1), y1 = min(y0 + 1, h - 1), wy = fy - Float(y0)
            for x in 0..<ow {
                let fx = max((Float(x) + 0.5) * sx - 0.5, 0), x0 = min(Int(fx), w - 1), x1 = min(x0 + 1, w - 1), wx = fx - Float(x0)
                let a = src[y0 * w + x0] * (1 - wx) + src[y0 * w + x1] * wx
                let b = src[y1 * w + x0] * (1 - wx) + src[y1 * w + x1] * wx
                out[y * ow + x] = a * (1 - wy) + b * wy
            }
        }
        return out
    }

    static func run() async -> String {
        var report = ""
        func say(_ s: String) { report += s + "\n"; Log.write(s) }
        guard let vit = await DepthPipeline.loadViT(dir: ScanState.modelsDir.appendingPathComponent("vit")) else { return "ViT missing" }
        for name in ["A", "B"] {
            let dir = ScanState.modelsDir.appendingPathComponent(name), test = dir.appendingPathComponent("selftest")
            guard let pipe = await DepthPipeline(vit: vit, dir: dir) else { say("Model \(name): missing"); continue }
            guard let md = try? Data(contentsOf: test.appendingPathComponent("selftest.json")),
                  let meta = try? JSONDecoder().decode(Meta.self, from: md),
                  let image = floats(test.appendingPathComponent("image.bin")),
                  let lidar = floats(test.appendingPathComponent("lidar_depth.bin")),
                  let conf = (try? Data(contentsOf: test.appendingPathComponent("lidar_conf.bin"))).map({ [UInt8]($0) }),
                  let reference = floats(test.appendingPathComponent("depth.bin")) else { say("Model \(name): no selftest data"); continue }
            say("Model \(name): \(pipe.W)x\(pipe.H), \(pipe.levels) levels")
            let (dh, dw, ih, iw) = (meta.lidar_shape[0], meta.lidar_shape[1], meta.image_shape[0], meta.image_shape[1])
            DepthPipeline.write(pipe.imageArr, planar: image, c: 3, h: DepthPipeline.rows * 14, w: DepthPipeline.cols * 14)

            // accuracy: depth after every refinement step against the reference, at image resolution
            var states: [([Float], Float, Float)] = []
            pipe.run(depth: lidar, conf: conf, dw: dw, dh: dh, steps: meta.steps) { _ in
                pipe.encodeFinal(); states.append((pipe.logz, pipe.metricScale, pipe.metricShift))
            }
            states.append((pipe.logz, pipe.metricScale, pipe.metricShift))
            let n = pipe.N
            let mask = bilinear(UnsafeBufferPointer(start: pipe.maskLogit.contents().bindMemory(to: Float16.self, capacity: n), count: n).map { Float($0) },
                                h: pipe.H, w: pipe.W, oh: ih, ow: iw)
            for (k, (logz, s, t)) in states.enumerated() {
                let up = bilinear(logz, h: pipe.H, w: pipe.W, oh: ih, ow: iw)
                var sum = 0.0, count = 0, disagree = 0
                for i in 0..<(ih * iw) {
                    let d = s * exp(up[i]) + t, g = reference[k * ih * iw + i]
                    let dv = mask[i] > 0 && d > 0, gv = g.isFinite && g > 0
                    if dv != gv { disagree += 1 }
                    if dv && gv { sum += Double(abs(d - g) / g); count += 1 }
                }
                say(String(format: "  K=%d  AbsRel vs PyTorch %.2e   mask disagreement %.3f %%   metric s %.5f t %.5f",
                           k, sum / Double(max(count, 1)), 100 * Double(disagree) / Double(ih * iw), s, t))
            }

            // latency: K = 1, warm
            for _ in 0..<2 { pipe.run(depth: lidar, conf: conf, dw: dw, dh: dh, steps: 1) }
            var totals: [Double] = [], best: [String: Double] = [:], order: [String] = []
            for _ in 0..<10 {
                let t0 = CFAbsoluteTimeGetCurrent()
                pipe.run(depth: lidar, conf: conf, dw: dw, dh: dh, steps: 1)
                totals.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
                for (stage, ms) in pipe.stages {
                    if best[stage] == nil { order.append(stage) }
                    best[stage] = min(best[stage] ?? .infinity, ms)
                }
            }
            totals.sort()
            say(String(format: "  K=1 end to end, warm: best %.1f ms, median %.1f ms", totals[0], totals[totals.count / 2]))
            say("    " + order.map { String(format: "%@ %.1f", $0, best[$0]!) }.joined(separator: " | "))
        }
        try? report.write(to: docs.appendingPathComponent("selftest.txt"), atomically: true, encoding: .utf8)
        return report
    }
}
