import SwiftUI
import MetalKit
import simd

/// What the viewer can show for a capture: the raw LiDAR, and each model after K refinement steps.
enum DepthSource: String, CaseIterable, Identifiable {
    case lidar = "LiDAR"
    case a0 = "A K=0", a1 = "A K=1", a3 = "A K=3"
    case b0 = "B K=0", b1 = "B K=1", b3 = "B K=3"
    var id: String { rawValue }
    static let modelA: [DepthSource] = [.a0, .a1, .a3], modelB: [DepthSource] = [.b0, .b1, .b3]
}

@MainActor
final class ScanState: ObservableObject {
    enum Phase { case preview, working, viewing }
    @Published var phase: Phase = .preview
    @Published var status = "loading models…"
    @Published var source: DepthSource = .a1
    @Published var marks: [SIMD3<Float>] = []
    @Published var measurements: [(SIMD3<Float>, SIMD3<Float>, Float)] = []
    @Published var timings: [DepthSource: Double] = [:]
    @Published var visible = 0
    /// Flying-pixel filter strength and the model's mask, adjustable in the viewer without re-running inference.
    @Published var edgeReject: Float = 0.03
    @Published var useMask = true
    @Published var ready = false
    /// Folder under Documents/captures that holds the current capture's images and maps.
    @Published var savedAs: String?

    let capture = LiDARCapture()
    let renderer: PointCloudRenderer? = MTLCreateSystemDefaultDevice().flatMap { PointCloudRenderer(device: $0) }
    private var clouds: [DepthSource: PointCloud] = [:]
    private var pipelines: [(DepthPipeline, [DepthSource], String)] = []

    /// Models are read from Documents/models/{vit, A, B} (see the README for how to put them there).
    nonisolated static let modelsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("models")

    func loadModels() async {
        guard pipelines.isEmpty else { return }
        let t0 = CFAbsoluteTimeGetCurrent()
        guard let vit = await DepthPipeline.loadViT(dir: Self.modelsDir.appendingPathComponent("vit")) else {
            status = "models missing: copy them to Documents/models"; Log.write("ViT load failed"); return
        }
        for (name, sources) in [("A", DepthSource.modelA), ("B", DepthSource.modelB)] {
            guard let p = await DepthPipeline(vit: vit, dir: Self.modelsDir.appendingPathComponent(name)) else {
                Log.write("Model \(name): missing"); continue
            }
            // The first call of each Core ML model loads its program and the first refiner step allocates its
            // activations: run once on a synthetic frame so the first real capture is warm.
            let dw = 256, dh = 192
            let depth = (0..<(dw * dh)).map { 1.2 + 0.004 * Float($0 % dw) + 0.002 * Float($0 / dw) }
            let conf = [UInt8](repeating: 2, count: dw * dh), grey = [UInt8](repeating: 128, count: 640 * 480 * 4)
            p.prepareImage(rgba: grey, width: 640, height: 480)
            for _ in 0..<2 { p.run(depth: depth, conf: conf, dw: dw, dh: dh, steps: 1) }
            pipelines.append((p, sources, name))
        }
        ready = !pipelines.isEmpty
        status = ready ? String(format: "models ready (%.0f s)", CFAbsoluteTimeGetCurrent() - t0) : "models missing: copy them to Documents/models"
        Log.write("models: \(pipelines.map { $0.2 }) loaded in \(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - t0)) s")
    }

    func shoot() {
        guard let f = capture.capture(), !f.depth.isEmpty else { status = "LiDAR is still starting"; return }
        let valid = f.depth.filter { $0 > 0 }.count
        Log.write("capture: image \(f.rgbW)x\(f.rgbH), LiDAR \(f.depthW)x\(f.depthH) with \(valid) valid px")
        // Without a single sensor return there is nothing to attach metric scale to.
        guard valid >= 32 else { status = "no LiDAR return — is the sensor covered or too close?"; return }
        phase = .working
        clearMeasurements()
        // The frame is in hand, so the camera has nothing left to contribute: left running, its 30 fps of ISP work and
        // preview compositing competes with the ViT's GPU attention (2.8x slower for Model A). `back()` restarts it.
        capture.pause()
        let pipelines = self.pipelines
        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            var clouds: [DepthSource: PointCloud] = [.lidar: Self.sensorCloud(f)]
            var store = CaptureStore(frame: f)
            var timings: [DepthSource: Double] = [:]
            for (pipe, sources, name) in pipelines {
                let t0 = CFAbsoluteTimeGetCurrent()
                pipe.prepareImage(rgba: f.rgba, width: f.rgbW, height: f.rgbH)
                // One run yields all three sources: the per-step hook snapshots K=0 and K=1 before the refiner
                // overwrites the log-depth, and what is left after the loop is K=3. The hook's own metric fit is not
                // part of any result, so its time is subtracted from every reported latency.
                var results: [(logz: [Float], s: Float, t: Float, seconds: Double)] = []
                var overhead = 0.0
                let ok = pipe.run(depth: f.depth, conf: f.conf, dw: f.depthW, dh: f.depthH, steps: 3) { k in
                    let tk = CFAbsoluteTimeGetCurrent()
                    pipe.encodeFinal()
                    if k < 2 { results.append((pipe.logz, pipe.metricScale, pipe.metricShift, tk - t0 - overhead)) }
                    overhead += CFAbsoluteTimeGetCurrent() - tk
                }
                guard ok else { Log.write("Model \(name) FAILED"); continue }
                results.append((pipe.logz, pipe.metricScale, pipe.metricShift, CFAbsoluteTimeGetCurrent() - t0 - overhead))
                store?.add(model: name, pipe: pipe, results: zip([0, 1, 3], results).map { ($0, $1.logz, $1.s, $1.t, $1.seconds) })
                let template = Self.modelCloud(f, pipe)
                for (source, r) in zip(sources, results) {
                    var c = template
                    c.logz = r.logz
                    for i in 0..<c.depth.count {
                        let d = r.s * exp(r.logz[i]) + r.t
                        c.depth[i] = c.maskLogit[i] > 0 && d > 0 ? d : .infinity
                    }
                    clouds[source] = c; timings[source] = r.seconds
                }
                Log.write("Model \(name) \(pipe.W)x\(pipe.H): "
                          + zip(sources, results).map { String(format: "%@ %.1f ms", $0.0.rawValue, $0.1.seconds * 1000) }.joined(separator: ", ")
                          + " | " + pipe.stages.map { String(format: "%@ %.1f", $0.0, $0.1) }.joined(separator: ", ")
                          + String(format: " | metric s %.4f t %.4f, filled %d px", pipe.metricScale, pipe.metricShift, pipe.filledPixels))
            }
            store?.finish()
            Log.write("saved " + (store.map { "captures/" + $0.dir.lastPathComponent } ?? "FAILED"))
            let (c, t, saved) = (clouds, timings, store?.dir.lastPathComponent)
            await MainActor.run {
                self.clouds = c; self.timings = t; self.savedAs = saved
                self.show(c[self.source] != nil ? self.source : .lidar, resetView: true)
                self.phase = .viewing
            }
        }
    }

    /// Intrinsics of the captured image scaled to a w x h grid: fx, fy, cx, cy.
    private nonisolated static func intrinsics(_ f: CapturedFrame, _ w: Int, _ h: Int) -> SIMD4<Float> {
        let k = f.intrinsics, sx = Float(w) / Float(f.rgbW), sy = Float(h) / Float(f.rgbH)
        return SIMD4<Float>(k[0][0] * sx, k[1][1] * sy, k[2][0] * sx, k[2][1] * sy)
    }

    private nonisolated static func colors(_ f: CapturedFrame, _ w: Int, _ h: Int) -> [SIMD3<Float>] {
        var out = [SIMD3<Float>](repeating: .zero, count: w * h)
        for y in 0..<h {
            let row = min(f.rgbH - 1, y * f.rgbH / h) * f.rgbW
            for x in 0..<w {
                let o = (row + min(f.rgbW - 1, x * f.rgbW / w)) * 4
                out[y * w + x] = SIMD3<Float>(Float(f.rgba[o]), Float(f.rgba[o + 1]), Float(f.rgba[o + 2])) / 255
            }
        }
        return out
    }

    private nonisolated static func sensorCloud(_ f: CapturedFrame) -> PointCloud {
        let w = f.depthW, h = f.depthH
        return PointCloud(points: [SIMD3<Float>](repeating: .zero, count: w * h), colors: colors(f, w, h), width: w, height: h,
                          depth: f.depth, intrinsics: intrinsics(f, w, h)).filtered(edgeReject: 0, useMask: false)
    }

    private nonisolated static func modelCloud(_ f: CapturedFrame, _ pipe: DepthPipeline) -> PointCloud {
        let w = pipe.W, h = pipe.H, n = w * h
        let mask = UnsafeBufferPointer(start: pipe.maskLogit.contents().bindMemory(to: Float16.self, capacity: n), count: n).map { Float($0) }
        return PointCloud(points: [SIMD3<Float>](repeating: .zero, count: n), colors: colors(f, w, h), width: w, height: h,
                          depth: [Float](repeating: .infinity, count: n), maskLogit: mask, intrinsics: intrinsics(f, w, h))
    }

    func show(_ s: DepthSource, resetView: Bool = false) {
        source = s
        guard let c = clouds[s], let r = renderer else { return }
        let shown = s == .lidar ? c : c.filtered(edgeReject: edgeReject, useMask: useMask)
        r.upload(points: shown.points, colors: shown.colors, resetView: resetView)
        r.marks = marks
        visible = shown.points.reduce(0) { $0 + ($1.z.isFinite ? 1 : 0) }
    }

    func available(_ s: DepthSource) -> Bool { clouds[s] != nil }

    func tap(_ p: CGPoint, _ size: CGSize) {
        guard let r = renderer, let hit = r.pick(at: p, in: size) else { return }
        marks.append(hit)
        if marks.count % 2 == 0 {
            let a = marks[marks.count - 2], b = marks[marks.count - 1]
            measurements.append((a, b, simd_distance(a, b)))
        }
        r.marks = marks
    }

    func clearMeasurements() { marks = []; measurements = []; renderer?.marks = [] }

    func back() {
        phase = .preview
        clearMeasurements()
        capture.start()
    }
}

struct ScanView: View {
    @StateObject private var st = ScanState()
    @State private var selfTest: String?

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            if let selfTest {
                ScrollView { Text(selfTest).font(.system(size: 11, design: .monospaced)).foregroundStyle(.white).padding() }
            } else if st.phase == .viewing { viewer } else { preview }
        }
        .preferredColorScheme(.dark)
        .task {
            // Documents/SELFTEST containing "1" runs the on-device check against the exported reference instead.
            if SelfTest.requested {
                selfTest = "running self-test…"
                selfTest = await SelfTest.run()
                return
            }
            st.capture.start()
            await st.loadModels()
            // Documents/AUTOSHOOT containing "1" takes one capture without a tap, for driving the app over the wire.
            if SelfTest.flag("AUTOSHOOT") {
                for _ in 0..<40 where !st.capture.depthReady { try? await Task.sleep(nanoseconds: 500_000_000) }
                st.shoot()
            }
        }
    }

    private var preview: some View {
        ZStack {
            CameraPreview(session: st.capture.session).ignoresSafeArea()
            VStack {
                HStack {
                    Text(st.status).font(.caption).foregroundStyle(.white)
                        .padding(8).background(.black.opacity(0.5)).clipShape(Capsule())
                    Spacer()
                    if !st.capture.hasLiDAR {
                        Text("no LiDAR").font(.caption).foregroundStyle(.orange)
                    } else if !st.capture.depthReady {
                        Text("starting LiDAR…").font(.caption).foregroundStyle(.orange)
                    }
                }.padding()
                Spacer()
                if st.phase == .working {
                    ProgressView("running…").tint(.white).foregroundStyle(.white)
                        .padding().background(.black.opacity(0.6)).clipShape(RoundedRectangle(cornerRadius: 12))
                } else {
                    Button(action: st.shoot) {
                        ZStack {
                            Circle().stroke(.white, lineWidth: 4).frame(width: 78, height: 78)
                            Circle().fill(.white).frame(width: 64, height: 64)
                        }
                    }
                    .disabled(!st.ready || !st.capture.depthReady)
                    .opacity(st.ready && st.capture.depthReady ? 1 : 0.4)
                }
                Spacer().frame(height: 28)
            }
        }
    }

    private var viewer: some View {
        ZStack {
            if let r = st.renderer {
                PointCloudView(renderer: r) { p, s in st.tap(p, s) }.ignoresSafeArea()
            }
            VStack {
                HStack {
                    Button("← Camera") { st.back() }.buttonStyle(.borderedProminent).tint(.white.opacity(0.15))
                    Spacer()
                    if let t = st.timings[st.source] {
                        Text(String(format: "%.0f ms", t * 1000)).font(.caption.monospaced()).foregroundStyle(.white.opacity(0.8))
                    }
                    Text("\(st.visible) pts").font(.caption2.monospaced()).foregroundStyle(.white.opacity(0.6))
                }.padding(.horizontal)
                if let saved = st.savedAs {
                    Text("saved to captures/\(saved)").font(.caption2.monospaced()).foregroundStyle(.white.opacity(0.6))
                        .frame(maxWidth: .infinity, alignment: .trailing).padding(.horizontal)
                }

                Picker("Source", selection: Binding(get: { st.source }, set: { st.show($0) })) {
                    ForEach(DepthSource.allCases.filter(st.available)) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented).padding(.horizontal)

                if st.source != .lidar {
                    HStack(spacing: 10) {
                        Toggle("mask", isOn: Binding(get: { st.useMask }, set: { st.useMask = $0; st.show(st.source) })).labelsHidden()
                        Text("mask").font(.caption2).foregroundStyle(.white.opacity(0.8))
                        Text("edges").font(.caption2).foregroundStyle(.white.opacity(0.8))
                        Picker("", selection: Binding(get: { st.edgeReject }, set: { st.edgeReject = $0; st.show(st.source) })) {
                            Text("off").tag(Float(0))
                            Text("mild").tag(Float(0.05))
                            Text("med").tag(Float(0.03))
                            Text("hard").tag(Float(0.012))
                        }
                        .pickerStyle(.segmented).frame(width: 240)
                    }
                    .padding(.horizontal)
                }
                Spacer()
                measurePanel
            }
            .padding(.top, 8)
        }
    }

    private var measurePanel: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(st.marks.count % 2 == 1 ? "tap the second point" : "tap two points to measure")
                    .font(.caption).foregroundStyle(.white.opacity(0.85))
                Spacer()
                if !st.measurements.isEmpty { Button("Clear") { st.clearMeasurements() }.font(.caption) }
            }
            ForEach(Array(st.measurements.enumerated()), id: \.offset) { i, m in
                Text(String(format: "%d.  %.1f cm", i + 1, m.2 * 100))
                    .font(.system(.callout, design: .monospaced)).foregroundStyle(.yellow)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.black.opacity(0.55))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding()
    }
}
