import SwiftUI
import MetalKit
import simd

/// Orbit camera state, shared between the renderer and the gesture handlers.
struct OrbitCamera {
    var target = SIMD3<Float>(0, 0, 1.5)
    var yaw: Float = 0, pitch: Float = 0
    var distance: Float = 2.5

    /// The cloud lives in camera space: +X right, +Y DOWN, +Z forward into the scene. The
    /// default eye therefore sits BEHIND the origin looking along +Z (otherwise the viewer
    /// starts behind the cloud looking at its back), and "up" is -Y (otherwise the scene is
    /// upside down).
    var eye: SIMD3<Float> {
        let cp = cos(pitch), sp = sin(pitch), cy = cos(yaw), sy = sin(yaw)
        return target + distance * SIMD3<Float>(cp * sy, -sp, -cp * cy)
    }

    func view() -> float4x4 {
        let e = eye, up = SIMD3<Float>(0, -1, 0)
        let f = normalize(target - e), s = normalize(cross(f, up)), u = cross(s, f)
        var m = float4x4(1)
        m.columns.0 = SIMD4<Float>(s.x, u.x, -f.x, 0)
        m.columns.1 = SIMD4<Float>(s.y, u.y, -f.y, 0)
        m.columns.2 = SIMD4<Float>(s.z, u.z, -f.z, 0)
        m.columns.3 = SIMD4<Float>(-dot(s, e), -dot(u, e), dot(f, e), 1)
        return m
    }

    static func perspective(_ fovY: Float, _ aspect: Float, _ n: Float, _ f: Float) -> float4x4 {
        let t = 1 / tan(fovY / 2)
        var m = float4x4(0)
        m.columns.0 = SIMD4<Float>(t / aspect, 0, 0, 0)
        m.columns.1 = SIMD4<Float>(0, t, 0, 0)
        m.columns.2 = SIMD4<Float>(0, 0, (f + n) / (n - f), -1)
        m.columns.3 = SIMD4<Float>(0, 0, 2 * f * n / (n - f), 0)
        return m
    }
}

/// Point-cloud renderer with orbit controls and tap-to-pick for measurement.
final class PointCloudRenderer: NSObject, MTKViewDelegate {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private var pipeline: MTLRenderPipelineState!
    private var linePipeline: MTLRenderPipelineState!
    private var depthState: MTLDepthStencilState!
    private var posBuf: MTLBuffer?, colBuf: MTLBuffer?, markBuf: MTLBuffer?
    private(set) var count = 0
    private var points: [SIMD3<Float>] = []

    var camera = OrbitCamera()
    var pointSize: Float = 4
    var marks: [SIMD3<Float>] = [] { didSet { uploadMarks() } }
    var viewportSize = CGSize(width: 1, height: 1)

    init?(device: MTLDevice) {
        guard let q = device.makeCommandQueue() else { return nil }
        self.device = device; self.queue = q
        super.init()
        guard let lib = device.makeDefaultLibrary(),
              let vf = lib.makeFunction(name: "pc_vertex"),
              let ff = lib.makeFunction(name: "pc_fragment"),
              let mv = lib.makeFunction(name: "marker_vertex"),
              let mf = lib.makeFunction(name: "marker_fragment") else { return nil }
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = vf; d.fragmentFunction = ff
        d.colorAttachments[0].pixelFormat = .bgra8Unorm
        d.depthAttachmentPixelFormat = .depth32Float
        pipeline = try? device.makeRenderPipelineState(descriptor: d)
        let l = MTLRenderPipelineDescriptor()
        l.vertexFunction = mv; l.fragmentFunction = mf
        l.colorAttachments[0].pixelFormat = .bgra8Unorm
        l.depthAttachmentPixelFormat = .depth32Float
        linePipeline = try? device.makeRenderPipelineState(descriptor: l)
        let ds = MTLDepthStencilDescriptor()
        ds.depthCompareFunction = .less; ds.isDepthWriteEnabled = true
        depthState = device.makeDepthStencilState(descriptor: ds)
        if pipeline == nil || linePipeline == nil { return nil }
    }

    /// `resetView` only on a NEW capture. Switching source or changing a filter re-uploads
    /// the buffers, and re-framing there would throw away the orientation the user has set --
    /// which is exactly when they are comparing two clouds and need the view to hold still.
    func upload(points p: [SIMD3<Float>], colors c: [SIMD3<Float>], resetView: Bool = false) {
        points = p
        count = p.count
        // MemoryLayout<SIMD3<Float>>.stride is 16 bytes (SIMD3 is padded to 4 lanes) while the
        // shader reads `packed_float3` at 12. Uploading the SIMD3 array directly makes the GPU
        // walk the buffer at 3/4 the rate the CPU wrote it, which draws the cloud four times at
        // four phase offsets with the colour channels rotated. Pack to 3 floats explicitly.
        var pp = [Float](repeating: 0, count: p.count * 3)
        var cc = [Float](repeating: 0, count: c.count * 3)
        for i in 0..<p.count { pp[i*3] = p[i].x; pp[i*3+1] = p[i].y; pp[i*3+2] = p[i].z }
        for i in 0..<c.count { cc[i*3] = c[i].x; cc[i*3+1] = c[i].y; cc[i*3+2] = c[i].z }
        posBuf = device.makeBuffer(bytes: &pp, length: MemoryLayout<Float>.stride * pp.count)
        colBuf = device.makeBuffer(bytes: &cc, length: MemoryLayout<Float>.stride * cc.count)
        // frame the cloud: centre on the median finite point so one stray value cannot throw it
        var zs = resetView ? p.compactMap { $0.z.isFinite ? $0.z : nil } : []
        if !zs.isEmpty {
            zs.sort()
            let mid = zs[zs.count / 2]
            camera.target = SIMD3<Float>(0, 0, mid)
            camera.distance = max(0.8, mid * 1.6)
        }
    }

    private func uploadMarks() {
        guard !marks.isEmpty else { markBuf = nil; return }
        var m = [Float](repeating: 0, count: marks.count * 3)
        for i in 0..<marks.count { m[i*3] = marks[i].x; m[i*3+1] = marks[i].y; m[i*3+2] = marks[i].z }
        markBuf = device.makeBuffer(bytes: &m, length: MemoryLayout<Float>.stride * m.count)
    }

    private func mvp() -> float4x4 {
        let aspect = Float(max(viewportSize.width, 1) / max(viewportSize.height, 1))
        return OrbitCamera.perspective(60 * .pi / 180, aspect, 0.05, 60) * camera.view()
    }

    /// Nearest visible point to a tap, in view coordinates. 300 k points is a trivial CPU scan
    /// and avoids an ID-buffer round trip.
    func pick(at p: CGPoint, in size: CGSize) -> SIMD3<Float>? {
        let m = mvp()
        var best: (Float, SIMD3<Float>)? = nil
        let tx = Float(p.x / size.width) * 2 - 1
        let ty = 1 - Float(p.y / size.height) * 2
        for q in points {
            guard q.z.isFinite, q.x.isFinite, q.y.isFinite else { continue }
            let c = m * SIMD4<Float>(q, 1)
            guard c.w > 0 else { continue }
            let sx = c.x / c.w, sy = c.y / c.w
            let d = (sx - tx) * (sx - tx) + (sy - ty) * (sy - ty)
            if d < 0.0009 {                       // ~3 % of the viewport
                let depth = c.z / c.w
                if best == nil || depth < best!.0 { best = (depth, q) }
            }
        }
        return best?.1
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) { viewportSize = size }

    func draw(in view: MTKView) {
        guard let rp = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let cb = queue.makeCommandBuffer(),
              let enc = cb.makeRenderCommandEncoder(descriptor: rp) else { return }
        viewportSize = view.drawableSize
        var u = (mvp: mvp(), size: pointSize, pad: SIMD3<Float>(0, 0, 0))
        if let pb = posBuf, let cbuf = colBuf, count > 0 {
            enc.setRenderPipelineState(pipeline)
            enc.setDepthStencilState(depthState)
            enc.setVertexBuffer(pb, offset: 0, index: 0)
            enc.setVertexBuffer(cbuf, offset: 0, index: 1)
            enc.setVertexBytes(&u, length: MemoryLayout.size(ofValue: u), index: 2)
            enc.drawPrimitives(type: .point, vertexStart: 0, vertexCount: count)
        }
        if let mb = markBuf, marks.count >= 2 {
            enc.setRenderPipelineState(linePipeline)
            enc.setVertexBuffer(mb, offset: 0, index: 0)
            enc.setVertexBytes(&u, length: MemoryLayout.size(ofValue: u), index: 2)
            var col = SIMD4<Float>(1, 0.85, 0.1, 1)
            enc.setFragmentBytes(&col, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
            enc.drawPrimitives(type: .line, vertexStart: 0, vertexCount: marks.count)
        }
        enc.endEncoding()
        cb.present(drawable)
        cb.commit()
    }
}

struct PointCloudView: UIViewRepresentable {
    let renderer: PointCloudRenderer
    var onTap: (CGPoint, CGSize) -> Void

    func makeUIView(context: Context) -> MTKView {
        let v = MTKView(frame: .zero, device: MTLCreateSystemDefaultDevice())
        v.delegate = renderer
        v.colorPixelFormat = .bgra8Unorm
        v.depthStencilPixelFormat = .depth32Float
        v.clearColor = MTLClearColorMake(0.04, 0.04, 0.05, 1)
        v.preferredFramesPerSecond = 60
        context.coordinator.attach(to: v)
        return v
    }
    func updateUIView(_ uiView: MTKView, context: Context) {}
    func makeCoordinator() -> Coord { Coord(renderer: renderer, onTap: onTap) }

    final class Coord: NSObject, UIGestureRecognizerDelegate {
        let renderer: PointCloudRenderer
        let onTap: (CGPoint, CGSize) -> Void
        private var lastPan = CGPoint.zero
        init(renderer: PointCloudRenderer, onTap: @escaping (CGPoint, CGSize) -> Void) {
            self.renderer = renderer; self.onTap = onTap
        }
        func attach(to v: MTKView) {
            let pan = UIPanGestureRecognizer(target: self, action: #selector(onPan(_:)))
            let pinch = UIPinchGestureRecognizer(target: self, action: #selector(onPinch(_:)))
            let tap = UITapGestureRecognizer(target: self, action: #selector(onTapG(_:)))
            for g in [pan, pinch, tap] as [UIGestureRecognizer] { g.delegate = self; v.addGestureRecognizer(g) }
        }
        func gestureRecognizer(_ g: UIGestureRecognizer,
                               shouldRecognizeSimultaneouslyWith o: UIGestureRecognizer) -> Bool { true }
        @objc func onPan(_ g: UIPanGestureRecognizer) {
            let t = g.translation(in: g.view); g.setTranslation(.zero, in: g.view)
            renderer.camera.yaw -= Float(t.x) * 0.005
            renderer.camera.pitch = max(-1.5, min(1.5, renderer.camera.pitch + Float(t.y) * 0.005))
        }
        @objc func onPinch(_ g: UIPinchGestureRecognizer) {
            renderer.camera.distance = max(0.2, min(40, renderer.camera.distance / Float(g.scale)))
            g.scale = 1
        }
        @objc func onTapG(_ g: UITapGestureRecognizer) {
            guard let v = g.view else { return }
            onTap(g.location(in: v), v.bounds.size)
        }
    }
}
