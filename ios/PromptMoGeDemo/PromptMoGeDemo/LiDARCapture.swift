import Foundation
import AVFoundation
import CoreImage
import CoreVideo
import SwiftUI
import simd

/// Appends to Documents/log.txt (readable through the Files app or `devicectl`) and prints.
enum Log {
    static let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("log.txt")
    static func write(_ s: String) {
        let line = s + "\n"
        if let h = try? FileHandle(forWritingTo: url) {
            h.seekToEndOfFile(); try? h.write(contentsOf: Data(line.utf8)); try? h.close()
        } else { try? line.write(to: url, atomically: true, encoding: .utf8) }
        print(s)
    }
}

/// One synchronised RGB + LiDAR capture, in plain buffers.
struct CapturedFrame {
    var rgbW: Int, rgbH: Int
    var rgba: [UInt8]                  // rgbW*rgbH*4
    var depthW: Int, depthH: Int
    var depth: [Float]                 // metres, 0 = invalid
    var conf: [UInt8]                  // {0,1,2}
    var intrinsics: simd_float3x3      // for the rgbW x rgbH image
}

/// The LiDAR through AVFoundation, without world tracking.
///
/// ARKit's `sceneDepth` only arrives once world tracking has initialised, which needs device motion;
/// `.builtInLiDARDepthCamera` has no such dependency: it is the sensor, streaming, with no pose estimation in the
/// way. `AVDepthData` exposes no per-pixel confidence, while the models were trained on ARKit's {0, 1, 2}: valid
/// pixels are therefore marked confident and invalid ones missing. The raw return has holes that ARKit's
/// `sceneDepth` does not, which is why the pipeline nearest-fills the prompt (`PromptBuilder.nearestFill`).
final class LiDARCapture: NSObject, ObservableObject, AVCaptureDataOutputSynchronizerDelegate {
    let session = AVCaptureSession()
    @Published var depthReady = false
    @Published var hasLiDAR = true
    @Published var note = ""

    private let videoOut = AVCaptureVideoDataOutput()
    private let depthOut = AVCaptureDepthDataOutput()
    private var sync: AVCaptureDataOutputSynchronizer?
    private let queue = DispatchQueue(label: "lidar.capture")
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    /// RGB and depth are only ever stored TOGETHER, from the same synchronized collection.
    /// Keeping them in two variables would let `capture()` pair buffers from different
    /// instants whenever one output dropped a frame -- a moving camera would then reconstruct
    /// a cloud whose colour and geometry disagree, which is exactly the failure a synchronizer
    /// exists to prevent.
    private struct Pair {
        let video: CVPixelBuffer
        let depth: AVDepthData
        let vTime: CMTime
        let dTime: CMTime
        var skewMs: Double { abs((vTime - dTime).seconds) * 1000 }
    }
    private var lastPair: Pair?
    private let pairLock = NSLock()
    private var frames = 0
    /// Timestamp difference of the most recent accepted pair, surfaced so sync is observable.
    @Published var skewMs: Double = 0

    func start() {
        queue.async { [weak self] in self?.configure() }
    }
    func pause() { queue.async { [weak self] in self?.session.stopRunning() } }

    private func configure() {
        guard session.inputs.isEmpty else {
            if !session.isRunning { session.startRunning() }
            return
        }
        session.beginConfiguration()
        session.sessionPreset = .photo
        guard let dev = AVCaptureDevice.default(.builtInLiDARDepthCamera, for: .video, position: .back)
                ?? AVCaptureDevice.default(.builtInDualCamera, for: .video, position: .back)
                ?? AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: dev) else {
            DispatchQueue.main.async { self.hasLiDAR = false; self.note = "no camera" }
            session.commitConfiguration(); return
        }
        let isLiDAR = dev.deviceType == .builtInLiDARDepthCamera
        DispatchQueue.main.async { self.hasLiDAR = isLiDAR }
        if session.canAddInput(input) { session.addInput(input) }

        videoOut.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:
                                    kCVPixelFormatType_32BGRA]
        videoOut.alwaysDiscardsLateVideoFrames = true
        if session.canAddOutput(videoOut) { session.addOutput(videoOut) }

        depthOut.isFilteringEnabled = false          // the raw sensor return, as trained on
        if session.canAddOutput(depthOut) { session.addOutput(depthOut) }
        if let c = depthOut.connection(with: .depthData) { c.isEnabled = true }

        // Pick the depth format with the most pixels: the prompt is resampled anyway, and a finer sensor grid
        // is strictly more information.
        if let fmt = dev.activeFormat.supportedDepthDataFormats.filter({
            CMFormatDescriptionGetMediaSubType($0.formatDescription) == kCVPixelFormatType_DepthFloat32
        }).max(by: { a, b in
            let da = CMVideoFormatDescriptionGetDimensions(a.formatDescription)
            let db = CMVideoFormatDescriptionGetDimensions(b.formatDescription)
            return Int(da.width) * Int(da.height) < Int(db.width) * Int(db.height)
        }) {
            try? dev.lockForConfiguration()
            dev.activeDepthDataFormat = fmt
            // Pair the cadences. Left alone, video and depth free-run at different rates and
            // the synchroniser can only pair the nearest pair it has -- measured at 48.6 ms of
            // skew, which is centimetres of parallax in a moving hand. Pinning both to the same
            // frame duration brings the pairing inside one frame interval.
            let fps = CMTime(value: 1, timescale: 30)
            dev.activeVideoMinFrameDuration = fps
            dev.activeVideoMaxFrameDuration = fps
            dev.activeDepthDataMinFrameDuration = fps
            dev.unlockForConfiguration()
            let d = CMVideoFormatDescriptionGetDimensions(fmt.formatDescription)
            Log.write("LiDAR depth format \(d.width)x\(d.height)")
        }
        session.commitConfiguration()

        sync = AVCaptureDataOutputSynchronizer(dataOutputs: [videoOut, depthOut])
        sync?.setDelegate(self, queue: queue)
        session.startRunning()
        Log.write("AVCapture running, device=\(dev.localizedName) lidar=\(isLiDAR)")
    }

    func dataOutputSynchronizer(_ s: AVCaptureDataOutputSynchronizer,
                                didOutput collection: AVCaptureSynchronizedDataCollection) {
        frames += 1
        guard let v = collection.synchronizedData(for: videoOut) as? AVCaptureSynchronizedSampleBufferData,
              let d = collection.synchronizedData(for: depthOut) as? AVCaptureSynchronizedDepthData,
              !v.sampleBufferWasDropped, !d.depthDataWasDropped,
              let pb = CMSampleBufferGetImageBuffer(v.sampleBuffer) else {
            return                                   // partial collection: keep the last good pair
        }
        let pair = Pair(video: pb, depth: d.depthData,
                        vTime: v.timestamp, dTime: d.timestamp)
        pairLock.lock(); lastPair = pair; pairLock.unlock()
        let skew = pair.skewMs
        if !depthReady {
            let m = d.depthData.depthDataMap
            Log.write(String(format: "synced pair: depth %dx%d, skew %.3f ms",
                            CVPixelBufferGetWidth(m), CVPixelBufferGetHeight(m), skew))
            DispatchQueue.main.async { self.depthReady = true; self.skewMs = skew }
        } else if frames % 60 == 0 {
            DispatchQueue.main.async { self.skewMs = skew }
        }
    }

    func capture() -> CapturedFrame? {
        pairLock.lock(); let pair = lastPair; pairLock.unlock()
        guard let pair else { return nil }
        let v = pair.video, dd = pair.depth
        let w = CVPixelBufferGetWidth(v), h = CVPixelBufferGetHeight(v)
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let ci = CIImage(cvPixelBuffer: v)
        rgba.withUnsafeMutableBytes { buf in
            ciContext.render(ci, toBitmap: buf.baseAddress!, rowBytes: w * 4,
                             bounds: CGRect(x: 0, y: 0, width: w, height: h),
                             format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())
        }
        // metres, float32
        let conv = dd.depthDataType == kCVPixelFormatType_DepthFloat32
            ? dd : dd.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        let dm = conv.depthDataMap
        CVPixelBufferLockBaseAddress(dm, .readOnly)
        let dw = CVPixelBufferGetWidth(dm), dh = CVPixelBufferGetHeight(dm)
        let stride = CVPixelBufferGetBytesPerRow(dm) / MemoryLayout<Float>.size
        var depth = [Float](repeating: 0, count: dw * dh)
        var conf = [UInt8](repeating: 0, count: dw * dh)
        if let base = CVPixelBufferGetBaseAddress(dm)?.assumingMemoryBound(to: Float.self) {
            for y in 0..<dh {
                for x in 0..<dw {
                    let z = base[y * stride + x]
                    let ok = z.isFinite && z > 0
                    depth[y * dw + x] = ok ? z : 0
                    conf[y * dw + x] = ok ? 2 : 0
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(dm, .readOnly)

        var K = simd_float3x3(diagonal: SIMD3<Float>(1, 1, 1))
        if let cal = conv.cameraCalibrationData {
            let m = cal.intrinsicMatrix
            let ref = cal.intrinsicMatrixReferenceDimensions
            let sx = Float(w) / Float(ref.width), sy = Float(h) / Float(ref.height)
            K = simd_float3x3(SIMD3<Float>(m[0][0] * sx, 0, 0),
                              SIMD3<Float>(0, m[1][1] * sy, 0),
                              SIMD3<Float>(m[2][0] * sx, m[2][1] * sy, 1))
        } else {
            // fall back to a plausible pinhole rather than silently producing wrong metres
            let f = Float(w) * 0.83
            K = simd_float3x3(SIMD3<Float>(f, 0, 0), SIMD3<Float>(0, f, 0),
                              SIMD3<Float>(Float(w) / 2, Float(h) / 2, 1))
            Log.write("WARNING: no camera calibration; using an assumed focal length")
        }
        return CapturedFrame(rgbW: w, rgbH: h, rgba: rgba, depthW: dw, depthH: dh,
                             depth: depth, conf: conf, intrinsics: K)
    }
}

/// Live preview for the AVFoundation session.
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView(); v.layer.session = session
        v.layer.videoGravity = .resizeAspectFill
        return v
    }
    func updateUIView(_ uiView: PreviewView, context: Context) {}
    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        // swiftlint:disable:next force_cast
        override var layer: AVCaptureVideoPreviewLayer { super.layer as! AVCaptureVideoPreviewLayer }

        /// The sensor delivers landscape-right regardless of how the iPad is held, so the
        /// preview has to be rotated to the interface or it appears turned on its side.
        override func layoutSubviews() {
            super.layoutSubviews()
            guard let c = layer.connection, c.isVideoRotationAngleSupported(0) else { return }
            let angle: CGFloat
            switch window?.windowScene?.effectiveGeometry.interfaceOrientation ?? .portrait {
            case .portrait:            angle = 90
            case .portraitUpsideDown:  angle = 270
            case .landscapeLeft:       angle = 180
            case .landscapeRight:      angle = 0
            default:                   angle = 90
            }
            if c.isVideoRotationAngleSupported(angle) { c.videoRotationAngle = angle }
        }
    }
}
