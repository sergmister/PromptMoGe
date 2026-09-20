import Foundation
import UIKit

/// Writes every capture to Documents/captures/<timestamp>/ so it can be pulled off the device later
/// (Files app, Finder, or `ios/pull_captures.sh`) and read with `ios/read_capture.py`.
///
///   rgb.jpg                       the camera frame at full resolution
///   lidar_depth.bin               float32 [dh, dw] metres, 0 = no return — the raw sensor, before any hole filling
///   lidar_conf.bin                uint8   [dh, dw] {0, 1, 2}
///   <model>_depth_k<K>.bin        float32 [H, W] metric depth after K refinement steps, unmasked
///   <model>_mask.bin              float32 [H, W] validity logit (> 0 = valid)
///   <model>_rays.bin              float32 [2, H, W] x/z and y/z: point = (rays * depth, depth)
///   meta.json                     shapes, camera intrinsics, per-result metric fit and latency
struct CaptureStore {
    struct ModelResult: Encodable { let steps: Int; let file: String; let scale: Float; let shift: Float; let ms: Double }
    struct Model: Encodable { let width: Int; let height: Int; let filledLidarPixels: Int; let stagesMs: [String: Double]; var results: [ModelResult] }
    struct Meta: Encodable {
        let date: String, device: String, system: String
        let image: [Int], lidar: [Int]              // [height, width]
        let intrinsics: [[Float]]                   // 3x3, row-major, for the full-resolution image
        var models: [String: Model] = [:]
    }

    static let root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("captures")
    let dir: URL
    private(set) var meta: Meta

    init?(frame f: CapturedFrame) {
        let stamp = DateFormatter(); stamp.dateFormat = "yyyyMMdd_HHmmss_SSS"; stamp.locale = Locale(identifier: "en_US_POSIX")
        let now = Date()
        dir = Self.root.appendingPathComponent(stamp.string(from: now))
        guard (try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)) != nil else { return nil }
        let k = f.intrinsics
        meta = Meta(date: ISO8601DateFormatter().string(from: now), device: Self.machine(),
                    system: UIDevice.current.systemName + " " + UIDevice.current.systemVersion,
                    image: [f.rgbH, f.rgbW], lidar: [f.depthH, f.depthW],
                    intrinsics: (0..<3).map { r in (0..<3).map { c in k[c][r] } })
        write(f.depth, "lidar_depth.bin"); write(f.conf, "lidar_conf.bin")
        if let jpeg = Self.jpeg(rgba: f.rgba, width: f.rgbW, height: f.rgbH) { try? jpeg.write(to: dir.appendingPathComponent("rgb.jpg")) }
    }

    /// One model's maps: call once per pipeline, after its run. `results` are (steps, log-depth, scale, shift, seconds).
    mutating func add(model name: String, pipe: DepthPipeline, results: [(Int, [Float], Float, Float, Double)]) {
        let n = pipe.N
        var model = Model(width: pipe.W, height: pipe.H, filledLidarPixels: pipe.filledPixels,
                          stagesMs: Dictionary(pipe.stages, uniquingKeysWith: { a, _ in a }), results: [])
        for (steps, logz, s, t, seconds) in results {
            let file = "\(name)_depth_k\(steps).bin"
            write(logz.map { s * exp($0) + t }, file)
            model.results.append(ModelResult(steps: steps, file: file, scale: s, shift: t, ms: seconds * 1000))
        }
        write(UnsafeBufferPointer(start: pipe.maskLogit.contents().bindMemory(to: Float16.self, capacity: n), count: n).map { Float($0) }, "\(name)_mask.bin")
        write(Array(UnsafeBufferPointer(start: pipe.refiner.xy.contents().bindMemory(to: Float.self, capacity: 2 * n), count: 2 * n)), "\(name)_rays.bin")
        meta.models[name] = model
    }

    func finish() {
        let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? enc.encode(meta).write(to: dir.appendingPathComponent("meta.json"))
    }

    private func write<T>(_ values: [T], _ name: String) {
        values.withUnsafeBytes { try? Data($0).write(to: dir.appendingPathComponent(name)) }
    }

    private static func jpeg(rgba: [UInt8], width: Int, height: Int) -> Data? {
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                                  space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { return nil }
        return UIImage(cgImage: image).jpegData(compressionQuality: 0.95)
    }

    private static func machine() -> String {
        var info = utsname(); uname(&info)
        return withUnsafeBytes(of: &info.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
    }
}
