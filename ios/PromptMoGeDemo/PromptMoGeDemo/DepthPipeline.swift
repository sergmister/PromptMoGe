import Foundation
import CoreML
import Metal
import Accelerate

/// Model A or Model B end to end on the device: camera image + LiDAR depth and confidence in, metric depth out.
///
///   prompt build (CPU) -> prompt_stem (GPU) + prompt_neck (ANE, concurrent with the ViT)
///   -> stem (ANE) -> 24 x [ attention (GPU) -> MLP (ANE) ] -> head (ANE) -> neck + heads (ANE)
///   -> point map into Metal -> refiner x K (GPU) -> robust metric fit + depth (GPU)
///
/// The Neural Engine is the efficient place for everything dense except self-attention, where the GPU's fused
/// kernel wins; the sparse 3-D refiner has no Core ML form at all and runs on custom Metal kernels. Per frame the
/// host only builds the prompt and moves tensors: the Core ML input arrays are allocated once and written in place
/// (fp16, honouring Core ML's stride padding), the ViT's block-to-block handoffs go through IOSurface-backed output
/// backings so no activation is copied, and the Core ML outputs are converted straight into the refiner's shared
/// Metal buffers.
final class DepthPipeline {
    /// The frozen DINOv2 encoder, shared by every model.
    struct ViT { let stem: MLModel; let head: MLModel; let attention: [MLModel]; let mlp: [MLModel]; let attentionInjected: [Int: MLModel] }

    static let rows = 30, cols = 40                       // 1200 tokens at 4:3
    static let injectBlocks = [0, 4, 8], taps = [5: 0, 11: 1, 17: 2, 23: 3]

    static func load(_ url: URL, _ units: MLComputeUnits) async -> MLModel? {
        guard let compiled = try? await MLModel.compileModel(at: url) else { return nil }
        let cfg = MLModelConfiguration(); cfg.computeUnits = units
        return try? await MLModel.load(contentsOf: compiled, configuration: cfg)
    }

    static func loadViT(dir: URL) async -> ViT? {
        guard let stem = await load(dir.appendingPathComponent("stem.mlpackage"), .cpuAndNeuralEngine),
              let head = await load(dir.appendingPathComponent("head.mlpackage"), .cpuAndNeuralEngine) else { return nil }
        var attention: [MLModel] = [], mlp: [MLModel] = [], injected: [Int: MLModel] = [:]
        for k in injectBlocks {
            guard let m = await load(dir.appendingPathComponent("blk\(k)_attn_inj.mlpackage"), .cpuAndGPU) else { return nil }
            injected[k] = m
        }
        for k in 0..<24 {
            guard let a = await load(dir.appendingPathComponent("blk\(k)_attn.mlpackage"), .cpuAndGPU),
                  let r = await load(dir.appendingPathComponent("blk\(k)_rest.mlpackage"), .cpuAndNeuralEngine) else { return nil }
            attention.append(a); mlp.append(r)
        }
        return ViT(stem: stem, head: head, attention: attention, mlp: mlp, attentionInjected: injected)
    }

    let vit: ViT
    let promptStem: MLModel, promptNeck: MLModel, neckHeads: MLModel
    let refiner: Refiner
    let levels: Int, H: Int, W: Int, N: Int
    let imageArr: MLMultiArray
    private let promptStemArr: MLMultiArray, promptNeckArr: MLMultiArray
    private let imageIn: MLFeatureProvider, promptStemIn: MLFeatureProvider, promptNeckIn: MLFeatureProvider
    /// Mask logit (fp16) and metric depth (fp32, +inf where masked) at the point-map resolution.
    let maskLogit: MTLBuffer, depth: MTLBuffer
    private let postDepth: MTLComputePipelineState
    private(set) var stages: [(String, Double)] = []
    private(set) var metricScale: Float = 1, metricShift: Float = 0
    /// Nearest-fill missing sensor pixels before inference (see `PromptBuilder.nearestFill`).
    var fillPrompt = true
    private(set) var filledPixels = 0

    /// Mutable inputs, so a feature provider is not rebuilt per call.
    private final class Inputs: NSObject, MLFeatureProvider {
        var values: [String: MLFeatureValue] = [:]
        var featureNames: Set<String> { Set(values.keys) }
        func featureValue(for n: String) -> MLFeatureValue? { values[n] }
    }
    private let attentionIn = Inputs(), mlpIn = Inputs()
    private let attentionOut = MLPredictionOptions()
    private var streamOut: [(MLMultiArray, MLPredictionOptions)] = [], tapOut: [(MLMultiArray, MLPredictionOptions)] = []

    /// An IOSurface-backed fp16 [1, rows, cols] array: usable as a Core ML output backing with no copy.
    private static func surfaceArray(rows: Int, cols: Int) -> MLMultiArray? {
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary, kCVPixelBufferMetalCompatibilityKey: true]
        guard CVPixelBufferCreate(nil, cols, rows, kCVPixelFormatType_OneComponent16Half, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let pb else { return nil }
        return MLMultiArray(pixelBuffer: pb, shape: [1, NSNumber(value: rows), NSNumber(value: cols)])
    }

    /// `dir` holds the three Core ML models and the refiner of one model (`python -m promptmoge.export`).
    init?(vit: ViT, dir: URL) async {
        // prompt_stem on the GPU: its first convolution writes a 15 MB activation, which is slow on the engine.
        guard let ps = await Self.load(dir.appendingPathComponent("prompt_stem.mlpackage"), .cpuAndGPU),
              let pn = await Self.load(dir.appendingPathComponent("prompt_neck.mlpackage"), .cpuAndNeuralEngine),
              let nh = await Self.load(dir.appendingPathComponent("neck_heads.mlpackage"), .cpuAndNeuralEngine),
              let rf = Refiner(dir: dir),
              let f = rf.library.makeFunction(name: "post_depth"),
              let pd = try? await rf.device.makeComputePipelineState(function: f) else { return nil }
        self.vit = vit; promptStem = ps; promptNeck = pn; neckHeads = nh; refiner = rf; postDepth = pd
        levels = rf.L; H = rf.H; W = rf.W; N = H * W
        guard H == Self.rows << (levels - 1), W == Self.cols << (levels - 1) else { return nil }
        func array(_ shape: [Int]) -> MLMultiArray { try! MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16) }
        imageArr = array([1, 3, Self.rows * 14, Self.cols * 14])
        promptStemArr = array([1, 4, Self.rows * 14, Self.cols * 14])
        promptNeckArr = array([1, 4, H, W])
        imageIn = try! MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(multiArray: imageArr)])
        promptStemIn = try! MLDictionaryFeatureProvider(dictionary: ["p": MLFeatureValue(multiArray: promptStemArr)])
        promptNeckIn = try! MLDictionaryFeatureProvider(dictionary: ["p": MLFeatureValue(multiArray: promptNeckArr)])
        maskLogit = rf.device.makeBuffer(length: N * 2, options: .storageModeShared)!
        depth = rf.device.makeBuffer(length: N * 4, options: .storageModeShared)!
        // Attention writes one surface; the MLP half reads it and writes into two alternating surfaces, or into a
        // dedicated one at the four blocks the encoder head taps -- so no buffer is read while it is written.
        let S = Self.rows * Self.cols + 1
        guard let a = Self.surfaceArray(rows: S, cols: 1024) else { return nil }
        attentionOut.outputBackings = ["a": a]
        mlpIn.values = ["x": MLFeatureValue(multiArray: a)]
        for i in 0..<6 {
            guard let y = Self.surfaceArray(rows: S, cols: 1024) else { return nil }
            let o = MLPredictionOptions(); o.outputBackings = ["y": y]
            if i < 2 { streamOut.append((y, o)) } else { tapOut.append((y, o)) }
        }
    }

    // ---------------------------------------------------------------------------------------------- host copies
    /// fp32 planar [c, h, w] -> a rank-4 MLMultiArray, honouring its strides and dtype.
    static func write(_ a: MLMultiArray, planar src: UnsafePointer<Float>, c: Int, h: Int, w: Int) {
        let st = a.strides.map { $0.intValue }
        let fp16 = a.dataType == .float16
        a.withUnsafeMutableBytes { raw, _ in
            let base = raw.baseAddress!
            if st[3] == 1 && st[2] == w && st[1] == h * w {
                let n = c * h * w
                if fp16 {
                    var sb = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: vImagePixelCount(n), rowBytes: n * 4)
                    var db = vImage_Buffer(data: base, height: 1, width: vImagePixelCount(n), rowBytes: n * 2)
                    vImageConvert_PlanarFtoPlanar16F(&sb, &db, 0)
                } else { memcpy(base, src, n * 4) }
                return
            }
            for ch in 0..<c {
                for y in 0..<h {
                    let off = ch * st[1] + y * st[2]
                    if fp16 {
                        var sb = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src + ch * h * w + y * w), height: 1,
                                               width: vImagePixelCount(w), rowBytes: w * 4)
                        var db = vImage_Buffer(data: base + off * 2, height: 1, width: vImagePixelCount(w), rowBytes: w * 2)
                        vImageConvert_PlanarFtoPlanar16F(&sb, &db, 0)
                    } else { memcpy(base + off * 4, src + ch * h * w + y * w, w * 4) }
                }
            }
        }
    }

    /// Nearest resize of a planar [4, dh, dw] fp32 prompt straight into a rank-4 fp16 input array. The reference
    /// resizes with F.interpolate(mode="nearest"): source = floor(dst * in / out). A destination row whose source
    /// row is the previous one's is copied rather than recomputed -- 192 source rows fill 420 or 480.
    static func writeNearest(_ a: MLMultiArray, from src: UnsafePointer<Float>, dw: Int, dh: Int, ow: Int, oh: Int) {
        let st = a.strides.map { $0.intValue }
        precondition(a.dataType == .float16, "prompt inputs are fp16")
        var colMap = [Int](repeating: 0, count: ow)
        for x in 0..<ow { colMap[x] = min(dw - 1, (x * dw) / ow) }
        a.withUnsafeMutableBytes { raw, _ in
            let base = raw.baseAddress!.assumingMemoryBound(to: Float16.self)
            colMap.withUnsafeBufferPointer { cm in
                for ch in 0..<4 {
                    var prevSy = -1
                    var prevRow: UnsafeMutablePointer<Float16>? = nil
                    for y in 0..<oh {
                        let sy = min(dh - 1, (y * dh) / oh)
                        let row = base + ch * st[1] + y * st[2]
                        if sy == prevSy, let pr = prevRow {
                            row.update(from: pr, count: ow)
                        } else {
                            let s = src + ch * dh * dw + sy * dw
                            for x in 0..<ow { row[x] = Float16(s[cm[x]]) }
                        }
                        prevSy = sy; prevRow = row
                    }
                }
            }
        }
    }

    /// One plane of a rank-4 fp16 output array -> fp32 at `dst`, honouring strides.
    static func readPlane(_ a: MLMultiArray, channel c: Int, h: Int, w: Int, into dst: UnsafeMutablePointer<Float>) {
        let st = a.strides.map { $0.intValue }
        a.withUnsafeBytes { raw in
            let base = raw.baseAddress!
            if a.dataType == .float16 {
                if st[3] == 1 && st[2] == w {
                    var sb = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: base + c * st[1] * 2), height: 1,
                                           width: vImagePixelCount(h * w), rowBytes: h * w * 2)
                    var db = vImage_Buffer(data: dst, height: 1, width: vImagePixelCount(h * w), rowBytes: h * w * 4)
                    vImageConvert_Planar16FtoPlanarF(&sb, &db, 0)
                    return
                }
                for y in 0..<h {
                    var sb = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: base + (c * st[1] + y * st[2]) * 2), height: 1,
                                           width: vImagePixelCount(w), rowBytes: w * 2)
                    var db = vImage_Buffer(data: dst + y * w, height: 1, width: vImagePixelCount(w), rowBytes: w * 4)
                    vImageConvert_Planar16FtoPlanarF(&sb, &db, 0)
                }
            } else {
                let p = base.assumingMemoryBound(to: Float.self)
                for y in 0..<h { (dst + y * w).update(from: p + c * st[1] + y * st[2], count: w) }
            }
        }
    }

    /// One plane, fp16 -> fp16 bytes (the mask logit).
    static func copyPlane16(_ a: MLMultiArray, channel c: Int, h: Int, w: Int, into dst: UnsafeMutableRawPointer) {
        let st = a.strides.map { $0.intValue }
        a.withUnsafeBytes { raw in
            let base = raw.baseAddress!
            if a.dataType == .float16 {
                for y in 0..<h { memcpy(dst + y * w * 2, base + (c * st[1] + y * st[2]) * 2, w * 2) }
            } else {
                let p = base.assumingMemoryBound(to: Float.self), d = dst.assumingMemoryBound(to: Float16.self)
                for y in 0..<h { for x in 0..<w { d[y * w + x] = Float16(p[c * st[1] + y * st[2] + x]) } }
            }
        }
    }

    // ---------------------------------------------------------------------------------------------- camera image
    private var rgbaSmall = [UInt8](), plane8 = [UInt8](), planeF = [Float]()

    /// The camera's RGBA frame -> the stem's [1, 3, 420, 560] input, 0..1 (the stem normalises). High-quality
    /// (Lanczos) resample -- a plain bilinear 4x reduction would alias -- then per channel 8-bit -> float -> fp16
    /// straight into the input array. Scratch buffers are allocated on first use and reused.
    func prepareImage(rgba: UnsafePointer<UInt8>, width w: Int, height h: Int) {
        let ow = Self.cols * 14, oh = Self.rows * 14
        if rgbaSmall.count != ow * oh * 4 {
            rgbaSmall = [UInt8](repeating: 0, count: ow * oh * 4)
            plane8 = [UInt8](repeating: 0, count: ow * oh)
            planeF = [Float](repeating: 0, count: ow * oh)
        }
        var src = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: rgba), height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w * 4)
        rgbaSmall.withUnsafeMutableBytes { small in
            var dst = vImage_Buffer(data: small.baseAddress!, height: vImagePixelCount(oh), width: vImagePixelCount(ow), rowBytes: ow * 4)
            _ = vImageScale_ARGB8888(&src, &dst, nil, vImage_Flags(kvImageHighQualityResampling))
            let st = imageArr.strides.map { $0.intValue }
            imageArr.withUnsafeMutableBytes { raw, _ in
                let base = raw.baseAddress!
                plane8.withUnsafeMutableBytes { p8 in
                    planeF.withUnsafeMutableBytes { pf in
                        var b8 = vImage_Buffer(data: p8.baseAddress!, height: vImagePixelCount(oh), width: vImagePixelCount(ow), rowBytes: ow)
                        var bf = vImage_Buffer(data: pf.baseAddress!, height: vImagePixelCount(oh), width: vImagePixelCount(ow), rowBytes: ow * 4)
                        for c in 0..<3 {
                            _ = vImageExtractChannel_ARGB8888(&dst, &b8, c, 0)
                            _ = vImageConvert_Planar8toPlanarF(&b8, &bf, 1.0, 0.0, 0)
                            for y in 0..<oh {
                                var rowF = vImage_Buffer(data: pf.baseAddress! + y * ow * 4, height: 1, width: vImagePixelCount(ow), rowBytes: ow * 4)
                                if imageArr.dataType == .float16 {
                                    var row16 = vImage_Buffer(data: base + (c * st[1] + y * st[2]) * 2, height: 1, width: vImagePixelCount(ow), rowBytes: ow * 2)
                                    _ = vImageConvert_PlanarFtoPlanar16F(&rowF, &row16, 0)
                                } else {
                                    memcpy(base + (c * st[1] + y * st[2]) * 4, pf.baseAddress! + y * ow * 4, ow * 4)
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------------------------------------- run
    /// `depth` / `conf`: the sensor planes (dw x dh, metres and {0, 1, 2}). The image must already be in `imageArr`
    /// (`prepareImage`). `perStep(k)` is called before refinement step k, with the state of K = k still in place.
    /// Afterwards `refiner.structure.logz` holds the refined log-depth and `self.depth` metric depth.
    @discardableResult
    func run(depth sensorDepth: UnsafePointer<Float>, conf sensorConf: UnsafePointer<UInt8>, dw: Int, dh: Int,
             steps K: Int, perStep: ((Int) -> Void)? = nil) -> Bool {
        stages = []
        var mark = CFAbsoluteTimeGetCurrent()
        func lap(_ n: String) { let t = CFAbsoluteTimeGetCurrent(); stages.append((n, (t - mark) * 1000)); mark = t }
        let rows = Self.rows, cols = Self.cols

        // The fill is applied to the sensor planes themselves, so the prompt, the refiner's residual and both fits
        // all see the same input (filled pixels carry confidence 1, which keeps them out of the metric fit).
        var sDepth = Array(UnsafeBufferPointer(start: sensorDepth, count: dw * dh))
        var sConf = Array(UnsafeBufferPointer(start: sensorConf, count: dw * dh))
        filledPixels = 0
        if fillPrompt {
            let f = PromptBuilder.nearestFill(depth: sDepth, conf: sConf, w: dw, h: dh)
            sDepth = f.depth; sConf = f.conf; filledPixels = f.filled
        }
        let prompt = PromptBuilder.build(depth: sDepth, conf: sConf, w: dw, h: dh)
        prompt.withUnsafeBufferPointer { Self.writeNearest(promptStemArr, from: $0.baseAddress!, dw: dw, dh: dh, ow: cols * 14, oh: rows * 14) }
        lap("prompt build")
        // prompt_neck feeds only the neck, after the ViT, and the engine idles while the GPU runs attention -- so it
        // runs concurrently with the ViT instead of before it.
        var neckInjections: MLFeatureProvider? = nil
        let neckDone = DispatchSemaphore(value: 0)
        DispatchQueue.global(qos: .userInitiated).async { [promptNeckArr, promptNeck, promptNeckIn, W, H] in
            prompt.withUnsafeBufferPointer { Self.writeNearest(promptNeckArr, from: $0.baseAddress!, dw: dw, dh: dh, ow: W, oh: H) }
            neckInjections = try? promptNeck.prediction(from: promptNeckIn)
            neckDone.signal()
        }
        guard let tokenInjections = try? promptStem.prediction(from: promptStemIn) else { neckDone.wait(); return false }
        lap("prompt stem")

        guard let s0 = try? vit.stem.prediction(from: imageIn), var x = s0.featureValue(for: "x")?.multiArrayValue else { neckDone.wait(); return false }
        var taps: [MLMultiArray] = []
        var flip = 0
        for k in 0..<24 {
            attentionIn.values = ["x": MLFeatureValue(multiArray: x)]
            var attention = vit.attention[k]
            if let injected = vit.attentionInjected[k], let inj = tokenInjections.featureValue(for: "i\(k)")?.multiArrayValue {
                attentionIn.values["inj"] = MLFeatureValue(multiArray: inj); attention = injected
            }
            let out: (MLMultiArray, MLPredictionOptions)
            if let t = Self.taps[k] { out = tapOut[t] } else { out = streamOut[flip]; flip ^= 1 }
            guard (try? attention.prediction(from: attentionIn, options: attentionOut)) != nil,
                  (try? vit.mlp[k].prediction(from: mlpIn, options: out.1)) != nil else { neckDone.wait(); return false }
            x = out.0
            if Self.taps[k] != nil { taps.append(x) }
        }
        lap("vit")
        guard let headIn = try? MLDictionaryFeatureProvider(dictionary: Dictionary(uniqueKeysWithValues:
                  taps.enumerated().map { ("t\($0.offset)", MLFeatureValue(multiArray: $0.element)) })),
              let enc = (try? vit.head.prediction(from: headIn))?.featureValue(for: "enc")?.multiArrayValue else { neckDone.wait(); return false }
        lap("head")
        neckDone.wait()
        guard let injections = neckInjections else { return false }
        lap("wait prompt neck")
        var neckIn: [String: MLFeatureValue] = ["enc": MLFeatureValue(multiArray: enc)]
        for i in 0..<levels {
            guard let v = injections.featureValue(for: "n\(i)") else { return false }
            neckIn["n\(i)"] = v
        }
        guard let nf = try? MLDictionaryFeatureProvider(dictionary: neckIn), let out = try? neckHeads.prediction(from: nf),
              let coord = out.featureValue(for: "coord")?.multiArrayValue,
              let mask = out.featureValue(for: "mask")?.multiArrayValue else { return false }
        lap("neck + heads")

        // into Metal
        let xy = refiner.xy.contents().bindMemory(to: Float.self, capacity: 2 * N)
        Self.readPlane(coord, channel: 0, h: H, w: W, into: xy)
        Self.readPlane(coord, channel: 1, h: H, w: W, into: xy + N)
        Self.readPlane(coord, channel: 2, h: H, w: W, into: refiner.structure.logz.contents().bindMemory(to: Float.self, capacity: N))
        Self.copyPlane16(mask, channel: 0, h: H, w: W, into: maskLogit.contents())
        let T = rows * cols
        let feature = refiner.encoderFeature.contents().bindMemory(to: Float.self, capacity: 1024 * T)
        for c in 0..<1024 { Self.readPlane(enc, channel: c, h: rows, w: cols, into: feature + c * T) }
        sDepth.withUnsafeBufferPointer { _ = memcpy(refiner.lidar.contents(), $0.baseAddress!, dw * dh * 4) }
        sConf.withUnsafeBufferPointer { _ = memcpy(refiner.conf.contents(), $0.baseAddress!, dw * dh) }
        refiner.setSensor(dw: dw, dh: dh)
        lap("to Metal")

        refiner.setFrame()
        for k in 0..<K {
            perStep?(k)
            guard refiner.step() else { return false }
        }
        lap("refiner x\(K)")
        encodeFinal()
        lap("metric fit + depth")
        return true
    }

    /// The final metric fit on the current log-depth and the depth map it implies.
    func encodeFinal() {
        let cb = refiner.queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
        refiner.fit.encodeFinal(e, logz: refiner.structure.logz, maskLogit: maskLogit, lidar: refiner.lidar,
                                conf: refiner.conf, H: H, W: W, dh: refiner.dh, dw: refiner.dw)
        e.setComputePipelineState(postDepth)
        e.setBuffer(refiner.structure.logz, offset: 0, index: 0); e.setBuffer(maskLogit, offset: 0, index: 1)
        e.setBuffer(depth, offset: 0, index: 2); e.setBuffer(refiner.fit.state, offset: 0, index: 3)
        var n = UInt32(N)
        e.setBytes(&n, length: 4, index: 4)
        e.dispatchThreads(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        metricScale = refiner.fit.s; metricShift = refiner.fit.t
    }

    var logz: [Float] { Array(UnsafeBufferPointer(start: refiner.structure.logz.contents().bindMemory(to: Float.self, capacity: N), count: N)) }
}
