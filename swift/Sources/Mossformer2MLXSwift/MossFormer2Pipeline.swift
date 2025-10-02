import Foundation
import MLX
import MLXNN
import AudioUtils

/// Main pipeline for MossFormer2 SE 48K speech enhancement
public class MossFormer2Pipeline {
    // Constants
    internal static let MAX_WAV_VALUE: Float = 32768.0
    
    // Configuration
    public let configuration: PipelineConfiguration // internal
    
    // Model
    public var modelWrapper: MossFormer2SE48K
    public var model: TestNet
    
    // Audio utilities
    internal let audioLoader: AudioLoader
    internal let audioSaver: AudioSaver
    
    // Cache for window functions
    private var windowCache: [String: MLXArray] = [:]
    
    /// Initialize the pipeline
    /// - Parameter configuration: Pipeline configuration
    public init(configuration: PipelineConfiguration = PipelineConfiguration()) {
        self.configuration = configuration
        
        print("Using fast LayerNorm optimization for all LayerNorm operations")
        
        // Initialize model
        self.modelWrapper = MossFormer2SE48K(args: configuration.args)
        self.model = modelWrapper.model
        
        // Initialize audio utilities
        // Create AudioLoader configuration from pipeline configuration
        let normMode: AudioLoader.Configuration.NormalizationMode
        switch configuration.normalizationMode {
        case .automatic:
            normMode = .automatic
        case .fixedScale(let scale):
            normMode = .manual(scale: scale)
        case .disabled:
            normMode = .none
        }
        let sampleRate = Double(configuration.sampleRate)
        let audioConfig = AudioLoader.Configuration(
            targetSampleRate: sampleRate,
            enableFloat16: configuration.enableFloat16,
            normalizationMode: normMode
        )
        self.audioLoader = AudioLoader(config: audioConfig)
        self.audioSaver = AudioSaver(config: .init(sampleRate: sampleRate))
    }
    
    /// Initialize the pipeline with custom normalization
    public init(disableNormalization: Bool = false) {
        // Create config with disabled normalization to match Python behavior
        let config = PipelineConfiguration(
            normalizationMode: disableNormalization ? .disabled : .automatic
        )
        self.configuration = config
        
        print("Using fast LayerNorm optimization for all LayerNorm operations")
        
        // Initialize model
        self.modelWrapper = MossFormer2SE48K(args: config.args)
        self.model = modelWrapper.model
        
        // Initialize audio utilities
        // Create AudioLoader configuration from pipeline configuration
        let normMode: AudioLoader.Configuration.NormalizationMode
        switch config.normalizationMode {
        case .automatic:
            normMode = .automatic
        case .fixedScale(let scale):
            normMode = .manual(scale: scale)
        case .disabled:
            normMode = .none
        }
        let sampleRate = Double(configuration.sampleRate)
        let audioConfig = AudioLoader.Configuration(
            targetSampleRate: sampleRate,
            enableFloat16: config.enableFloat16,
            normalizationMode: normMode
        )
        self.audioLoader = AudioLoader(config: audioConfig)
        self.audioSaver = AudioSaver(config: .init(sampleRate: sampleRate))
    }
    
    /// Load model weights from NPZ file
    /// - Parameter weightsPath: Path to the weights file (mossformer2_full.npz)
    public func loadWeights(from weightsPath: String) throws {
        print("Loading weights from \(weightsPath)...")
        let loadStart = Date()
        
        // Load weights from MLX format (NPZ file)
        let weights = try MLX.loadArrays(url: URL(fileURLWithPath: weightsPath))
        
        // The weights have structure: model.mossformer.xxx
        // We need to apply them to model which contains mossformer
        // So we should keep the structure but remove just "model."
        var processedWeights: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasPrefix("model.") {
                // Remove "model." prefix since we're updating the model directly
                let newKey = String(key.dropFirst(6))
                processedWeights[newKey] = value
            } else {
                processedWeights[key] = value
            }
        }
        
        // Get model's current parameters to filter weights
        let modelParams = model.parameters()
        let flatModelParams = modelParams.flattened()
        
        // Filter processedWeights to only include keys that exist in the model
        var filteredWeights: [String: MLXArray] = [:]
        let modelParamKeys = Set(flatModelParams.map { $0.0 })
        for (key, value) in processedWeights {
            if modelParamKeys.contains(key) {
                // Handle shape mismatches for PReLU weights
                if key.contains("prelu.weight") && value.ndim == 0 {
                    // Convert scalar to shape [1] for PReLU
                    filteredWeights[key] = value.reshaped([1])
                } else {
                    filteredWeights[key] = value
                }
            }
        }
        
        // Apply only the weights that exist in the model
        let nestedParams = NestedDictionary<String, MLXArray>.unflattened(filteredWeights)
        
        try model.update(parameters: nestedParams, verify: .none)
                
        let loadEnd = Date()
        print("\nWeights loaded successfully!")
        
        // Calculate total parameters
        let totalParams = weights.values.reduce(0) { sum, array in
            sum + array.size
        }
        print("  Total parameters: \(totalParams)")
        print("  Weight loading time: \(String(format: "%.3f", loadEnd.timeIntervalSince(loadStart)))s")
    }
    
    /// Enhance audio file
    /// - Parameters:
    ///   - audioPath: Path to input audio file
    ///   - outputPath: Path for output audio file (optional)
    /// - Returns: Enhanced audio as MLXArray
    public func enhanceAudio(from audioPath: String, outputPath: String? = nil) throws -> MLXArray {
        let totalStart = Date()
        
        // Load audio
        let audio = try audioLoader.load(from: audioPath)
        
        // Decode audio through model
        let decodeStart = Date()
        let enhancedAudio = try decodeOneAudio(model: model, inputs: audio, args: configuration.args)
        eval(enhancedAudio) // MARK: eval for fair timing
        let decodeEnd = Date()
        
        print("Total decode time: \(String(format: "%.3f", decodeEnd.timeIntervalSince(decodeStart)))s")
        
        // Save if output path provided
        if let outputPath = outputPath {
            try audioSaver.save(enhancedAudio, to: outputPath)
            print("Enhanced audio saved to: \(outputPath)")
        }
        
        let totalEnd = Date()
        print("Total processing time: \(String(format: "%.3f", totalEnd.timeIntervalSince(totalStart)))s")
        
        return enhancedAudio
    }
    
    /// Pure MLX implementation of the audio decoding function
    private func decodeOneAudio(
        model: TestNet,
        inputs: MLXArray,
        args: PipelineConfiguration.Args
    ) throws -> MLXArray {
        // Extract the first element from the input tensor
        var inputArray = inputs
        if inputs.ndim == 2 {
            inputArray = inputs[0, 0...]
        }
        
        let inputLen = inputArray.shape[0]
        let originalLen = inputLen  // Store original length for trimming output
        
        // Clean any potential NaN values
        inputArray = MLX.where(MLX.isNaN(inputArray), MLXArray(0.0), inputArray)
        
        // NOTE: Audio scaling is done in processShortAudio/processLongAudio
        // to match Python implementation which scales just before fbank computation
        
        // Check if input length exceeds the defined threshold for online decoding
        if Double(inputLen) > Double(args.samplingRate) * args.oneTimeDecodeLength {  // 20 seconds
            // Process long audio in segments
            return try processLongAudio(
                model: model,
                audio: inputArray,
                originalLen: originalLen,
                args: args
            )
        } else {
            // Process the entire audio at once
            return try processShortAudio(
                model: model,
                audio: inputArray,
                args: args
            )
        }
    }
    
    /// Process short audio (< 20 seconds) in one pass
    internal func processShortAudio(
        model: TestNet,
        audio: MLXArray,
        args: PipelineConfiguration.Args
    ) throws -> MLXArray {
        // Add batch dimension for compute_fbank
        // let audioBatch = audio.reshaped([1, -1])
        
        // STFT should be applied to scaled audio to match Python
        let scaledAudio = audio * Self.MAX_WAV_VALUE
        
        // Scale audio by MAX_WAV_VALUE before computing fbank
        // let scaledAudioBatch = audioBatch * Self.MAX_WAV_VALUE
        let scaledAudioBatch = scaledAudio.reshaped([1, -1])
        
        var fbanks = Fbank.computeFbank(scaledAudioBatch, args: args.fbankArgs)
        
        // Compute deltas for filter banks
        let fbankTr = fbanks.transposed(1, 0)
        let fbankDelta = try Fbank.computeDeltas(fbankTr)
        let fbankDeltaDelta = try Fbank.computeDeltas(fbankDelta)
        
        // Transpose back and concatenate
        fbanks = MLX.concatenated([
            fbanks,
            fbankDelta.transposed(1, 0),
            fbankDeltaDelta.transposed(1, 0)
        ], axis: 1)
        
        // Add batch dimension and pass through model
        fbanks = fbanks.expandedDimensions(axis: 0)
        let outList = model(fbanks)
        var predMask = outList.last!  // Get the predicted mask
        
        // Remove batch dimension for STFT masking
        predMask = predMask[0]
        
        let window = createWindow(type: args.winType, length: args.winLen)
        let spectrum = stft(scaledAudio, 
                           nFFT: args.fftLen, 
                           hopLength: args.winInc, 
                           winLength: args.winLen, 
                           window: window,
                           center: false)  // Match Python default
        
        // spectrum is a tuple (real, imag) from STFT
        var (realPart, imagPart) = spectrum
        
        // Remove batch dimension if present to match Python
        if realPart.ndim == 3 && realPart.shape[0] == 1 {
            realPart = realPart[0]
            imagPart = imagPart[0]
        }
        
        // Transpose mask and apply to spectrum
        predMask = predMask.transposed(1, 0).expandedDimensions(axis: -1)
        
        // Trim mask to match STFT output time dimension if needed
        /*if predMask.shape[1] != realPart.shape[1] {
            if predMask.shape[1] > realPart.shape[1] {
                // Mask has more frames, trim it
                predMask = predMask[0..., 0..<realPart.shape[1], 0...]
            } else {
                // Spectrum has more frames, trim it
                realPart = realPart[0..., 0..<predMask.shape[1]]
                imagPart = imagPart[0..., 0..<predMask.shape[1]]
            }
        }*/
        
        // Remove the extra dimension from predMask if it exists
        if predMask.ndim == 3 && predMask.shape[2] == 1 {
            predMask = predMask.squeezed(axis: 2)
        }
        
        let maskedReal = realPart * predMask
        let maskedImag = imagPart * predMask
        // Add batch dimension back for istft
        let batchedReal = maskedReal.expandedDimensions(axis: 0)
        let batchedImag = maskedImag.expandedDimensions(axis: 0)
        let outputs = istft(
            real: batchedReal,
            imag: batchedImag,
            nFFT: args.fftLen,
            hopLength: args.winInc,
            winLength: args.winLen,
            window: window,
            center: false,  // Match Python default
            length: audio.shape[0]  // Restore exact length matching
        ).squeezed(axis: 0)  // Remove batch dimension after istft
        
        
        // Scale back down to match Python behavior
        return outputs / Self.MAX_WAV_VALUE
    }
    
    /// Process long audio (> 20 seconds) in segments
    internal func processLongAudio(
        model: TestNet,
        audio: MLXArray,
        originalLen: Int,
        args: PipelineConfiguration.Args
    ) throws -> MLXArray {
        let window = Int(Double(args.samplingRate) * args.decodeWindow)  // 4s window
        let stride = Int(Double(window) * 0.75)  // 3s stride (75% overlap)
        
        // Pad input if necessary
        var paddedAudio = audio
        let audioLen = audio.shape[0]
        
        if audioLen < window {
            let padding = window - audioLen
            paddedAudio = MLX.concatenated([audio, MLXArray.zeros([padding])], axis: 0)
        } else if audioLen < window + stride {
            let padding = window + stride - audioLen
            paddedAudio = MLX.concatenated([audio, MLXArray.zeros([padding])], axis: 0)
        } else {
            let remainder = (audioLen - window) % stride
            if remainder != 0 {
                let padding = stride - remainder
                paddedAudio = MLX.concatenated([audio, MLXArray.zeros([padding])], axis: 0)
            }
        }
        
        let t = paddedAudio.shape[0]
        var outputSegments: [MLXArray] = []
        var outputRanges: [(Int, Int)] = []
        let giveUpLength = (window - stride) / 2
        
        // Process audio in sliding window segments
        var idx = 0
        while idx + window <= t {
            // Select segment
            let audioSegment = paddedAudio[idx..<(idx + window)]
            
            // Process segment (similar to processShortAudio)
            let segmentOutput = try processShortAudio(
                model: model,
                audio: audioSegment,
                args: args
            )
            
            // Store the output segment and range to use
            if idx == 0 {
                outputSegments.append(segmentOutput[0..<(segmentOutput.shape[0] - giveUpLength)])
                outputRanges.append((idx, idx + window - giveUpLength))
            } else {
                let lastWindow = segmentOutput[(segmentOutput.shape[0] - window)...]
                outputSegments.append(lastWindow[giveUpLength..<(lastWindow.shape[0] - giveUpLength)])
                outputRanges.append((idx + giveUpLength, idx + window - giveUpLength))
            }
            
            idx += stride
        }
        
        // Reconstruct the full output from segments more efficiently
        // Pre-allocate the output array
        let outputsArray = MLXArray.zeros([t])
        
        // Use a single update operation if possible
        for (segment, (start, end)) in zip(outputSegments, outputRanges) {
            outputsArray[start..<end] = segment
        }
        
        // Trim output to original length
        return outputsArray[0..<originalLen]
    }
    
    /// Create window function with caching
    internal func createWindow(type: String, length: Int) -> MLXArray {
        let cacheKey = "\(type.lowercased())_\(length)"
        
        // Check cache first
        if let cachedWindow = windowCache[cacheKey] {
            return cachedWindow
        }

        // Create window
        let window: MLXArray
        switch type.lowercased() {
        case "hamming":
            let n = MLXArray(0..<length).asType(.float32)
            let denominator = MLXArray(Float(length - 1))
            let cosArg = 2.0 * MLXArray(Float.pi) * n / denominator
            window = MLXArray(0.54) - MLXArray(0.46) * MLX.cos(cosArg)
        case "hann", "hanning":
            window = createHannWindow(length, periodic: true)
        default:
            fatalError("Unsupported window type: \(type)")
        }

        // Cache and return
        windowCache[cacheKey] = window
        return window
    }
}
