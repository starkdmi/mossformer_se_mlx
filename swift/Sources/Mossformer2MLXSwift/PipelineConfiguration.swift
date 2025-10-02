import Foundation
import MLX
import AudioUtils

/// Configuration for the MossFormer2 SE 48K pipeline
public struct PipelineConfiguration {
    // Audio parameters matching Python args
    public let sampleRate: Int = 48000
    public let winLen: Int = 1920
    public let winInc: Int = 384
    public let fftLen: Int = 1920
    public let numMels: Int = 60
    public let winType: String = "hamming"
    
    // Segmentation parameters for long audio
    public let oneTimeDecodeLength: Double = 20.0  // 20 seconds threshold
    public let decodeWindow: Double = 4.0  // 4 second window for segmented processing
    
    // Optimization flags
    public let enableFloat16: Bool
    public let enableFastLayerNorm: Bool
    
    // Logging and debugging
    public let enableVerboseLogging: Bool
    public let enableProfiling: Bool
    
    // Audio processing
    public enum NormalizationMode {
        case automatic
        case fixedScale(Float)
        case disabled
    }
    
    public enum ResamplingQuality {
        case high
        case balanced
        case fast
    }
    
    public let normalizationMode: NormalizationMode
    public let resamplingQuality: ResamplingQuality
    
    public init(
        enableFloat16: Bool = true,
        enableFastLayerNorm: Bool = true,
        enableVerboseLogging: Bool = false,
        enableProfiling: Bool = false,
        normalizationMode: NormalizationMode = .automatic,
        resamplingQuality: ResamplingQuality = .balanced
    ) {
        self.enableFloat16 = enableFloat16
        self.enableFastLayerNorm = enableFastLayerNorm
        self.enableVerboseLogging = enableVerboseLogging
        self.enableProfiling = enableProfiling
        self.normalizationMode = normalizationMode
        self.resamplingQuality = resamplingQuality
    }
    
    /// Create a namespace-like object similar to Python args
    public var args: Args {
        Args(
            samplingRate: sampleRate,
            winLen: winLen,
            winInc: winInc,
            fftLen: fftLen,
            numMels: numMels,
            winType: winType,
            oneTimeDecodeLength: oneTimeDecodeLength,
            decodeWindow: decodeWindow
        )
    }
    
    /// Namespace struct to match Python args
    public struct Args {
        public let samplingRate: Int
        public let winLen: Int
        public let winInc: Int
        public let fftLen: Int
        public let numMels: Int
        public let winType: String
        public let oneTimeDecodeLength: Double
        public let decodeWindow: Double
        
        /// Convert to AudioUtils Fbank.Args
        public var fbankArgs: Fbank.Args {
            return Fbank.Args(
                samplingRate: samplingRate,
                winLen: winLen,
                winInc: winInc,
                numMels: numMels,
                winType: winType
            )
        }
    }
}

// MARK: - Precision Management

/// Manages Float16 precision settings across the pipeline
public class PrecisionManager {
    public static let shared = PrecisionManager()
    
    private init() {}
    
    /// Check if Float16 is available on the current device
    public func isFloat16Available() -> Bool {
        // Check for Metal Performance Shaders support
        // Float16 is typically available on A11 Bionic and later (iPhone 8/X and newer)
        #if os(iOS)
        if #available(iOS 11.0, *) {
            return true
        }
        return false
        #elseif os(macOS)
        // M1 and later Macs support Float16
        return true
        #else
        return false
        #endif
    }
}

// MARK: - Performance Profiling

/// Simple performance profiler for tracking pipeline stages
public class PerformanceProfiler {
    public static let shared = PerformanceProfiler()
    
    private var events: [ProfileEvent] = []
    private let queue = DispatchQueue(label: "com.mossformer2.profiler")
    
    private init() {}
    
    public enum EventType: String {
        case audioLoading = "Audio Loading"
        case featureExtraction = "Feature Extraction"
        case modelInference = "Model Inference"
        case audioReconstruction = "Audio Reconstruction"
        case audioSaving = "Audio Saving"
    }
    
    public struct ProfileEvent {
        let type: EventType
        let startTime: Date
        var endTime: Date?
        var metadata: [String: Any]
    }
    
    public func startEvent(_ type: EventType, metadata: [String: Any] = [:]) {
        queue.async {
            self.events.append(ProfileEvent(
                type: type,
                startTime: Date(),
                endTime: nil,
                metadata: metadata
            ))
        }
    }
    
    public func endEvent(_ type: EventType, metadata: [String: Any] = [:]) {
        queue.async {
            if let index = self.events.lastIndex(where: { $0.type == type && $0.endTime == nil }) {
                self.events[index].endTime = Date()
                // Merge additional metadata
                for (key, value) in metadata {
                    self.events[index].metadata[key] = value
                }
            }
        }
    }
    
    public func generateReport() -> String {
        queue.sync {
            var report = "Performance Profile Report\n"
            report += "========================\n\n"
            
            for event in events {
                report += "\(event.type.rawValue):\n"
                if let endTime = event.endTime {
                    let duration = endTime.timeIntervalSince(event.startTime)
                    report += "  Duration: \(String(format: "%.3f", duration))s\n"
                }
                for (key, value) in event.metadata {
                    report += "  \(key): \(value)\n"
                }
                report += "\n"
            }
            
            return report
        }
    }
    
    public func reset() {
        queue.async {
            self.events.removeAll()
        }
    }
}

// MARK: - Float16 Utilities

/// Utilities for Float16 optimization
public struct Float16Utils {
    /// Create an optimized zeros array
    public static func optimizedZeros(_ shape: [Int], referenceDType: DType? = nil) -> MLXArray {
        let dtype = (referenceDType == DType.float16) ? DType.float16 : DType.float32
        return MLXArray.zeros(shape, dtype: dtype)
    }
    
    /// Create padding array with appropriate dtype
    public static func createPadding(shape: [Int], referenceDType: DType) -> MLXArray {
        return MLXArray.zeros(shape, dtype: referenceDType)
    }
    
    /// Create intermediate tensor with memory-efficient dtype
    public static func createIntermediateTensor(shape: [Int], dtype: DType) -> MLXArray {
        return MLXArray.zeros(shape, dtype: dtype)
    }
    
    /// Optimize tensor for spectral processing
    public static func optimizeForSpectralProcessing(_ array: MLXArray) -> MLXArray {
        // For spectral data, Float16 can be used if available
        if PrecisionManager.shared.isFloat16Available() && array.dtype == .float32 {
            return array.asType(.float16)
        }
        return array
    }
}