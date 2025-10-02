import XCTest
import MLX
import MLXNN
import AVFoundation
import AudioUtils
@testable import Mossformer2MLXSwift

/// Demonstration tests showcasing MossFormer2 speech enhancement capabilities
///
/// These tests serve as both functional tests and usage examples, demonstrating:
/// - Single file audio enhancement with automatic model downloading
/// - Batch processing of multiple audio files
/// - Performance optimization techniques (warmup, caching)
/// - Real-time factor measurement
///
/// **Setup:**
/// Place a `test.wav` file in the Tests/Mossformer2MLXSwiftTests/ directory
///
/// **Usage:**
/// ```bash
/// swift test --filter testSingleFileEnhancement
/// swift test --filter testBatchProcessing
/// ```
final class MossFormer2Demos: XCTestCase {

    // MARK: - Configuration

    private static let sampleRate = 48000

    // MARK: - Demo 1: Single File Enhancement

    /// Demonstrates enhancing a single audio file with automatic model download from HuggingFace
    ///
    /// Requires: test.wav in Tests/Mossformer2MLXSwiftTests/ directory
    func testSingleFileEnhancement() throws {
        // Get test audio path from test directory
        let testDir = URL(fileURLWithPath: #file).deletingLastPathComponent()
        let audioPath = testDir.appendingPathComponent("test.wav").path

        guard FileManager.default.fileExists(atPath: audioPath) else {
            throw XCTSkip("test.wav not found at: \(audioPath)\nPlace test.wav in Tests/Mossformer2MLXSwiftTests/ directory")
        }

        printSeparator()
        print("MossFormer2 Speech Enhancement - MLX Swift")
        printSeparator()

        // 1. Download and load model
        print("Loading MossFormer2 SE 48K model...")
        let weightsURL = try ModelDownloader().downloadModelSync()

        let config = PipelineConfiguration()
        let pipeline = MossFormer2Pipeline(configuration: config)

        let loadStart = Date()
        try pipeline.loadWeights(from: weightsURL.path)
        let loadTime = Date().timeIntervalSince(loadStart)
        print("Model loading time: \(String(format: "%.2f", loadTime))s")

        // 2. Load and analyze input audio
        print("\nLoading audio...")
        let audioLoader = AudioLoader(config: .init(
            targetSampleRate: Double(Self.sampleRate),
            enableFloat16: config.enableFloat16
        ))
        let audio = try audioLoader.load(from: audioPath)

        let inputLength = audio.shape[audio.ndim - 1]
        let duration = Float(inputLength) / Float(Self.sampleRate)

        print("  Input: \(audioPath)")
        print("  Sample rate: \(Self.sampleRate) Hz, Duration: \(String(format: "%.2f", duration))s")

        // 3. Warmup
        print("\nWarming up model...")
        let warmupStart = Date()
        try performWarmup(pipeline: pipeline, config: config)
        let warmupTime = Date().timeIntervalSince(warmupStart)
        print("Warmup complete: \(String(format: "%.2f", warmupTime))s")

        // 4. Process audio
        print("\nProcessing audio...")
        let outputPath = testDir // FileManager.default.temporaryDirectory
            .appendingPathComponent("enhanced_demo.wav").path

        let processStart = Date()
        let enhanced = try pipeline.enhanceAudio(from: audioPath, outputPath: outputPath)
        eval(enhanced) // Force evaluation for accurate timing
        let processTime = Date().timeIntervalSince(processStart)

        // 5. Display results
        printSeparator()
        print("Processing complete!")
        printSeparator()

        let realTimeFactor = Double(duration) / processTime
        print("Audio duration: \(String(format: "%.2f", duration))s")
        print("Processing time: \(String(format: "%.2f", processTime))s")
        print("Real-time factor: \(String(format: "%.2f", realTimeFactor))x")

        // Verify output
        XCTAssertTrue(FileManager.default.fileExists(atPath: outputPath))
        let fileSize = try FileManager.default.attributesOfItem(atPath: outputPath)[.size] as? Int ?? 0
        print("Output saved to: \(outputPath)")
        print("Output size: \(fileSize) bytes")

        printSeparator()
    }

    // MARK: - Demo 2: Batch Processing

    /// Demonstrates batch processing of multiple audio files
    ///
    /// Requires: test.wav in Tests/Mossformer2MLXSwiftTests/ directory
    ///
    /// This demo processes the same test.wav file 3 times to demonstrate batch processing.
    /// - Loading multiple audio files
    /// - Batch enhancement with output paths
    /// - Performance comparison vs sequential processing
    func testBatchProcessing() async throws {
        // Get test audio path from test directory
        let testDir = URL(fileURLWithPath: #file).deletingLastPathComponent()
        let audioPath = testDir.appendingPathComponent("test.wav").path

        guard FileManager.default.fileExists(atPath: audioPath) else {
            throw XCTSkip("test.wav not found at: \(audioPath)\nPlace test.wav in Tests/Mossformer2MLXSwiftTests/ directory")
        }

        // Create batch by duplicating the same file
        let batchSize = 3
        let audioPaths = Array(repeating: audioPath, count: batchSize)

        printSeparator()
        print("MossFormer2 Batch Processing - MLX Swift")
        printSeparator()

        // 1. Download and load model
        print("Loading MossFormer2 SE 48K model...")
        let weightsURL = try ModelDownloader().downloadModelSync()

        let config = PipelineConfiguration()
        let pipeline = MossFormer2Pipeline(configuration: config)

        let loadStart = Date()
        try pipeline.loadWeights(from: weightsURL.path)
        let loadTime = Date().timeIntervalSince(loadStart)
        print("Model loading time: \(String(format: "%.2f", loadTime))s")

        // 2. Analyze batch
        print("\nBatch information:")
        print("  Number of files: \(audioPaths.count)")

        let audioLoader = AudioLoader(config: .init(
            targetSampleRate: Double(Self.sampleRate),
            enableFloat16: config.enableFloat16
        ))

        let audio = try audioLoader.load(from: audioPath)
        let length = audio.shape[audio.ndim - 1]
        let duration = Float(length) / Float(Self.sampleRate)
        let totalDuration = Float(batchSize) * duration

        for i in 0..<batchSize {
            print("  File \(i + 1): \(String(format: "%.2f", duration))s - \((audioPath as NSString).lastPathComponent)")
        }

        print("  Total duration: \(String(format: "%.2f", totalDuration))s")

        // 3. Warmup
        print("\nWarming up model...")
        let warmupStart = Date()
        try performWarmup(pipeline: pipeline, config: config)
        _ = try await pipeline.enhanceAudioBatch(from: [audioPaths[0]])
        let warmupTime = Date().timeIntervalSince(warmupStart)
        print("Warmup complete: \(String(format: "%.2f", warmupTime))s")

        // 4. Sequential processing (for comparison)
        print("\nSequential processing...")
        let seqStart = Date()
        for (i, path) in audioPaths.enumerated() {
            _ = try pipeline.enhanceAudio(from: path)
            print("  Processed file \(i + 1)/\(audioPaths.count)")
        }
        let seqTime = Date().timeIntervalSince(seqStart)

        print("Sequential time: \(String(format: "%.2f", seqTime))s")
        print("Average per file: \(String(format: "%.2f", seqTime / Double(audioPaths.count)))s")

        // 5. Batch processing
        print("\nBatch processing...")
        let outputPaths = (0..<batchSize).map { i in
            testDir // FileManager.default.temporaryDirectory
                .appendingPathComponent("enhanced_batch_\(i).wav").path
        }

        let batchStart = Date()
        let batchResults = try await pipeline.enhanceAudioBatch(from: audioPaths, outputPaths: outputPaths)
        let batchTime = Date().timeIntervalSince(batchStart)

        print("Batch time: \(String(format: "%.2f", batchTime))s")
        print("Average per file: \(String(format: "%.2f", batchTime / Double(audioPaths.count)))s")

        // 6. Results
        printSeparator()
        print("Batch processing complete!")
        printSeparator()

        print("Performance comparison:")
        print("  Sequential: \(String(format: "%.2f", seqTime))s")
        print("  Batch: \(String(format: "%.2f", batchTime))s")

        let speedup = seqTime / batchTime
        if speedup > 1.0 {
            print("  Speedup: \(String(format: "%.2f", speedup))x faster")
            let efficiency = ((seqTime - batchTime) / seqTime) * 100
            print("  Efficiency gain: \(String(format: "%.1f", efficiency))%")
        } else {
            print("  Note: Batch overhead present for small batches")
        }

        // Verify outputs
        print("\nOutput files:")
        XCTAssertEqual(batchResults.shape[0], audioPaths.count)

        for (i, outputPath) in outputPaths.enumerated() {
            XCTAssertTrue(FileManager.default.fileExists(atPath: outputPath))
            let fileSize = try FileManager.default.attributesOfItem(atPath: outputPath)[.size] as? Int ?? 0
            print("  \(i + 1). \(outputPath) (\(fileSize) bytes)")
        }

        printSeparator()
    }

    // MARK: - Helper Methods

    /// Perform comprehensive warmup of all pipeline components
    private func performWarmup(pipeline: MossFormer2Pipeline, config: PipelineConfiguration) throws {
        // Create small random audio for component warmup
        let warmupSamples = 9600  // 0.2s at 48kHz
        let warmupAudio = MLXRandom.uniform(low: -0.1, high: 0.1, [warmupSamples])

        // 1. Warm up STFT
        let window = try! createWindow(winType: "hamming", winLen: 1920, periodic: false)
        let (warmupReal, warmupImag) = stft(
            warmupAudio,
            nFFT: 1920,
            hopLength: 384,
            winLength: 1920,
            window: window,
            center: false
        )

        // 2. Warm up Fbank computation
        _ = Fbank.computeFbank(warmupAudio * 32768.0, args: config.args.fbankArgs)

        // 3. Warm up iSTFT
        _ = istft(
            real: warmupReal,
            imag: warmupImag,
            nFFT: 1920,
            hopLength: 384,
            winLength: 1920,
            window: window,
            center: false,
            length: warmupSamples
        )

        // 4. Warm up full pipeline
        let warmupPath = FileManager.default.temporaryDirectory
            .appendingPathComponent("warmup_audio.wav").path
        let audioSaver = AudioSaver(config: .init(sampleRate: Double(Self.sampleRate)))
        try audioSaver.save(warmupAudio, to: warmupPath)

        _ = try pipeline.enhanceAudio(from: warmupPath)

        // Clean up
        try? FileManager.default.removeItem(atPath: warmupPath)
    }

    /// Print separator line
    private func printSeparator() {
        print(String(repeating: "=", count: 60))
    }
}
