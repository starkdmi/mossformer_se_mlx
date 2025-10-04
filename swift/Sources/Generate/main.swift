import Foundation
import ArgumentParser
import MLX
import MLXNN
import MLXRandom
import AudioUtils
import Mossformer2MLXSwift
import Hub

let sampleRate = 48000

func printSeparator() {
    print(String(repeating: "=", count: 60))
}

func performWarmup(pipeline: MossFormer2Pipeline, config: PipelineConfiguration) throws {
    // Create small random audio for component warmup
    let warmupSamples = 9600  // 0.2s at 48kHz
    let warmupAudio = MLXRandom.uniform(low: -0.1, high: 0.1, [warmupSamples])

    // Warm up STFT
    let window = try! createWindow(winType: "hamming", winLen: 1920, periodic: false)
    let (warmupReal, warmupImag) = stft(
        warmupAudio,
        nFFT: 1920,
        hopLength: 384,
        winLength: 1920,
        window: window,
        center: false
    )

    // Warm up Fbank computation
    _ = Fbank.computeFbank(warmupAudio * 32768.0, args: config.args.fbankArgs)

    // Warm up iSTFT
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

    // Warm up full pipeline
    let warmupPath = FileManager.default.temporaryDirectory
        .appendingPathComponent("warmup_audio.wav").path
    let audioSaver = AudioSaver(config: .init(sampleRate: Double(sampleRate)))
    try audioSaver.save(warmupAudio, to: warmupPath)

    _ = try pipeline.enhanceAudio(from: warmupPath)

    // Clean up
    try? FileManager.default.removeItem(atPath: warmupPath)
}

enum Precision: String, ExpressibleByArgument {
    case fp32
    case fp16
}

struct Generate: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "generate",
        abstract: "MossFormer2 Speech Enhancement - Extract speech from noisy audio",
        discussion: """
        Enhances audio by removing noise while preserving speech quality.
        Automatically downloads the model from HuggingFace on first run.

        Output is saved as 'enhanced_<input-name>.wav' in the same directory as the input file.
        """
    )

    @Argument(help: "Path to input audio file")
    var inputFile: String

    @Option(name: .shortAndLong, help: "Model precision (fp32 or fp16)")
    var precision: Precision = .fp32

    func validate() throws {
        guard FileManager.default.fileExists(atPath: inputFile) else {
            throw ValidationError("File not found: \(inputFile)")
        }
    }

    func run() throws {
        printSeparator()
        print("MossFormer2 Speech Enhancement - MLX Swift")
        printSeparator()

        // Download and load model
        print("Loading MossFormer2 SE 48K model (\(precision.rawValue))...")
        let weightsURL = try ModelDownloader(precision: precision.rawValue).downloadModelSync()

        let config = PipelineConfiguration()
        let pipeline = MossFormer2Pipeline(configuration: config)

        let loadStart = Date()
        try pipeline.loadWeights(from: weightsURL.path)
        let loadTime = Date().timeIntervalSince(loadStart)
        print("Model loading time: \(String(format: "%.2f", loadTime))s")

        // Load and analyze input audio
        print("\nLoading audio...")
        let audioLoader = AudioLoader(config: .init(
            targetSampleRate: Double(sampleRate),
            enableFloat16: precision == .fp16
        ))
        let audio = try audioLoader.load(from: inputFile)

        let inputLength = audio.shape[audio.ndim - 1]
        let duration = Float(inputLength) / Float(sampleRate)

        print("  Input: \(inputFile)")
        print("  Sample rate: \(sampleRate) Hz, Duration: \(String(format: "%.2f", duration))s")

        // Warmup (optional)
        print("\nWarming up model...")
        let warmupStart = Date()
        try performWarmup(pipeline: pipeline, config: config)
        let warmupTime = Date().timeIntervalSince(warmupStart)
        print("Warmup complete: \(String(format: "%.2f", warmupTime))s")

        // Process audio
        print("\nProcessing audio...")
        let audioURL = URL(fileURLWithPath: inputFile)
        let outputPath = audioURL.deletingLastPathComponent()
            .appendingPathComponent("enhanced_\(audioURL.deletingPathExtension().lastPathComponent).wav").path

        let processStart = Date()
        let enhanced = try pipeline.enhanceAudio(from: inputFile, outputPath: outputPath)
        eval(enhanced) // Force evaluation for accurate timing
        let processTime = Date().timeIntervalSince(processStart)

        // Display results
        printSeparator()
        print("Processing complete!")
        printSeparator()

        let realTimeFactor = Double(duration) / processTime
        print("Audio duration: \(String(format: "%.2f", duration))s")
        print("Processing time: \(String(format: "%.2f", processTime))s")
        print("Real-time factor: \(String(format: "%.2f", realTimeFactor))x")

        print("Output saved to: \(outputPath)")
        printSeparator()
    }
}

// Run the CLI
Generate.main()
