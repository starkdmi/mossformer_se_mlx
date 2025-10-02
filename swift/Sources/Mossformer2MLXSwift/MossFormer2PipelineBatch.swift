import Foundation
import AudioUtils
import MLX
import MLXNN

// MARK: - Batch Processing Extension for MossFormer2Pipeline

extension MossFormer2Pipeline {

    /// Enhance multiple audio files in a batch
    /// - Parameters:
    ///   - audioPaths: Array of paths to input audio files
    ///   - outputPaths: Array of paths for output audio files (optional)
    /// - Returns: Batch of enhanced audio as MLXArray
    public func enhanceAudioBatch(from audioPaths: [String], outputPaths: [String]? = nil) async throws -> MLXArray {
        let totalStart = Date()
        
        print("=== Batch Audio Enhancement ===")
        print("Processing \(audioPaths.count) audio files")
        
        // Load all audio files
        let loadStart = Date()
        let audioArrays = try await withThrowingTaskGroup(of: (Int, MLXArray).self) { group in
            for (index, path) in audioPaths.enumerated() {
                group.addTask {
                    let audio = try self.audioLoader.load(from: path)
                    return (index, audio)
                }
            }
            
            var results: [(Int, MLXArray)] = []
            for try await result in group {
                results.append(result)
            }
            return results.sorted { $0.0 < $1.0 }.map { $0.1 }
        }
        let loadTime = Date().timeIntervalSince(loadStart)
        print("Batch loading time: \(String(format: "%.3f", loadTime))s")
        
        // Remove extra dimensions from loaded audio
        var cleanedAudioArrays: [MLXArray] = []
        for audio in audioArrays {
            if audio.ndim > 1 {
                // Remove extra dimensions, e.g. [1, N] -> [N]
                cleanedAudioArrays.append(audio.squeezed())
            } else {
                cleanedAudioArrays.append(audio)
            }
        }
        
        // Find the maximum length for padding
        let maxLength = cleanedAudioArrays.map { $0.shape[0] }.max() ?? 0
        
        // Pad all audio to the same length and stack
        var paddedAudios: [MLXArray] = []
        var originalLengths: [Int] = []
        
        for audio in cleanedAudioArrays {
            originalLengths.append(audio.shape[0])
            if audio.shape[0] < maxLength {
                let padding = MLXArray.zeros([maxLength - audio.shape[0]])
                let padded = MLX.concatenated([audio, padding], axis: 0)
                paddedAudios.append(padded)
            } else {
                paddedAudios.append(audio)
            }
        }
        
        // Stack into batch
        let batchAudio = MLX.stacked(paddedAudios, axis: 0)
        
        // Decode batch through model
        let decodeStart = Date()
        let enhancedBatch = try decodeAudioBatch(
            model: model,
            inputs: batchAudio,
            originalLengths: originalLengths,
            args: configuration.args
        )
        eval(enhancedBatch) // TODO: For timings
        let decodeTime = Date().timeIntervalSince(decodeStart)
        
        print("Batch decode time: \(String(format: "%.3f", decodeTime))s")
        
        // Save if output paths provided
        if let outputPaths = outputPaths {
            let saveStart = Date()
            
            // Save directly to specified paths
            for (idx, path) in outputPaths.enumerated() {
                let enhanced = enhancedBatch[idx, 0..<originalLengths[idx]]
                try audioSaver.save(enhanced, to: path)
            }
            
            let saveTime = Date().timeIntervalSince(saveStart)
            
            print("Batch save time: \(String(format: "%.3f", saveTime))s")
            print("Saved \(outputPaths.count) enhanced audio files")
        }
        
        let totalTime = Date().timeIntervalSince(totalStart)
        print("Total batch processing time: \(String(format: "%.3f", totalTime))s")
        print("Average time per file: \(String(format: "%.3f", totalTime / Double(audioPaths.count)))s")
        
        return enhancedBatch
    }
    
    /// Decode multiple audio samples in batch
    private func decodeAudioBatch(
        model: TestNet,
        inputs: MLXArray,
        originalLengths: [Int],
        args: PipelineConfiguration.Args
    ) throws -> MLXArray {
        let batchSize = inputs.shape[0]
        
        // Clean any potential NaN values
        let cleanInputs = MLX.where(MLX.isNaN(inputs), MLXArray(0.0), inputs)
        
        // Check if any audio exceeds the threshold for long processing
        let threshold = Int(Double(args.samplingRate) * args.oneTimeDecodeLength)
        let needsLongProcessing = originalLengths.contains { $0 > threshold }
        
        if needsLongProcessing {
            // Process each audio separately (some might be long, some short)
            var results: [MLXArray] = []
            
            for i in 0..<batchSize {
                let audio = cleanInputs[i, 0..<originalLengths[i]]
                
                if originalLengths[i] > threshold {
                    let result = try processLongAudio(
                        model: model,
                        audio: audio,
                        originalLen: originalLengths[i],
                        args: args
                    )
                    results.append(result)
                } else {
                    let result = try processShortAudio(
                        model: model,
                        audio: audio,
                        args: args
                    )
                    results.append(result)
                }
            }
            
            // Pad results to max length and stack
            let maxResultLength = results.map { $0.shape[0] }.max() ?? 0
            var paddedResults: [MLXArray] = []
            
            for result in results {
                if result.shape[0] < maxResultLength {
                    let padding = MLXArray.zeros([maxResultLength - result.shape[0]])
                    paddedResults.append(MLX.concatenated([result, padding], axis: 0))
                } else {
                    paddedResults.append(result)
                }
            }
            
            return MLX.stacked(paddedResults, axis: 0)
        } else {
            // All audio is short, process as batch
            return try processShortAudioBatch(
                model: model,
                audioBatch: cleanInputs,
                originalLengths: originalLengths,
                args: args
            )
        }
    }
    
    /// Process batch of short audio (< 20 seconds) in one pass
    private func processShortAudioBatch(
        model: TestNet,
        audioBatch: MLXArray,
        originalLengths: [Int],
        args: PipelineConfiguration.Args
    ) throws -> MLXArray {
        let batchSize = audioBatch.shape[0]
        
        // Scale audio by MAX_WAV_VALUE before computing fbank
        let scaledAudioBatch = audioBatch * Self.MAX_WAV_VALUE
        
        // Compute fbank for batch using optimized batch processing
        let fbanks = Fbank.computeFbank(scaledAudioBatch, args: args.fbankArgs)
        
        // Check if fbanks is empty or has unexpected shape
        guard fbanks.ndim >= 2 && fbanks.size > 0 else {
            print("Warning: fbanks has unexpected shape: \(fbanks.shape) or is empty")
            return MLXArray.zeros([batchSize, 1])
        }
        
        // Handle batch deltas computation
        let batchFbanksWithDeltas: MLXArray
        
        if fbanks.ndim == 3 {
            // Batch processing path
            // Transpose for delta computation: [B, T, F] -> [B, F, T]
            let fbanksTr = fbanks.transposed(0, 2, 1)
            
            // Compute deltas using batch-optimized version
            let fbankDelta = try Fbank.computeDeltas(fbanksTr)
            let fbankDeltaDelta = try Fbank.computeDeltas(fbankDelta)
            
            // Transpose back: [B, F, T] -> [B, T, F]
            let deltaTransposed = fbankDelta.transposed(0, 2, 1)
            let deltaDeltaTransposed = fbankDeltaDelta.transposed(0, 2, 1)
            
            // Concatenate along feature dimension
            batchFbanksWithDeltas = MLX.concatenated([
                fbanks,
                deltaTransposed,
                deltaDeltaTransposed
            ], axis: 2)
        } else {
            // Single audio path (fallback)
            let fbankTr = fbanks.transposed(1, 0)
            let fbankDelta = try Fbank.computeDeltas(fbankTr)
            let fbankDeltaDelta = try Fbank.computeDeltas(fbankDelta)
            
            let singleResult = MLX.concatenated([
                fbanks,
                fbankDelta.transposed(1, 0),
                fbankDeltaDelta.transposed(1, 0)
            ], axis: 1)
            
            batchFbanksWithDeltas = singleResult.expandedDimensions(axis: 0)
        }
        
        // Pass through model (model already handles batch dimension)
        let outList = model(batchFbanksWithDeltas)
        let predMasks = outList.last!  // Get the predicted masks [B, T, F]
        
        // Process STFT for entire batch at once
        let window = createWindow(type: args.winType, length: args.winLen)
        
        // Scale entire batch
        let scaledBatch = audioBatch * Self.MAX_WAV_VALUE
        
        // Batch STFT
        let (batchReal, batchImag) = stft(
            scaledBatch,
            nFFT: args.fftLen,
            hopLength: args.winInc,
            winLength: args.winLen,
            window: window,
            center: false
        )
        
        // Apply masks to batch
        // predMasks shape: [B, T, F], need to transpose to [B, F, T]
        let batchMasks = predMasks.transposed(0, 2, 1)
        
        // Trim masks or spectra to match dimensions if needed
        let timeFrames = batchReal.shape[2]
        let maskFrames = batchMasks.shape[2]
        
        let alignedReal: MLXArray
        let alignedImag: MLXArray
        let alignedMasks: MLXArray
        
        if maskFrames != timeFrames {
            if maskFrames > timeFrames {
                alignedReal = batchReal
                alignedImag = batchImag
                alignedMasks = batchMasks[0..., 0..., 0..<timeFrames]
            } else {
                alignedReal = batchReal[0..., 0..., 0..<maskFrames]
                alignedImag = batchImag[0..., 0..., 0..<maskFrames]
                alignedMasks = batchMasks
            }
        } else {
            alignedReal = batchReal
            alignedImag = batchImag
            alignedMasks = batchMasks
        }
        
        // Apply masks
        let maskedReal = alignedReal * alignedMasks
        let maskedImag = alignedImag * alignedMasks
        
        // Batch iSTFT using the proven mlxOptimizedISTFT
        let maxAudioLength = originalLengths.max() ?? 0
        let batchOutput = istft(
            real: maskedReal,
            imag: maskedImag,
            nFFT: args.fftLen,
            hopLength: args.winInc,
            winLength: args.winLen,
            window: window,
            center: false,
            length: maxAudioLength
        )
        
        // Scale back down and extract individual results
        let scaledOutput = batchOutput / Self.MAX_WAV_VALUE
        
        var enhancedAudios: [MLXArray] = []
        for i in 0..<batchSize {
            // Extract and trim to original length
            let audio = scaledOutput[i, 0..<originalLengths[i]]
            enhancedAudios.append(audio)
        }
        
        // Pad and stack results
        let maxLength = enhancedAudios.map { $0.shape[0] }.max() ?? 0
        var paddedResults: [MLXArray] = []
        
        for audio in enhancedAudios {
            if audio.shape[0] < maxLength {
                let padding = MLXArray.zeros([maxLength - audio.shape[0]])
                paddedResults.append(MLX.concatenated([audio, padding], axis: 0))
            } else {
                paddedResults.append(audio)
            }
        }
        
        return MLX.stacked(paddedResults, axis: 0)
    }

    /// Process batch of audio arrays directly (without file I/O)
    public func enhanceAudioArrayBatch(_ audioArrays: [MLXArray]) throws -> [MLXArray] {
        // Find the maximum length for padding
        let maxLength = audioArrays.map { $0.shape[0] }.max() ?? 0
        let originalLengths = audioArrays.map { $0.shape[0] }
        
        // Pad all audio to the same length and stack
        var paddedAudios: [MLXArray] = []
        
        for audio in audioArrays {
            if audio.shape[0] < maxLength {
                let padding = MLXArray.zeros([maxLength - audio.shape[0]])
                let padded = MLX.concatenated([audio, padding], axis: 0)
                paddedAudios.append(padded)
            } else {
                paddedAudios.append(audio)
            }
        }
        
        // Stack into batch
        let batchAudio = MLX.stacked(paddedAudios, axis: 0)
        
        // Decode batch
        let enhancedBatch = try decodeAudioBatch(
            model: model,
            inputs: batchAudio,
            originalLengths: originalLengths,
            args: configuration.args
        )
        
        // Extract individual results with original lengths
        var results: [MLXArray] = []
        for i in 0..<audioArrays.count {
            results.append(enhancedBatch[i, 0..<originalLengths[i]])
        }
        
        return results
    }
}
