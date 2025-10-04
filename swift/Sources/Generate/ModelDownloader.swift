import Foundation
import Hub

/// Downloads and caches MossFormer2 model files from HuggingFace
class ModelDownloader {
    enum DownloadError: Error {
        case weightsNotFound(path: String)
        case downloadFailed

        var localizedDescription: String {
            switch self {
            case .weightsNotFound(let path):
                return "Weights file not found at \(path)"
            case .downloadFailed:
                return "Failed to download model"
            }
        }
    }

    private let modelId = "starkdmi/MossFormer2_SE_48K_MLX"
    private let weightsFilename: String

    init(precision: String = "fp32") {
        self.weightsFilename = "model_\(precision).safetensors"
    }

    /// Downloads model weights from HuggingFace
    /// - Returns: Path to weights file
    func downloadModel() async throws -> URL {
        let repo = Hub.Repo(id: modelId)

        print("Downloading model from HuggingFace: \(modelId)")

        // Download model weights file
        let modelDirectory = try await HubApi.shared.snapshot(
            from: repo,
            matching: [weightsFilename],
            progressHandler: { progress in
                let percent = Int(progress.fractionCompleted * 100)
                print("  Download progress: \(percent)%")
            }
        )

        print("Download completed to: \(modelDirectory.path)")

        let weightsURL = modelDirectory.appendingPathComponent(weightsFilename)

        // Verify file exists
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw DownloadError.weightsNotFound(path: weightsURL.path)
        }

        return weightsURL
    }

    /// Synchronous wrapper for downloading model
    /// - Returns: Path to weights file
    func downloadModelSync() throws -> URL {
        var result: URL?
        var error: Error?

        let group = DispatchGroup()
        group.enter()

        Task {
            do {
                result = try await downloadModel()
            } catch let err {
                error = err
            }
            group.leave()
        }

        group.wait()

        if let error = error {
            throw error
        }

        guard let result = result else {
            throw DownloadError.downloadFailed
        }

        return result
    }
}
