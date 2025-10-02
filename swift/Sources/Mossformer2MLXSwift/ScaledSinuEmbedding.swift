import Foundation
import MLX
import MLXNN

/// Optimized MLX implementation of ScaledSinuEmbedding.
///
/// ScaledSinuEmbedding provides sinusoidal positional encodings for inputs.
/// It generates position embeddings using sine and cosine functions with
/// different frequencies and applies learnable scaling.
///
/// Performance optimizations:
/// - Efficient broadcasting operations
/// - Optional caching for repeated sequence lengths
/// - Reduced memory allocations
public class ScaledSinuEmbedding: Module, UnaryLayer {
    public let dim: Int
    public let useCache: Bool
    public let maxCacheSize: Int
    
    public var scale: MLXArray
    public var inv_freq: MLXArray
    
    // Cache for computed embeddings
    private var embeddingCache: [Int: MLXArray] = [:]
    
    /// Initialize ScaledSinuEmbedding
    /// - Parameters:
    ///   - dim: Dimension of the positional embeddings
    ///   - useCache: Whether to cache embeddings for repeated sequence lengths
    ///   - maxCacheSize: Maximum number of different sequence lengths to cache
    public init(
        dim: Int,
        useCache: Bool = true,
        maxCacheSize: Int = 50
    ) {
        self.dim = dim
        self.useCache = useCache
        self.maxCacheSize = maxCacheSize
        
        // Initialize learnable scale parameter
        self.scale = MLXArray.ones([1])
        
        // Calculate inverse frequencies for sinusoidal embeddings
        // inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        let positions = MLXArray(0..<dim).asType(.float32)[.stride(by: 2)]
        self.inv_freq = 1.0 / MLX.pow(10000.0, positions / Float(dim))
        
        super.init()
    }
    
    /// Optimized forward pass for the ScaledSinuEmbedding layer
    /// - Parameter x: Input tensor of shape (batch_size, sequence_length, ...)
    /// - Returns: Positional encoding tensor of shape (sequence_length, dim)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Extract sequence length from input
        let seqLen = x.shape[1]
        
        // Check cache if enabled
        if useCache, let cached = embeddingCache[seqLen] {
            return cached * scale
        }
        
        // Create position indices efficiently
        let positions = MLXArray(0..<seqLen).asType(.float32)
        
        // Compute sinusoidal values using efficient broadcasting
        // Optimized: Direct broadcasting with expandedDimensions
        let sinusoids = positions.expandedDimensions(axis: 1) * inv_freq.expandedDimensions(axis: 0)
        
        // Compute sin and cos and concatenate in one operation
        // This reduces memory allocations compared to separate operations
        let baseEmbeddings = MLX.concatenated([MLX.sin(sinusoids), MLX.cos(sinusoids)], axis: -1)
        
        // Cache if enabled and under limit
        if useCache && embeddingCache.count < maxCacheSize {
            embeddingCache[seqLen] = baseEmbeddings
        }
        
        // Apply learnable scaling
        return baseEmbeddings * scale
    }
    
    /// Clear the embedding cache to free memory
    public func clearCache() {
        embeddingCache.removeAll()
    }
}
