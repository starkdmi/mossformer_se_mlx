import Foundation
import MLX
import MLXNN

/// MLX implementation of ScaleNorm.
///
/// ScaleNorm implements a scaled normalization technique for neural network layers.
/// It computes the L2 norm along the last dimension and applies learnable scaling.
public class ScaleNorm: Module, UnaryLayer {
    public let dim: Int
    public let scale: Float
    public let eps: Float
    
    // Learnable scaling parameter
    public var g: MLXArray
    
    /// Initialize ScaleNorm
    /// - Parameters:
    ///   - dim: Dimension of the input features (used to calculate scale factor)
    ///   - eps: Small value to prevent division by zero (default: 1e-8)
    public init(dim: Int, eps: Float = 1e-8) {
        self.dim = dim
        self.scale = pow(Float(dim), -0.5)  // Calculate scale factor: 1/sqrt(dim)
        self.eps = eps
        
        // Initialize learnable scaling parameter
        self.g = MLXArray.ones([1])
        
        super.init()
    }
    
    /// Forward pass for the ScaleNorm layer
    /// - Parameter x: Input tensor of any shape where the last dimension has size `dim`
    /// - Returns: Scaled and normalized output tensor of the same shape as input
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Compute L2 norm along the last dimension
        let norm = MLX.sqrt((x * x).sum(axis: -1, keepDims: true)) * scale
        
        // Clamp norm to prevent division by zero
        let clampedNorm = MLX.maximum(norm, eps)

        // Normalize and scale with fused operation
        return x * (g / clampedNorm)
    }
}
