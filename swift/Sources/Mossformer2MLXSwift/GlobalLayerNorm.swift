import Foundation
import MLX
import MLXNN

/// MLX implementation of Global Layer Normalization.
/// 
/// This class calculates Global Layer Normalization, providing identical
/// behavior to the PyTorch version for both 3D and 4D tensors.
public class GlobalLayerNorm: Module, UnaryLayer {
    public let dim: Int
    public let shape: Int
    public let eps: Float
    public let elementwiseAffine: Bool
    
    // Use @ModuleInfo for trainable parameters
    public var weight: MLXArray
    public var bias: MLXArray
    
    /// Initialize Global Layer Normalization
    /// - Parameters:
    ///   - dim: Input dimension size
    ///   - shape: Number of dimensions (3 or 4)
    ///   - eps: Small value for numerical stability (default: 1e-8)
    ///   - elementwiseAffine: Whether to use learnable affine parameters (default: true)
    public init(
        dim: Int,
        shape: Int,
        eps: Float = 1e-8,
        elementwiseAffine: Bool = true
    ) {
        self.dim = dim
        self.shape = shape
        self.eps = eps
        self.elementwiseAffine = elementwiseAffine
        
        // Initialize weight and bias before super.init()
        switch shape {
        case 3:
            // Weight and bias for 3D tensors: initialize as [dim, 1] to match PyTorch
            self.weight = MLXArray.ones([dim, 1])
            self.bias = MLXArray.zeros([dim, 1])
        case 4:
            // Weight and bias for 4D tensors: initialize as [dim, 1, 1] to match PyTorch
            self.weight = MLXArray.ones([dim, 1, 1])
            self.bias = MLXArray.zeros([dim, 1, 1])
        default:
            fatalError("Unsupported shape: \(shape). Only 3 and 4 are supported.")
        }
        
        super.init()
    }
    
    /// Forward pass for Global Layer Normalization
    /// - Parameter x: Input tensor of size [N, C, L] for 3D or [N, C, K, S] for 4D
    /// - Returns: Normalized tensor of the same shape as input
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard x.ndim == 3 || x.ndim == 4 else {
            fatalError("Expected 3D or 4D tensor, got \(x.ndim)D tensor")
        }
        
        if x.ndim == 3 {
            // For 3D tensors: normalize over dimensions (1, 2)
            // Input is [B, C, L] in PyTorch convention
            let mean = x.mean(axes: [1, 2], keepDims: true)
            let variance = ((x - mean) ** 2).mean(axes: [1, 2], keepDims: true)
            
            if elementwiseAffine {
                // Optimize broadcasting for weight and bias
                // Reshape weight and bias once for efficient broadcasting
                let w: MLXArray
                let b: MLXArray
                
                if weight.ndim > 1 {
                    w = weight.squeezed().reshaped([1, -1, 1])
                    b = bias.squeezed().reshaped([1, -1, 1])
                } else {
                    w = weight.reshaped([1, -1, 1])
                    b = bias.reshaped([1, -1, 1])
                }
                
                // Fuse normalization and affine transformation
                return w * (x - mean) / MLX.sqrt(variance + eps) + b
            } else {
                return (x - mean) / MLX.sqrt(variance + eps)
            }
            
        } else { // x.ndim == 4
            // For 4D tensors: normalize over dimensions (1, 2, 3)
            let mean = x.mean(axes: [1, 2, 3], keepDims: true)
            let variance = ((x - mean) ** 2).mean(axes: [1, 2, 3], keepDims: true)
            
            if elementwiseAffine {
                // Optimize broadcasting for weight and bias
                // Reshape weight and bias once for efficient broadcasting
                let w: MLXArray
                let b: MLXArray
                
                if weight.ndim > 1 {
                    w = weight.squeezed().reshaped([1, -1, 1, 1])
                    b = bias.squeezed().reshaped([1, -1, 1, 1])
                } else {
                    w = weight.reshaped([1, -1, 1, 1])
                    b = bias.reshaped([1, -1, 1, 1])
                }
                
                // Fuse normalization and affine transformation
                return w * (x - mean) / MLX.sqrt(variance + eps) + b
            } else {
                return (x - mean) / MLX.sqrt(variance + eps)
            }
        }
    }
}
