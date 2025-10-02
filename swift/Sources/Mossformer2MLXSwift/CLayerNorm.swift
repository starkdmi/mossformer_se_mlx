import Foundation
import MLX
import MLXNN

/// MLX implementation of CLayerNorm (Channel-wise Layer Normalization).
///
/// This class applies layer normalization along the channel dimension.
/// Unlike the PyTorch version which expects [N, C, T], this MLX version
/// works directly with [N, T, C] format to avoid unnecessary transpositions.
public class CLayerNorm: Module, UnaryLayer {
    public let normalizedShape: Int
    public let eps: Float
    public let elementwiseAffine: Bool
    
    public var weight: MLXArray?
    public var bias: MLXArray?
    
    /// Initialize CLayerNorm
    /// - Parameters:
    ///   - normalizedShape: Input shape from last dimension
    ///   - eps: Small value for numerical stability (default: 1e-8)
    ///   - elementwiseAffine: Whether to use learnable affine parameters (default: true)
    public init(
        normalizedShape: Int,
        eps: Float = 1e-8,
        elementwiseAffine: Bool = true
    ) {
        self.normalizedShape = normalizedShape
        self.eps = eps
        self.elementwiseAffine = elementwiseAffine
        
        if elementwiseAffine {
            self.weight = MLXArray.ones([normalizedShape])
            self.bias = MLXArray.zeros([normalizedShape])
        }
        
        super.init()
    }
    
    /// Forward pass applying channel-wise layer normalization
    /// - Parameter x: Input tensor of shape [batch_size, sequence_length, channels]
    /// - Returns: Normalized tensor of shape [batch_size, sequence_length, channels]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard x.ndim == 3 else {
            fatalError("CLayerNorm only accepts 3-D tensor as input, got \(x.ndim)D")
        }
        
        // x is already in [N, T, C] format
        // Apply LayerNorm along the channel dimension (last dimension)
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = x.variance(axis: -1, keepDims: true)
        var output = (x - mean) / MLX.sqrt(variance + eps)
        
        if elementwiseAffine, let weight = weight, let bias = bias {
            output = output * weight + bias
        }
        
        return output
    }
}
