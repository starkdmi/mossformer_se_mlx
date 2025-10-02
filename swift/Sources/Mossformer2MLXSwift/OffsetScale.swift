import Foundation
import MLX
import MLXNN

/// MLX Swift implementation of OffsetScale.
///
/// OffsetScale applies learned offsets and scales to the input tensor for multiple heads.
/// It performs element-wise scaling and offset operations and returns a list of tensors,
/// one for each head.
///
/// Arguments:
///     dim: Dimension of the input features
///     heads: Number of heads (default: 1)
public class OffsetScale: Module {
    public let dim: Int
    public let heads: Int
    
    /// Scale parameters (gamma) for each head
    public var gamma: MLXArray
    
    /// Offset parameters (beta) for each head
    public var beta: MLXArray
    
    /// Initialize OffsetScale
    /// - Parameters:
    ///   - dim: Dimension of the input features
    ///   - heads: Number of heads (default: 1)
    public init(_ dim: Int, heads: Int = 1) {
        self.dim = dim
        self.heads = heads
        
        // Initialize scale parameters (gamma) with normal distribution BEFORE super.init()
        self.gamma = MLXRandom.normal([heads, dim], scale: 0.02) + 1.0
        
        // Initialize offset parameters (beta) with zeros BEFORE super.init()
        self.beta = MLXArray.zeros([heads, dim])
        
        super.init()
    }
    
    /// Forward pass for the OffsetScale layer
    /// - Parameter x: Input tensor of shape (..., dim)
    /// - Returns: List of tensors with applied offsets and scales for each head.
    ///           Each tensor has shape (..., dim)
    public func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        // Apply scaling and offsets using einsum-like operation
        // PyTorch: einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        
        // Expand x to include head dimension: (..., dim) -> (..., 1, dim)
        let xExpanded = x.expandedDimensions(axis: -2)
        
        // Broadcast multiplication with gamma: (..., 1, dim) * (heads, dim) -> (..., heads, dim)
        let scaled = xExpanded * gamma
        
        // Add beta offset: (..., heads, dim) + (heads, dim) -> (..., heads, dim)
        let out = scaled + beta
        
        // Unbind along the head dimension to create a list of tensors
        var headOutputs: [MLXArray] = []
        for h in 0..<heads {
            // Extract head h: (..., dim)
            let headOutput = out[.ellipsis, h, 0...]
            headOutputs.append(headOutput)
        }
        
        return headOutputs
    }
}
