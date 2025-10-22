import Foundation
import MLX
import MLXNN

/// MLX Swift implementation of FFConvM.
///
/// This class provides identical behavior to the Python MLX version.
///
/// FFConvM structure: norm -> linear -> silu -> conv_module -> dropout
///
/// Args:
///     dimIn (int): Input dimension
///     dimOut (int): Output dimension
///     normKlass: Normalization class (default: LayerNorm)
///     dropout (float): Dropout probability (default: 0.1)
public class FFConvM: Module {
    public let dimIn: Int
    public let dimOut: Int
    
    // Components
    @ModuleInfo var norm: Module
    @ModuleInfo var linear: Linear
    @ModuleInfo var conv_module: ConvModule
    @ModuleInfo var dropout: Dropout
    private let dropoutRate: Float
    
    /// Initialize FFConvM
    /// - Parameters:
    ///   - dimIn: Input dimension
    ///   - dimOut: Output dimension
    ///   - normKlass: Normalization class type (default: LayerNorm)
    ///   - dropout: Dropout probability (default: 0.1)
    public init(
        dimIn: Int,
        dimOut: Int,
        normKlass: Module.Type? = nil,
        dropout: Float = 0.1
    ) {
        self.dimIn = dimIn
        self.dimOut = dimOut
        self.dropoutRate = dropout
        
        // Sequential structure matching PyTorch: norm, linear, silu, conv_module, dropout
        // Index 0: normalization
        if let normType = normKlass {
            if normType == LayerNorm.self {
                self.norm = LayerNorm(dimensions: dimIn)
            } else if normType == ScaleNorm.self {
                self.norm = ScaleNorm(dim: dimIn)
            } else {
                // Default to ScaleNorm since that's what the weights expect
                self.norm = ScaleNorm(dim: dimIn)
            }
        } else {
            // Default to ScaleNorm
            self.norm = ScaleNorm(dim: dimIn)
        }
        
        // Index 1: linear transformation
        self.linear = Linear(dimIn, dimOut)
        
        // Index 2: SiLU activation (no parameters)
        
        // Index 3: ConvModule
        self.conv_module = ConvModule(dim: dimOut)
        
        // Index 4: Dropout
        self.dropout = Dropout(p: dropout)
        
        super.init()
    }
    
    /// Forward pass for FFConvM
    /// - Parameter x: Input tensor of shape (batch, time, dimIn)
    /// - Returns: Output tensor of shape (batch, time, dimOut)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Follow PyTorch FFConvM.mdl sequential structure
        var output = x
        
        // Cast norm to proper type and call it
        if let layerNorm = norm as? LayerNorm {
            // Use fast LayerNorm for better performance
            output = MLXFast.layerNorm(output, weight: layerNorm.weight, bias: layerNorm.bias, eps: layerNorm.eps)      // Index 0
        } else if let scaleNorm = norm as? ScaleNorm {
            output = scaleNorm(output)      // Index 0
        } else {
            fatalError("Unexpected norm type")
        }
        
        // Linear layer
        output = linear(output)         // Index 1
        
        output = silu(output)           // Index 2
        
        output = conv_module(output)    // Index 3
        
        // output = dropout(output)   // Index 4 - Commented out for inference only
        
        return output
    }
}

// MARK: - ConvModule Implementation

/// MLX Swift implementation of ConvModule.
///
/// This class provides identical behavior to the Python MLX version.
///
/// ConvModule structure:
/// - Transpose (1, 2): (B, T, C) -> (B, C, T)
/// - DepthwiseConv1d: channels -> same channels with depthwise convolution
/// - Transpose back + residual connection
///
/// Args:
///     inChannels (int): Number of input channels
///     kernelSize (int): Convolution kernel size (default: 17)
///     expansionFactor (int): Expansion factor (default: 2, currently unused)
///     dropoutP (float): Dropout probability (default: 0.1, currently unused)
public class ConvModule: Module {
    public let inChannels: Int
    public let kernelSize: Int
    public let padding: Int
    
    /// Convolution weight: Shape (out_channels, kernel_size, 1) for MLX conv1d
    @ModuleInfo var weight: MLXArray
    
    /// Initialize ConvModule
    /// - Parameters:
    ///   - dim: Number of input channels (alternative parameter name)
    ///   - inChannels: Number of input channels
    ///   - kernelSize: Convolution kernel size (default: 17)
    ///   - expansionFactor: Expansion factor (default: 2, currently unused)
    ///   - dropoutP: Dropout probability (default: 0.1, currently unused)
    public init(
        dim: Int? = nil,
        inChannels: Int? = nil,
        kernelSize: Int = 17,
        expansionFactor: Int = 2,
        dropoutP: Float = 0.1
    ) {
        // Support both parameter names
        let channels = dim ?? inChannels ?? 256
        
        // Validate inputs like PyTorch version
        precondition((kernelSize - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding")
        precondition(expansionFactor == 2, "Currently, Only Supports expansion_factor 2")
        
        self.inChannels = channels
        self.kernelSize = kernelSize
        self.padding = (kernelSize - 1) / 2
        
        // Initialize weight storage BEFORE super.init()
        // Shape: (out_channels, kernel_size, 1) for MLX conv1d
        // For depthwise conv, out_channels = in_channels
        // Initialize with zeros like Python to ensure weights are loaded from checkpoint
        self.weight = MLXArray.zeros([channels, kernelSize, 1])
        
        super.init()
    }
    
    /// Forward pass for ConvModule exactly matching PyTorch behavior:
    /// inputs + self.sequential(inputs).transpose(1, 2)
    ///
    /// Where sequential is:
    /// 1. Transpose(1, 2): (B, T, C) → (B, C, T)
    /// 2. DepthwiseConv1d: convolution on (B, C, T)
    ///
    /// - Parameter x: Input tensor of shape (B, T, C)
    /// - Returns: Output tensor of shape (B, T, C)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x  // (B, T, C)
        
        let convOut = DepthwiseConv1d.apply(
            x,
            weight: weight,
            stride: 1,
            padding: padding,
            groups: inChannels
        )
        
        return residual + convOut
    }
}
