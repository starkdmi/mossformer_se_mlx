import Foundation
import MLX
import MLXNN

/// MLX implementation of Gated_FSMN_Block with mathematical equivalence to PyTorch.
///
/// A 1-D convolutional block that incorporates a gated FSMN.
/// This block consists of:
/// 1. Conv1d layer with PReLU activation
/// 2. CLayerNorm normalization
/// 3. Gated FSMN module
/// 4. Another CLayerNorm
/// 5. Final Conv1d projection
/// 6. Residual connection
public class GatedFSMNBlock: Module {
    public let dim: Int
    public let innerChannels: Int
    public let groupSize: Int
    public let normType: String
    
    // Model components
    @ModuleInfo var conv1: Conv1d
    @ModuleInfo var prelu: PReLU
    @ModuleInfo var norm1: CLayerNorm
    @ModuleInfo var norm2: CLayerNorm
    @ModuleInfo var gated_fsmn: GatedFSMN
    @ModuleInfo var conv2: Conv1d
    
    /// Initialize Gated_FSMN_Block
    /// - Parameters:
    ///   - dim: Dimensionality of the input/output
    ///   - innerChannels: Number of channels in the inner layers (default: 256)
    ///   - groupSize: Size of the groups for normalization (default: 256)
    ///   - normType: Type of normalization to use ('scalenorm' or 'layernorm')
    public init(
        dim: Int,
        innerChannels: Int = 256,
        groupSize: Int = 256,
        normType: String = "scalenorm"
    ) {
        self.dim = dim
        self.innerChannels = innerChannels
        self.groupSize = groupSize
        self.normType = normType
        
        // First convolutional layer
        self.conv1 = Conv1d(
            inputChannels: dim,
            outputChannels: innerChannels,
            kernelSize: 1,
            bias: true
        )
        
        // PReLU activation
        self.prelu = PReLU(count: 1)
        
        // Normalization layers
        self.norm1 = CLayerNorm(normalizedShape: innerChannels)
        self.norm2 = CLayerNorm(normalizedShape: innerChannels)
        
        // Gated FSMN
        self.gated_fsmn = GatedFSMN(
            inChannels: innerChannels,
            outChannels: innerChannels,
            lorder: 20,
            hiddenSize: innerChannels
        )
        
        // Final convolutional layer
        self.conv2 = Conv1d(
            inputChannels: innerChannels,
            outputChannels: dim,
            kernelSize: 1,
            bias: true
        )
        
        super.init()
    }
    
    /// Forward pass for the Gated FSMN Block
    /// - Parameter x: Input tensor of shape [batch_size, seq_length, dim]
    /// - Returns: Output tensor of shape [batch_size, seq_length, dim]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x

        // First convolution - input is already [B, T, D]
        var output = conv1(x)

        // PReLU activation
        output = prelu(output)

        // First normalization (now accepts [B, T, C] directly)
        output = norm1(output)
        
        // Gated FSMN (expects [B, T, C])
        output = gated_fsmn(output)
        
        // Second normalization (now accepts [B, T, C] directly)
        output = norm2(output)
        
        // Final convolution
        output = conv2(output)
        
        // Residual connection
        let result = output + residual
        return result
    }
}

// MARK: - GatedFSMN Implementation

/// MLX Swift implementation of Gated_FSMN with mathematical equivalence to Python MLX.
///
/// Gated Frequency Selective Memory Network (FSMN) that combines two feedforward
/// convolutional networks with a frequency selective memory module.
///
/// The gated FSMN uses a gating mechanism to selectively combine:
/// 1. A feedforward branch (toU) processed through FSMN for temporal memory
/// 2. A feedforward branch (toV) used as the gate
/// 3. The original input as a residual connection
///
/// The operation is: output = gate * memory + input
/// where gate = toV(input) and memory = fsmn(toU(input))
///
/// Args:
///     inChannels (int): Number of input channels
///     outChannels (int): Number of output channels
///     lorder (int): Order of the filter for FSMN (memory span)
///     hiddenSize (int): Number of hidden units in the network
public class GatedFSMN: Module {
    public let inChannels: Int
    public let outChannels: Int
    public let lorder: Int
    public let hiddenSize: Int
    
    // Components
    @ModuleInfo var to_u: FFConvM
    @ModuleInfo var to_v: FFConvM
    @ModuleInfo var fsmn: UniDeepFSMN
    
    /// Initialize Gated FSMN
    /// - Parameters:
    ///   - inChannels: Number of input channels
    ///   - outChannels: Number of output channels
    ///   - lorder: Order of the filter for FSMN (memory span)
    ///   - hiddenSize: Number of hidden units in the network
    public init(
        inChannels: Int,
        outChannels: Int,
        lorder: Int,
        hiddenSize: Int
    ) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.lorder = lorder
        self.hiddenSize = hiddenSize
        
        // Feedforward network for the first branch (u)
        // This branch will be processed through FSMN for temporal memory
        self.to_u = FFConvM(
            dimIn: inChannels,
            dimOut: hiddenSize,
            normKlass: LayerNorm.self,
            dropout: 0.1
        )
        
        // Feedforward network for the second branch (v)
        // This branch acts as the gate
        self.to_v = FFConvM(
            dimIn: inChannels,
            dimOut: hiddenSize,
            normKlass: LayerNorm.self,
            dropout: 0.1
        )
        
        // Frequency selective memory network
        // Following PyTorch implementation exactly
        self.fsmn = UniDeepFSMN(
            inputDim: inChannels,
            outputDim: outChannels,
            lorder: lorder,
            hiddenSize: hiddenSize
        )
        
        super.init()
    }
    
    /// Forward pass for the Gated FSMN.
    ///
    /// The gated FSMN performs the following operations:
    /// 1. Process input through first branch (toU)
    /// 2. Process input through second branch (toV) - acts as gate
    /// 3. Apply FSMN to first branch output for temporal memory
    /// 4. Combine: output = gate * memory + input (residual connection)
    ///
    /// - Parameter x: Input tensor of shape (batch, time, inChannels)
    /// - Returns: Output tensor of shape (batch, time, outChannels)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Store original input for residual connection
        let inputResidual = x
        
        // Process input through both branches
        let xU = to_u(x)    // First branch - will be processed through FSMN
        let xV = to_v(x)    // Second branch - acts as gate
        
        // Apply FSMN to the first branch for temporal memory
        let xUProcessed = fsmn(xU)
        
        // Gated combination with residual connection
        // Following PyTorch: x = x_v * x_u + input
        return xV * xUProcessed + inputResidual
    }
}
