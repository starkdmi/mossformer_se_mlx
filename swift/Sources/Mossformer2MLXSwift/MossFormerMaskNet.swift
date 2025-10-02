import Foundation
import MLX
import MLXNN

/// MLX implementation of MossFormer_MaskNet for mask prediction.
///
/// This class is designed for predicting masks used in source separation tasks.
/// It processes input tensors through various layers including convolutional layers,
/// normalization, and a computation block to produce the final output.
public class MossFormerMaskNet: Module {
    public let inChannels: Int
    public let outChannels: Int
    public let outChannelsFinal: Int
    public let numBlocks: Int
    public let normType: String
    public let numSpks: Int
    public let skipAroundIntra: Bool
    public let useGlobalPosEnc: Bool
    public let maxLength: Int
    
    // Model components
    @ModuleInfo var norm: Module
    @ModuleInfo var conv1d_encoder: Conv1d
    @ModuleInfo var pos_enc: ScaledSinuEmbedding?
    @ModuleInfo var mdl: ComputationBlock
    @ModuleInfo var conv1d_out: Conv1d
    @ModuleInfo var conv1_decoder: Conv1d
    @ModuleInfo var prelu: PReLU
    @ModuleInfo var activation: ReLU
    @ModuleInfo var output: Conv1d
    @ModuleInfo var output_gate: Conv1d
    
    // Compiled forward function for better performance
    private var forwardCompiled: ((MLXArray) -> MLXArray)!
    
    /// Initialize MossFormer_MaskNet
    /// - Parameters:
    ///   - inChannels: Number of channels at the output of the encoder
    ///   - outChannels: Number of channels that would be inputted to the MossFormer2 blocks
    ///   - outChannelsFinal: Number of channels that are finally outputted
    ///   - numBlocks: Number of layers in the Dual Computation Block (default: 24)
    ///   - normType: Normalization type ('ln' for LayerNorm, 'gln' for GlobalLayerNorm) (default: 'gln')
    ///   - numSpks: Number of sources (speakers) (default: 2)
    ///   - skipAroundIntra: If true, applies skip connections around intra-block connections (default: true)
    ///   - useGlobalPosEnc: If true, uses global positional encodings (default: true)
    ///   - maxLength: Maximum sequence length for input tensors (default: 20000)
    public init(
        inChannels: Int,
        outChannels: Int,
        outChannelsFinal: Int,
        numBlocks: Int = 24,
        normType: String = "gln",
        numSpks: Int = 2,
        skipAroundIntra: Bool = true,
        useGlobalPosEnc: Bool = true,
        maxLength: Int = 20000
    ) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.outChannelsFinal = outChannelsFinal
        self.numBlocks = numBlocks
        self.normType = normType
        self.numSpks = numSpks
        self.skipAroundIntra = skipAroundIntra
        self.useGlobalPosEnc = useGlobalPosEnc
        self.maxLength = maxLength
        
        // Initialize normalization layer first
        switch normType {
        case "gln":
            self.norm = GlobalLayerNorm(dim: inChannels, shape: 3)
        case "ln":
            // Group norm with 1 group is equivalent to layer norm
            self.norm = GroupNorm(groupCount: 1, dimensions: inChannels, eps: 1e-8)
        default:
            fatalError("Unsupported norm type: \(normType)")
        }
        
        // Encoder convolutional layer (1x1 convolution)
        self.conv1d_encoder = Conv1d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1, bias: false)
        
        // Positional encoding
        if useGlobalPosEnc {
            self.pos_enc = ScaledSinuEmbedding(dim: outChannels)
        } else {
            self.pos_enc = nil
        }
        
        // Computation block
        self.mdl = ComputationBlock(
            numBlocks: numBlocks,
            outChannels: outChannels,
            norm: "ln",
            skipAroundIntra: skipAroundIntra,
            useMossformer2: false  // Use MossFormerM (default)
        )
        
        // Output layers
        self.conv1d_out = Conv1d(inputChannels: outChannels, outputChannels: outChannels * numSpks, kernelSize: 1, bias: true)
        self.conv1_decoder = Conv1d(inputChannels: outChannels, outputChannels: outChannelsFinal, kernelSize: 1, bias: false)
        
        // PReLU activation
        self.prelu = PReLU(count: 1)
        
        // ReLU activation for final output
        self.activation = ReLU()
        
        // Gated output layers
        self.output = Conv1d(inputChannels: outChannels, outputChannels: outChannels, kernelSize: 1, bias: true)
        self.output_gate = Conv1d(inputChannels: outChannels, outputChannels: outChannels, kernelSize: 1, bias: true)
        
        super.init()
        
        // Compile the forward function for better performance
        self.forwardCompiled = MLX.compile(self.forward)
    }
    
    /// Internal forward pass implementation
    /// - Parameter x: Input tensor of shape (batch_size, in_channels, sequence_length)
    /// - Returns: Output tensor of shape (batch_size, out_channels_final, sequence_length)
    private func forward(_ x: MLXArray) -> MLXArray {
        var output = x
        
        // Normalize the input
        // [B, N, L]
        if normType == "gln", let gln = norm as? GlobalLayerNorm {
            output = gln(output)
            // Apply encoder convolution
            // MLX Conv1d expects NLC format, so transpose from NCL to NLC
            output = output.transposed(0, 2, 1)  // [B, L, N]
            output = conv1d_encoder(output)       // [B, L, out_channels]
            output = output.transposed(0, 2, 1)  // [B, out_channels, L]
        } else if let groupNorm = norm as? GroupNorm {
            // For GroupNorm, transpose once to NLC and keep it
            output = output.transposed(0, 2, 1)  // [B, L, N]
            output = groupNorm(output)
            // Apply encoder convolution (already in NLC format)
            output = conv1d_encoder(output)       // [B, L, out_channels]
            output = output.transposed(0, 2, 1)  // [B, out_channels, L]
        }
        
        if useGlobalPosEnc, let posEnc = pos_enc {
                let base = output  // Store the base for adding positional embedding
            output = output.transposed(0, 2, 1)  // Change shape to [B, L, out_channels] for positional encoding
            var emb = posEnc(output)  // Get positional embeddings
            
            // Note: pos_enc returns [seq_len, dim], we need to expand to [B, L, dim]
            if emb.ndim == 2 {
                // If positional encoding returns [L, dim], expand to [B, L, dim]
                // MLX will automatically broadcast when we add it to base
                emb = emb.expandedDimensions(axis: 0)
            }
            emb = emb.transposed(0, 2, 1)  // Change back to [B, out_channels, L]
            output = base + emb  // Add positional embeddings to the base
        }
        
        // Process through the computation block
        // [B, out_channels, L]
        output = mdl(output)
        
        output = prelu(output)  // Apply activation
        
        // Expand to multiple speakers
        // [B, out_channels, L] -> [B, out_channels*num_spks, L]
        output = output.transposed(0, 2, 1)  // [B, L, out_channels]
        output = conv1d_out(output)           // [B, L, out_channels*num_spks]
        output = output.transposed(0, 2, 1)  // [B, out_channels*num_spks, L]
        let B = output.shape[0]
        let S = output.shape[2]  // sequence length
        
        // Reshape to [B*num_spks, out_channels, L]
        // This prepares the output for gating
        output = output.reshaped([B * numSpks, -1, S])
        
        // Apply gated output layers
        // [B*num_spks, out_channels, L]
        output = output.transposed(0, 2, 1)  // [B*num_spks, L, out_channels]
        let outputVal = MLX.tanh(self.output(output))     // [B*num_spks, L, out_channels]
        let gateVal = MLX.sigmoid(output_gate(output))     // [B*num_spks, L, out_channels]
        output = outputVal * gateVal  // Element-wise multiplication for gating
        
        // Decode to final output (already in NLC format)
        // [B*num_spks, L, out_channels] -> [B*num_spks, L, out_channels_final]
        output = conv1_decoder(output)        // [B*num_spks, L, out_channels_final]
        output = output.transposed(0, 2, 1)  // [B*num_spks, out_channels_final, L]
        
        // Reshape to [B, num_spks, out_channels_final, L] for output
        let N = output.shape[1]
        let L = output.shape[2]
        output = output.reshaped([B, numSpks, N, L])
        output = activation(output)  // Apply final activation
        
        // Transpose to [num_spks, B, out_channels_final, L] for output
        // return the 1st spk signal as the target speech
        output = output.transposed(1, 0, 2, 3)  // [num_spks, B, out_channels_final, L]
        let finalOutput = output[0].transposed(0, 2, 1)    // Return [B, L, out_channels_final] (first speaker only)
        return finalOutput
    }
    
    /// Forward pass through the MossFormer_MaskNet (compiled for performance)
    /// - Parameter x: Input tensor of shape (batch_size, in_channels, sequence_length)
    /// - Returns: Output tensor of shape (batch_size, out_channels_final, sequence_length)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return forwardCompiled(x)
    }
}
