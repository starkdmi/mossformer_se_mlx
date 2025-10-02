import Foundation
import MLX
import MLXNN

/// MLX implementation of Computation_Block for dual-path processing.
///
/// This class implements the computation block that contains MossFormer layers
/// with normalization and skip connections for intra-chunk processing.
public class ComputationBlock: Module {
    public let numBlocks: Int
    public let outChannels: Int
    public let norm: String?
    public let skipAroundIntra: Bool
    public let useMossformer2: Bool
    
    // MossFormer model
    @ModuleInfo var intra_mdl: MossFormerVariant
    // Normalization layer
    @ModuleInfo var intra_norm: GroupNorm?
    
    /// Initialize ComputationBlock
    /// - Parameters:
    ///   - numBlocks: Number of MossFormer blocks
    ///   - outChannels: Dimensionality of model output
    ///   - norm: Normalization type ('ln' for layer norm, nil for no norm) (default: 'ln')
    ///   - skipAroundIntra: Skip connection around the intra layer (default: true)
    ///   - useMossformer2: Use MossFormerM2 instead of MossFormerM (default: false)
    public init(
        numBlocks: Int,
        outChannels: Int,
        norm: String? = "ln",
        skipAroundIntra: Bool = true,
        useMossformer2: Bool = false
    ) {
        self.numBlocks = numBlocks
        self.outChannels = outChannels
        self.norm = norm
        self.skipAroundIntra = skipAroundIntra
        self.useMossformer2 = useMossformer2
        
        // Initialize normalization layer
        if let norm = norm {
            if norm == "ln" {
                // In PyTorch, norm="ln" actually means GroupNorm(1, dim, eps=1e-8)
                self.intra_norm = GroupNorm(groupCount: 1, dimensions: outChannels, eps: 1e-8)
            }
            // Add other normalization types as needed
        } else {
            self.intra_norm = nil
        }
        
        // Initialize the appropriate MossFormer model
        if useMossformer2 {
            self.intra_mdl = MossFormerM2(numBlocks: numBlocks, dModel: outChannels)
        } else {
            // Default MossFormer model
            self.intra_mdl = MossFormerM(numBlocks: numBlocks, dModel: outChannels)
        }
        
        super.init()
    }
    
    /// Forward pass through the Computation_Block
    /// - Parameter x: Input tensor of shape (batch_size, out_channels, sequence_length)
    /// - Returns: Output tensor of shape (batch_size, out_channels, sequence_length)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Convert to NLC format for MossFormer processing
        var intra = x.transposed(0, 2, 1)  // [B, S, N]
        
        // Apply MossFormer model (operates in NLC format)
        intra = intra_mdl(intra)
        
        // Apply normalization if specified (GroupNorm expects NLC format in MLX)
        if let intraNorm = intra_norm {
            intra = intraNorm(intra)
        }
        
        // Convert back to NCL format
        intra = intra.transposed(0, 2, 1)  // [B, N, S]
        
        // Apply skip connection if enabled
        if skipAroundIntra {
            intra = intra + x
        }
        
        return intra
    }
}
