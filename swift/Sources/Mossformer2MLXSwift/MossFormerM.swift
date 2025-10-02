import Foundation
import MLX
import MLXNN

/// Protocol for MossFormer model variants to enable polymorphic usage
protocol MossFormerVariant: Module {
    func callAsFunction(_ src: MLXArray) -> MLXArray
}

/// MLX implementation of MossFormerM transformer encoder based on MossFormer2 layers.
///
/// This class implements the transformer encoder using MossFormer2 layers with
/// Gated FSMN blocks for enhanced sequence processing.
public class MossFormerM: Module, MossFormerVariant {
    public let numBlocks: Int
    public let dModel: Int
    public let causal: Bool
    public let groupSize: Int
    public let queryKeyDim: Int
    public let expansionFactor: Float
    public let attnDropout: Float
    
    // MossFormer blocks with GFSMN
    @ModuleInfo var mossformerM: MossformerBlockGFSMN
    // Layer normalization
    @ModuleInfo var norm: LayerNorm
    
    /// Initialize MossFormerM
    /// - Parameters:
    ///   - numBlocks: Number of mossformer2 blocks to include
    ///   - dModel: The dimension of the input embedding
    ///   - causal: True for causal / false for non causal (default: false)
    ///   - groupSize: The chunk size for segmenting sequence (default: 256)
    ///   - queryKeyDim: The attention vector dimension (default: 128)
    ///   - expansionFactor: The expansion factor for the linear projection in conv module (default: 4.0)
    ///   - attnDropout: Dropout for the self-attention (default: 0.1)
    public init(
        numBlocks: Int,
        dModel: Int,
        causal: Bool = false,
        groupSize: Int = 256,
        queryKeyDim: Int = 128,
        expansionFactor: Float = 4.0,
        attnDropout: Float = 0.1
    ) {
        self.numBlocks = numBlocks
        self.dModel = dModel
        self.causal = causal
        self.groupSize = groupSize
        self.queryKeyDim = queryKeyDim
        self.expansionFactor = expansionFactor
        self.attnDropout = attnDropout
        
        // Initialize the MossFormer blocks with GFSMN
        self.mossformerM = MossformerBlockGFSMN(
            dim: dModel,
            depth: numBlocks,
            groupSize: groupSize,
            queryKeyDim: queryKeyDim,
            expansionFactor: expansionFactor,
            causal: causal,
            attnDropout: attnDropout
        )
        
        // Layer normalization
        self.norm = LayerNorm(dimensions: dModel, eps: 1e-8)
        
        super.init()
    }
    
    /// Forward pass through the MossFormerM model
    /// - Parameter src: Input tensor of shape (batch_size, sequence_length, d_model)
    /// - Returns: Output tensor of shape (batch_size, sequence_length, d_model)
    public func callAsFunction(_ src: MLXArray) -> MLXArray {
        
        // Apply MossFormer blocks
        var output = mossformerM(src)
        
        // Apply layer normalization
        output = norm(output)
        
        return output
    }
}

// MARK: - MossformerBlock Implementation

/// MLX Swift implementation of Mossformer Block with attention mechanisms.
///
/// This block is designed to process input sequences using attention
/// layers and incorporates rotary positional embeddings. It allows
/// for configurable normalization types and can handle causal
/// attention.
///
/// Args:
///     dim (int): Dimensionality of the input.
///     depth (int): Number of attention layers in the block.
///     groupSize (int, optional): Size of groups for normalization. Default is 256.
///     queryKeyDim (int, optional): Dimension of the query and key in attention. Default is 128.
///     expansionFactor (float, optional): Expansion factor for feedforward layers. Default is 4.
///     causal (bool, optional): If True, enables causal attention. Default is false.
///     attnDropout (float, optional): Dropout rate for attention layers. Default is 0.1.
///     normType (str, optional): Type of normalization to use ('scalenorm' or 'layernorm'). Default is 'scalenorm'.
///     shiftTokens (bool, optional): If True, shifts tokens in the attention layer. Default is true.
public class MossformerBlock: Module {
    public let dim: Int
    public let depth: Int
    public let groupSize: Int
    public let queryKeyDim: Int
    public let expansionFactor: Float
    public let causal: Bool
    public let attnDropout: Float
    public let normType: String
    public let shiftTokens: Bool
    
    // Layers
    @ModuleInfo var layers: [FLASHShareAFFConvM]
    
    /// Initialize MossformerBlock
    public init(
        dim: Int,
        depth: Int,
        groupSize: Int = 256,
        queryKeyDim: Int = 128,
        expansionFactor: Float = 4.0,
        causal: Bool = false,
        attnDropout: Float = 0.1,
        normType: String = "scalenorm",
        shiftTokens: Bool = true
    ) {
        self.dim = dim
        self.depth = depth
        self.groupSize = groupSize
        self.queryKeyDim = queryKeyDim
        self.expansionFactor = expansionFactor
        self.causal = causal
        self.attnDropout = attnDropout
        self.normType = normType
        self.shiftTokens = shiftTokens
        
        // Ensure normalization type is valid
        precondition(["scalenorm", "layernorm"].contains(normType), "norm_type must be one of scalenorm or layernorm")
        
        // Select normalization class based on the provided type
        let normKlass: Module.Type = (normType == "scalenorm") ? ScaleNorm.self : LayerNorm.self
        
        // Rotary positional embedding for attention
        // Max rotary embedding dimensions of 32, partial Rotary embeddings, from Wang et al - GPT-J
        let rotaryPosEmb = RoPE(
            dimensions: min(32, queryKeyDim),
            traditional: false, // Must match Python MLX implementation
            base: 10000
        )
        
        // Create a list of attention layers using FLASH_ShareA_FFConvM
        var layerArray: [FLASHShareAFFConvM] = []
        
        // Initialize layers
        for _ in 0..<depth {
            let layer = FLASHShareAFFConvM(
                dim: dim,
                groupSize: groupSize,
                queryKeyDim: queryKeyDim,
                expansionFactor: expansionFactor,
                causal: causal,
                dropout: attnDropout,
                rotaryPosEmb: rotaryPosEmb,
                normKlass: normKlass,
                shiftTokens: shiftTokens
            )
            layerArray.append(layer)
        }
        
        self.layers = layerArray
        
        super.init()
    }
    
    /// Forward pass for the Mossformer Block.
    /// - Parameters:
    ///   - x: Input tensor of shape (batch_size, seq_len, dim)
    ///   - mask: Optional mask tensor for attention operations
    /// - Returns: Output tensor after processing through the block
    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var output = x
        
        // Process input through each attention layer
        for (_, layer) in layers.enumerated() {
            output = layer(output, mask: mask)  // Apply attention layer with optional mask
        }
        
        return output  // Return the final output tensor
    }
}

// MARK: - MossformerBlockGFSMN for compatibility

/// MossformerBlockGFSMN implementation using attention and FSMN components
public class MossformerBlockGFSMN: Module {
    public let dim: Int
    public let depth: Int
    public let groupSize: Int
    public let queryKeyDim: Int
    public let expansionFactor: Float
    public let causal: Bool
    public let attnDropout: Float
    
    // FLASH attention layers
    @ModuleInfo var layers: [FLASHShareAFFConvM]
    
    // Gated FSMN blocks - CRITICAL!
    @ModuleInfo var fsmn: [GatedFSMNBlock]
    
    /// Initialize with the same parameters but includes FSMN blocks
    public init(
        dim: Int,
        depth: Int,
        groupSize: Int = 256,
        queryKeyDim: Int = 128,
        expansionFactor: Float = 4.0,
        causal: Bool = false,
        attnDropout: Float = 0.1
    ) {
        self.dim = dim
        self.depth = depth
        self.groupSize = groupSize
        self.queryKeyDim = queryKeyDim
        self.expansionFactor = expansionFactor
        self.causal = causal
        self.attnDropout = attnDropout
        
        // Normalization type is scalenorm for MossformerBlockGFSMN
        let normType = "scalenorm"
        let normKlass: Module.Type = ScaleNorm.self
        
        // Rotary positional embedding for attention
        let rotaryPosEmb = RoPE(
            dimensions: min(32, queryKeyDim),
            traditional: false, // Must match Python MLX implementation
            base: 10000
        )
        
        // Create FLASH attention layers
        var layerArray: [FLASHShareAFFConvM] = []
        for _ in 0..<depth {
            let layer = FLASHShareAFFConvM(
                dim: dim,
                groupSize: groupSize,
                queryKeyDim: queryKeyDim,
                expansionFactor: expansionFactor,
                causal: causal,
                dropout: attnDropout,
                rotaryPosEmb: rotaryPosEmb,
                normKlass: normKlass,
                shiftTokens: true
            )
            layerArray.append(layer)
        }
        self.layers = layerArray
        
        // Create Gated FSMN blocks - matching Python implementation
        var fsmnArray: [GatedFSMNBlock] = []
        for _ in 0..<depth {
            let fsmnBlock = GatedFSMNBlock(
                dim: dim,
                innerChannels: 256,  // Matching Python: inner_channels=256
                groupSize: groupSize,
                normType: normType
            )
            fsmnArray.append(fsmnBlock)
        }
        self.fsmn = fsmnArray
        
        super.init()
    }
    
    /// Forward pass for the Mossformer Block with Gated FSMN.
    /// - Parameters:
    ///   - x: Input tensor of shape (batch_size, seq_len, dim)
    ///   - mask: Optional mask tensor for attention operations
    /// - Returns: Output tensor after processing through the block
    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var output = x
        
        // Process through interleaved FLASH and FSMN layers
        for i in 0..<depth {
            // Apply FLASH attention layer
            // Only pass mask if it's not nil to ensure compiled path is used
            if let mask = mask {
                output = layers[i](output, mask: mask)
            } else {
                output = layers[i](output)
            }
            
            // Apply Gated FSMN block - CRITICAL!
            output = fsmn[i](output)
        }
        
        
        return output
    }
}
