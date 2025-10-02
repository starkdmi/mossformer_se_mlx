import Foundation
import MLX
import MLXNN

/// MLX Swift implementation of MossFormerM2 transformer encoder.
///
/// This class implements the transformer encoder using standard MossFormer blocks
/// without the Gated FSMN components (previous version).
///
/// Args:
///     numBlocks (int): Number of mossformer blocks to include.
///     dModel (int): The dimension of the input embedding.
///     causal (bool, optional): True for causal / false for non causal. Default is false.
///     groupSize (int, optional): The chunk size. Default is 256.
///     queryKeyDim (int, optional): The attention vector dimension. Default is 128.
///     expansionFactor (float, optional): The expansion factor for the linear projection in conv module. Default is 4.0.
///     attnDropout (float, optional): Dropout for the self-attention. Default is 0.1.
///
/// Shape:
///     - Input: (batch_size, sequence_length, d_model)
///     - Output: (batch_size, sequence_length, d_model)
///
/// Example:
///     >>> let x = MLXRandom.normal([8, 60, 512])
///     >>> let net = MossFormerM2(numBlocks: 8, dModel: 512)
///     >>> let output = net(x)
///     >>> print(output.shape)  // [8, 60, 512]
public class MossFormerM2: Module, MossFormerVariant {
    public let numBlocks: Int
    public let dModel: Int?
    public let causal: Bool
    public let groupSize: Int
    public let queryKeyDim: Int
    public let expansionFactor: Float
    public let attnDropout: Float
    
    // Components
    public var mossformerM: MossformerBlock
    public var norm: LayerNorm
    
    /// Initialize MossFormerM2
    /// - Parameters:
    ///   - numBlocks: Number of mossformer blocks to include
    ///   - dModel: The dimension of the input embedding
    ///   - causal: True for causal / false for non causal (default: false)
    ///   - groupSize: The chunk size (default: 256)
    ///   - queryKeyDim: The attention vector dimension (default: 128)
    ///   - expansionFactor: The expansion factor for the linear projection in conv module (default: 4.0)
    ///   - attnDropout: Dropout for the self-attention (default: 0.1)
    public init(
        numBlocks: Int,
        dModel: Int? = nil,
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
        
        guard let dModel = dModel else {
            fatalError("dModel must be specified")
        }
        
        // Initialize the MossFormer blocks (without GFSMN)
        self.mossformerM = MossformerBlock(
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
    
    /// Forward pass through the MossFormerM2 model
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
