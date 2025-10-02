import Foundation
import MLX
import MLXNN
import MLXFast

/// MLX Swift implementation of FLASH_ShareA_FFConvM with mathematical equivalence to Python MLX.
///
/// Fast Shared Dual Attention Mechanism with feed-forward convolutional blocks.
/// Published in paper: "MossFormer: Pushing the Performance Limit of Monaural Speech Separation
/// using Gated Single-Head Transformer with Convolution-Augmented Joint Self-Attentions", ICASSP 2023.
/// (https://arxiv.org/abs/2302.11824)
///
/// Args:
///     dim (int): Input dimension
///     groupSize (int): Size of groups for processing (default: 256)
///     queryKeyDim (int): Dimension of the query and key (default: 128)
///     expansionFactor (float): Factor to expand the hidden dimension (default: 4.0)
///         Note: The architecture is designed for expansionFactor=4.0. Other values
///         will cause dimension mismatches because toOut expects dim*2 inputs, and
///         the gated outputs produce hiddenDim/2 = dim*expansionFactor/2 dimensions
///     causal (bool): Whether to use causal masking (default: false)
///     dropout (float): Dropout rate (default: 0.1)
///     rotaryPosEmb: Rotary positional embeddings for attention (default: nil)
///     normKlass: Normalization class to use (default: LayerNorm)
///     shiftTokens (bool): Whether to shift tokens for attention calculation (default: true)
public class FLASHShareAFFConvM: Module {
    // Cache for causal masks
    private static var causalMaskCache: [Int: MLXArray] = [:]
    // Configuration
    public let dim: Int
    public let groupSize: Int
    public let queryKeyDim: Int
    public let expansionFactor: Float
    public let causal: Bool
    public let dropoutP: Float
    public let shiftTokens: Bool
    
    // Calculated dimensions
    public let hiddenDim: Int
    
    // Components
    @ModuleInfo var to_hidden: FFConvM
    @ModuleInfo var to_qk: FFConvM
    @ModuleInfo var qk_offset_scale: OffsetScale
    @ModuleInfo var to_out: FFConvM
    @ModuleInfo var dropout: Dropout
    @ModuleInfo var rotary_pos_emb: Module?
    
    // Compiled forward function for better performance
    private var forwardCompiled: ((MLXArray) -> MLXArray)!
    
    /// Initialize FLASH_ShareA_FFConvM
    public init(
        dim: Int,
        groupSize: Int = 256,
        queryKeyDim: Int = 128,
        expansionFactor: Float = 4.0,
        causal: Bool = false,
        dropout: Float = 0.1,
        rotaryPosEmb: Module? = nil,
        normKlass: Module.Type = LayerNorm.self,
        shiftTokens: Bool = true
    ) {
        self.dim = dim
        self.groupSize = groupSize
        self.queryKeyDim = queryKeyDim
        self.expansionFactor = expansionFactor
        self.causal = causal
        self.dropoutP = dropout
        self.shiftTokens = shiftTokens
        
        // Calculate hidden dimension
        self.hiddenDim = Int(Float(dim) * expansionFactor)
        
        // Initialize positional embeddings and dropout
        self.rotary_pos_emb = rotaryPosEmb
        self.dropout = Dropout(p: dropout)
        
        // Feed-forward layers
        self.to_hidden = FFConvM(
            dimIn: dim,
            dimOut: hiddenDim,
            normKlass: normKlass,
            dropout: dropout
        )
        self.to_qk = FFConvM(
            dimIn: dim,
            dimOut: queryKeyDim,
            normKlass: normKlass,
            dropout: dropout
        )
        
        // Offset and scale for query and key (4 heads: quad_q, lin_q, quad_k, lin_k)
        self.qk_offset_scale = OffsetScale(queryKeyDim, heads: 4)
        
        self.to_out = FFConvM(
            dimIn: dim * 2,
            dimOut: dim,
            normKlass: normKlass,
            dropout: dropout
        )
        
        super.init()
        
        // Compile the forward function for better performance
        self.forwardCompiled = MLX.compile(self.forwardCore)
    }
    
    /// Internal forward pass implementation
    /// - Parameters:
    ///   - x: Input tensor of shape (batch, seq_len, features)
    ///   - mask: Optional mask for attention
    /// - Returns: Output tensor after applying attention and projections
    private func forward(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        
        // Pre-normalization step
        let residual = x  // Save residual for skip connection
        
        // Token shifting if enabled
        let processedX: MLXArray
        if shiftTokens {
            // Split input into two halves
            let halfDim = x.shape[2] / 2
            let xShift = x[0..., 0..., 0..<halfDim]
            let xPass = x[0..., 0..., halfDim...]
            
            // Pad xShift: shift tokens by 1 position forward
            let seqLen = xShift.shape[1]
            var xShiftPadded: MLXArray
            
            if seqLen > 1 {
                // Create padding: (batch, 1, channels)
                let batchSize = xShift.shape[0]
                let channels = xShift.shape[2]
                let padding = MLXArray.zeros([batchSize, 1, channels])
                
                // Concatenate padding at the beginning and remove last timestep
                xShiftPadded = MLX.concatenated([padding, xShift[0..., 0..<(seqLen-1), 0...]], axis: 1)
            } else {
                // For single timestep, just zero it out
                xShiftPadded = MLXArray.zeros(xShift.shape)
            }
            
            // Concatenate shifted and unshifted parts
            processedX = MLX.concatenated([xShiftPadded, xPass], axis: -1)
        } else {
            processedX = x
        }
        
        // Initial projections
        let hiddenOutput = to_hidden(processedX)
        
        // Split into v and u
        let halfHiddenDim = hiddenOutput.shape[2] / 2
        let v = hiddenOutput[0..., 0..., 0..<halfHiddenDim]
        let u = hiddenOutput[0..., 0..., halfHiddenDim...]
        
        // to_qk projection
        
        let qk = to_qk(processedX)
        
        // Offset and scale - returns list of 4 tensors
        let qkOutputs = qk_offset_scale(qk)
        guard qkOutputs.count == 4 else {
            fatalError("Expected 4 outputs from qk_offset_scale, got \(qkOutputs.count)")
        }
        let quadQ = qkOutputs[0]
        let linQ = qkOutputs[1]
        let quadK = qkOutputs[2]
        let linK = qkOutputs[3]
        
        // Calculate attention
        // Use Metal kernel optimized version
        let (attV, attU) = calAttentionMetalKernel(
            x: x,
            quadQ: quadQ,
            linQ: linQ,
            quadK: quadK,
            linK: linK,
            v: v,
            u: u,
            mask: mask
        )
        
        // Original implementation (commented out)
        // let (attV, attU) = calAttention(
        //     x: x,
        //     quadQ: quadQ,
        //     linQ: linQ,
        //     quadK: quadK,
        //     linK: linK,
        //     v: v,
        //     u: u,
        //     mask: mask
        // )
        
        
        // Output calculation with gating
        // PyTorch: out = (att_u * v) * self.gateActivate(att_v * u)
        // gateActivate is nn.Sigmoid()
        let attU_v = attU * v
        let attV_u = attV * u
        let gate = MLX.sigmoid(attV_u)
        
        let out = attU_v * gate
        
        // Final projection and residual connection
        
        let finalOut = to_out(out)
        
        let result = residual + finalOut  // Residual connection
        
        return result
    }
    
    /// Core forward pass for compilation (no optional parameters)
    /// - Parameter x: Input tensor of shape (batch, seq_len, features)
    /// - Returns: Output tensor after applying attention and projections
    private func forwardCore(_ x: MLXArray) -> MLXArray {
        return forward(x, mask: nil)
    }
    
    /// Forward pass for FLASH layer (compiled for performance)
    /// - Parameters:
    ///   - x: Input tensor of shape (batch, seq_len, features)
    ///   - mask: Optional mask for attention
    /// - Returns: Output tensor after applying attention and projections
    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        if mask == nil {
            // Using compiled path
            return forwardCompiled(x)
        } else {
            // Using uncompiled path with mask
            return forward(x, mask: mask)
        }
    }
    
    /// Calculate attention output using quadratic and linear attention mechanisms.
    /// This is the original implementation without Metal kernel optimizations.
    private func calAttentionMetalKernel(
        x: MLXArray,
        quadQ: MLXArray,
        linQ: MLXArray,
        quadK: MLXArray,
        linK: MLXArray,
        v: MLXArray,
        u: MLXArray,
        mask: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let b = x.shape[0]
        let n = x.shape[1]
        let g = groupSize
        
        // Apply mask to linear keys if provided
        var maskedLinK = linK
        if let mask = mask {
            // Expand mask for broadcasting: (batch, seq_len) -> (batch, seq_len, 1)
            let linMask = mask.expandedDimensions(axis: -1)
            // Apply mask: set masked positions to 0
            maskedLinK = linK * linMask.asType(linK.dtype)
        }
        
        // Apply rotary positional embeddings if available
        var rotatedQuadQ = quadQ
        var rotatedLinQ = linQ
        var rotatedQuadK = quadK
        var rotatedLinK = maskedLinK
        
        if let rope = rotary_pos_emb {
            // Apply rotary embeddings
            if let ropeModule = rope as? RoPE {
                rotatedQuadQ = ropeModule(quadQ)
                rotatedLinQ = ropeModule(linQ)
                rotatedQuadK = ropeModule(quadK)
                rotatedLinK = ropeModule(maskedLinK)
            } else {
                fatalError("Unexpected rope type")
            }
        }
        
        // Padding for group processing
        let padding = paddingToMultipleOf(n: n, mult: g)
        
        var paddedQuadQ = rotatedQuadQ
        var paddedQuadK = rotatedQuadK
        var paddedLinQ = rotatedLinQ
        var paddedLinK = rotatedLinK
        var paddedV = v
        var paddedU = u
        var paddedMask = mask
        
        if padding > 0 {
            // Pad all tensors along sequence dimension
            let padWidth: [IntOrPair] = [IntOrPair((0, 0)), IntOrPair((0, padding)), IntOrPair((0, 0))]
            
            paddedQuadQ = MLX.padded(rotatedQuadQ, widths: padWidth)
            paddedQuadK = MLX.padded(rotatedQuadK, widths: padWidth)
            paddedLinQ = MLX.padded(rotatedLinQ, widths: padWidth)
            paddedLinK = MLX.padded(rotatedLinK, widths: padWidth)
            paddedV = MLX.padded(v, widths: padWidth)
            paddedU = MLX.padded(u, widths: padWidth)
            
            if mask != nil {
                // Create default mask: True for original positions, False for padding
                paddedMask = MLX.ones([b, n], dtype: .bool)
                paddedMask = MLX.padded(paddedMask!, widths: [IntOrPair((0, 0)), IntOrPair((0, padding))], value: MLXArray(false))
            }
        }
        
        // Group along sequence for attention
        let newSeqLen = paddedQuadQ.shape[1]
        let numGroups = newSeqLen / g
        
        func groupReshape(_ tensor: MLXArray) -> MLXArray {
            let batchSize = tensor.shape[0]
            let dim = tensor.shape[2]
            return tensor.reshaped([batchSize, numGroups, g, dim])
        }
        
        let groupedQuadQ = groupReshape(paddedQuadQ)
        let groupedQuadK = groupReshape(paddedQuadK)
        let groupedLinQ = groupReshape(paddedLinQ)
        let groupedLinK = groupReshape(paddedLinK)
        let groupedV = groupReshape(paddedV)
        let groupedU = groupReshape(paddedU)
        
        var groupedMask: MLXArray? = nil
        if paddedMask != nil {
            // Reshape mask: (batch, seq_len) -> (batch, num_groups, 1, group_size)
            groupedMask = paddedMask!.reshaped([b, numGroups, g])
            groupedMask = groupedMask!.expandedDimensions(axis: 2)
        }
        
        // Calculate quadratic attention output using optimized kernel
        let quadOutV = FlashAttentionImplementations.simpleKernel(groupedQuadQ, groupedQuadK, groupedV, g)
        let quadOutU = FlashAttentionImplementations.simpleKernel(groupedQuadQ, groupedQuadK, groupedU, g)
        
        // Calculate linear attention output
        var linOutV: MLXArray
        var linOutU: MLXArray
        
        if causal {
            // Causal linear attention with cumulative sum
            var linKV = MLX.matmul(groupedLinK.transposed(0, 1, 3, 2), groupedV) / Float(g)
            linKV = MLX.cumsum(linKV, axis: 1)
            linKV = MLX.padded(linKV, widths: [IntOrPair((0, 0)), IntOrPair((1, 0)), IntOrPair((0, 0)), IntOrPair((0, 0))])[0..., 0 ..< -1, 0..., 0...]
            linOutV = MLX.matmul(groupedLinQ, linKV)
            
            // Same for u
            var linKU = MLX.matmul(groupedLinK.transposed(0, 1, 3, 2), groupedU) / Float(g)
            linKU = MLX.cumsum(linKU, axis: 1)
            linKU = MLX.padded(linKU, widths: [IntOrPair((0, 0)), IntOrPair((1, 0)), IntOrPair((0, 0)), IntOrPair((0, 0))])[0..., 0 ..< -1, 0..., 0...]
            linOutU = MLX.matmul(groupedLinQ, linKU)
        } else {
            // Non-causal linear attention
            let batchSize = groupedLinK.shape[0]
            let totalSeq = groupedLinK.shape[1] * groupedLinK.shape[2]
            let queryKeyDim = groupedLinK.shape[3]
            let valueDim = groupedV.shape[3]
            
            // Reshape for non-causal attention
            let linKReshaped = groupedLinK.reshaped([batchSize, totalSeq, queryKeyDim])
            let vReshaped = groupedV.reshaped([batchSize, totalSeq, valueDim])
            
            // Compute k^T @ v
            let linKV = MLX.matmul(linKReshaped.transposed(0, 2, 1), vReshaped) / Float(n)
            
            // Compute linear attention output
            let linQReshaped = groupedLinQ.reshaped([batchSize, numGroups * g, queryKeyDim])
            let linOutVReshaped = MLX.matmul(linQReshaped, linKV)
            linOutV = linOutVReshaped.reshaped([batchSize, numGroups, g, valueDim])
            
            // Same for u
            let uReshaped = groupedU.reshaped([batchSize, totalSeq, valueDim])
            let linKU = MLX.matmul(linKReshaped.transposed(0, 2, 1), uReshaped) / Float(n)
            let linOutUReshaped = MLX.matmul(linQReshaped, linKU)
            linOutU = linOutUReshaped.reshaped([batchSize, numGroups, g, valueDim])
        }
        
        // Combine quadratic and linear attention outputs
        let combinedOutV = quadOutV + linOutV
        let combinedOutU = quadOutU + linOutU
        
        // Reshape back to original format and remove padding
        func ungroupReshape(_ tensor: MLXArray) -> MLXArray {
            let batchSize = tensor.shape[0]
            let numGroups = tensor.shape[1]
            let groupSize = tensor.shape[2]
            let dim = tensor.shape[3]
            return tensor.reshaped([batchSize, numGroups * groupSize, dim])
        }
        
        let finalOutV = ungroupReshape(combinedOutV)[0..., 0..<n, 0...]
        let finalOutU = ungroupReshape(combinedOutU)[0..., 0..<n, 0...]
        
        return (finalOutV, finalOutU)
    }
    
    // Original attention implementation
    private func calAttention(
        x: MLXArray,
        quadQ: MLXArray,
        linQ: MLXArray,
        quadK: MLXArray,
        linK: MLXArray,
        v: MLXArray,
        u: MLXArray,
        mask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let b = x.shape[0]
        let n = x.shape[1]
        let g = groupSize
        
        // Apply mask to linear keys if provided
        var maskedLinK = linK
        if let mask = mask {
            // Expand mask for broadcasting: (batch, seq_len) -> (batch, seq_len, 1)
            let linMask = mask.expandedDimensions(axis: -1)
            // Apply mask: set masked positions to 0
            maskedLinK = linK * linMask.asType(linK.dtype)
        }
        
        // Apply rotary positional embeddings if available
        var rotatedQuadQ = quadQ
        var rotatedLinQ = linQ
        var rotatedQuadK = quadK
        var rotatedLinK = maskedLinK
        
        
        if let rope = rotary_pos_emb {
            // Check if it's a custom implementation with rotate_queries_or_keys method
            // For now, we'll use direct call assuming it's mlx.nn.RoPE
            // Cast rope to proper type and call it
            if let ropeModule = rope as? RoPE {
                rotatedQuadQ = ropeModule(quadQ)
                rotatedLinQ = ropeModule(linQ)
                rotatedQuadK = ropeModule(quadK)
                rotatedLinK = ropeModule(maskedLinK)
            } else {
                fatalError("Unexpected rope type")
            }
        }
        
        
        // Padding for group processing
        let padding = paddingToMultipleOf(n: n, mult: g)
        
        var paddedQuadQ = rotatedQuadQ
        var paddedQuadK = rotatedQuadK
        var paddedLinQ = rotatedLinQ
        var paddedLinK = rotatedLinK
        var paddedV = v
        var paddedU = u
        var paddedMask = mask
        
        if padding > 0 {
            // Pad all tensors along sequence dimension
            let padWidth: [IntOrPair] = [IntOrPair((0, 0)), IntOrPair((0, padding)), IntOrPair((0, 0))]
            
            // Stack tensors for batch padding to reduce operations
            paddedQuadQ = MLX.padded(rotatedQuadQ, widths: padWidth)
            paddedQuadK = MLX.padded(rotatedQuadK, widths: padWidth)
            paddedLinQ = MLX.padded(rotatedLinQ, widths: padWidth)
            paddedLinK = MLX.padded(rotatedLinK, widths: padWidth)
            paddedV = MLX.padded(v, widths: padWidth)
            paddedU = MLX.padded(u, widths: padWidth)
            
            // Create padding mask efficiently
            let maskToUse: MLXArray
            if let existingMask = mask {
                maskToUse = existingMask
            } else {
                // Create default mask: True for original positions, False for padding
                maskToUse = MLXArray.ones([b, n], dtype: .bool)
            }
            paddedMask = MLX.padded(maskToUse, widths: [IntOrPair((0, 0)), IntOrPair((0, padding))], value: MLXArray(false))
        }
        
        // Group along sequence for attention
        let newSeqLen = paddedQuadQ.shape[1]
        let numGroups = newSeqLen / g
        
        // Reshape to group format: (batch, seq_len, dim) -> (batch, num_groups, group_size, dim)
        func groupReshape(_ tensor: MLXArray) -> MLXArray {
            let batchSize = tensor.shape[0]
            let dim = tensor.shape[2]
            return tensor.reshaped([batchSize, numGroups, g, dim])
        }
        
        let groupedQuadQ = groupReshape(paddedQuadQ)
        let groupedQuadK = groupReshape(paddedQuadK)
        let groupedLinQ = groupReshape(paddedLinQ)
        let groupedLinK = groupReshape(paddedLinK)
        let groupedV = groupReshape(paddedV)
        let groupedU = groupReshape(paddedU)

        var groupedMask: MLXArray?
        if let mask = paddedMask {
            // Reshape mask: (batch, seq_len) -> (batch, num_groups, 1, group_size)
            let reshapedMask = mask.reshaped([b, numGroups, g])
            groupedMask = reshapedMask.expandedDimensions(axis: 2)
        }
        
        // Calculate quadratic attention output
        let quadKTransposed = groupedQuadK.transposed(0, 1, 3, 2)
        let sim = MLX.matmul(groupedQuadQ, quadKTransposed) / Float(g)
        
        // ReLU², use MLX.square for better numerical stability (matches Python's optimized implementation)
        var attn = MLXNN.reluSquared(sim)
        
        // Apply mask if causal
        if causal {
            // Get cached causal mask or create new one
            let expandedMask: MLXArray
            if let cachedMask = Self.causalMaskCache[g] {
                expandedMask = cachedMask
            } else {
                // Create causal mask more efficiently
                let causalMask = MLX.tril(MLXArray.ones([g, g], dtype: .float32))
                // Expand for batch and groups - MLX will broadcast automatically
                expandedMask = causalMask.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
                // Cache it
                Self.causalMaskCache[g] = expandedMask
            }
            // Apply mask using where for better performance
            attn = MLX.where(expandedMask.asType(attn.dtype) .> 0, attn, MLXArray.zeros(attn.shape, dtype: attn.dtype))
        }
        
        if let mask = groupedMask {
            // Apply attention mask
            attn = attn * mask.asType(attn.dtype)
        }
        
        // Compute attention outputs
        let quadOutV = MLX.matmul(attn, groupedV)
        let quadOutU = MLX.matmul(attn, groupedU)
        
        // Calculate linear attention output
        var linOutV: MLXArray
        var linOutU: MLXArray
        
        if causal {
            // Causal linear attention with cumulative sum
            // lin_kv = mx.matmul(mx.transpose(lin_k, [0, 1, 3, 2]), v) / g
            var linKV = MLX.matmul(groupedLinK.transposed(0, 1, 3, 2), groupedV) / Float(g)
            // Cumulative sum over groups
            linKV = MLX.cumsum(linKV, axis: 1)
            // Pad and shift
            linKV = MLX.padded(linKV, widths: [IntOrPair((0, 0)), IntOrPair((1, 0)), IntOrPair((0, 0)), IntOrPair((0, 0))])[0..., 0..<(-1), 0..., 0...]
            // lin_out_v = mx.matmul(lin_q, lin_kv)
            linOutV = MLX.matmul(groupedLinQ, linKV)
            
            // Same for u
            var linKU = MLX.matmul(groupedLinK.transposed(0, 1, 3, 2), groupedU) / Float(g)
            linKU = MLX.cumsum(linKU, axis: 1)
            linKU = MLX.padded(linKU, widths: [IntOrPair((0, 0)), IntOrPair((1, 0)), IntOrPair((0, 0)), IntOrPair((0, 0))])[0..., 0..<(-1), 0..., 0...]
            linOutU = MLX.matmul(groupedLinQ, linKU)
        } else {
            // Non-causal linear attention
            let batchSize = groupedLinK.shape[0]
            let totalSeq = groupedLinK.shape[1] * groupedLinK.shape[2]
            let queryKeyDim = groupedLinK.shape[3]
            let valueDim = groupedV.shape[3]
            
            // Reshape lin_k: (batch, total_seq, query_key_dim)
            let linKReshaped = groupedLinK.reshaped([batchSize, totalSeq, queryKeyDim])
            // Reshape v: (batch, total_seq, value_dim)
            let vReshaped = groupedV.reshaped([batchSize, totalSeq, valueDim])
            
            // Compute k^T @ v: (batch, query_key_dim, value_dim)
            // Note: Division should be by original sequence length n, not padded length
            let linKV = MLX.matmul(linKReshaped.transposed(0, 2, 1), vReshaped) / Float(n)
            
            // Compute linear attention output
            let linQReshaped = groupedLinQ.reshaped([batchSize, numGroups * g, queryKeyDim])
            let linOutVReshaped = MLX.matmul(linQReshaped, linKV)
            linOutV = linOutVReshaped.reshaped([batchSize, numGroups, g, valueDim])
            
            // Same for u
            let uReshaped = groupedU.reshaped([batchSize, totalSeq, valueDim])
            let linKU = MLX.matmul(linKReshaped.transposed(0, 2, 1), uReshaped) / Float(n)
            let linOutUReshaped = MLX.matmul(linQReshaped, linKU)
            linOutU = linOutUReshaped.reshaped([batchSize, numGroups, g, valueDim])
        }
        
        // Combine quadratic and linear attention outputs
        let combinedOutV = quadOutV + linOutV
        let combinedOutU = quadOutU + linOutU
        
        // Reshape back to original format and remove padding
        func ungroupReshape(_ tensor: MLXArray) -> MLXArray {
            let batchSize = tensor.shape[0]
            let dim = tensor.shape[3]
            return tensor.reshaped([batchSize, numGroups * g, dim])
        }
        
        let finalOutV = ungroupReshape(combinedOutV)[0..., 0..<n, 0...]  // Remove padding
        let finalOutU = ungroupReshape(combinedOutU)[0..., 0..<n, 0...]  // Remove padding
        
        return (finalOutV, finalOutU)
    }
    
    /// Calculate padding needed to make n a multiple of mult
    private func paddingToMultipleOf(n: Int, mult: Int) -> Int {
        let remainder = n % mult
        if remainder == 0 {
            return 0
        }
        return mult - remainder
    }
}

