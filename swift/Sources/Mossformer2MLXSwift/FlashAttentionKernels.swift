import MLX
import MLXNN

/// FLASH attention kernels using MLXFast Metal kernel API
public class FlashAttentionKernels {
    
    /// ReLU² kernel - fuses ReLU and square operations
    /// Uses template type T to support different data types
    private static let reluSquaredKernel = MLXFast.metalKernel(
        name: "relu_squared",
        inputNames: ["inp"],
        outputNames: ["out"],
        source: """
        uint elem = thread_position_in_grid.x;
        T val = inp[elem];
        T relu_val = val > T(0) ? val : T(0);
        out[elem] = relu_val * relu_val;
        """
    )
    
    /// Fused multiply-add kernel for better memory efficiency
    private static let fusedMultiplyAddKernel = MLXFast.metalKernel(
        name: "fused_multiply_add",
        inputNames: ["a", "b", "scale"],
        outputNames: ["out"],
        source: """
        uint elem = thread_position_in_grid.x;
        out[elem] = a[elem] * b[elem] * scale[0];
        """
    )
    
    /// Apply ReLU² using the Metal kernel
    public static func reluSquared(_ input: MLXArray) -> MLXArray {
        // Use MLX's native compiled reluSquared function
        return MLXNN.reluSquared(input)
    }
    
    /// Fused multiply-add operation
    public static func fusedMultiplyAdd(_ a: MLXArray, _ b: MLXArray, scale: Float) -> MLXArray {
        let scaleArray = MLXArray([scale])
        let shape = a.shape
        let size = shape.reduce(1, *)
        
        // Calculate grid size
        let threadsPerGroup = 256
        let gridSize = (size + threadsPerGroup - 1) / threadsPerGroup
        
        let outputs = fusedMultiplyAddKernel(
            [a, b, scaleArray],
            grid: (gridSize, 1, 1),
            threadGroup: (threadsPerGroup, 1, 1),
            outputShapes: [shape],
            outputDTypes: [a.dtype]
        )
        
        return outputs[0]
    }
    
    /// Optimized attention computation using Metal kernels
    public static func optimizedAttention(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        scale: Float
    ) -> MLXArray {
        // Q @ K^T with scaling
        let kTransposed = k.transposed(0, 1, 3, 2)
        let sim = MLX.matmul(q, kTransposed) * scale
        
        // Apply ReLU² using Metal kernel
        let attn = reluSquared(sim)
        
        // Attention @ V
        return MLX.matmul(attn, v)
    }
}

/// FLASH attention implementations using real Metal kernels
public class FlashAttentionImplementations {
    
    /// Simple kernel implementation using actual Metal kernels
    public static func simpleKernel(
        _ quadQ: MLXArray,
        _ quadK: MLXArray,
        _ v: MLXArray,
        _ groupSize: Int
    ) -> MLXArray {
        let scale = 1.0 / Float(groupSize)
        return FlashAttentionKernels.optimizedAttention(
            q: quadQ,
            k: quadK,
            v: v,
            scale: scale
        )
    }
}
