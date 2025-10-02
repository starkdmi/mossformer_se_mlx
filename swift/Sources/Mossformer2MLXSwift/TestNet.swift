import Foundation
import MLX
import MLXNN

/// The TestNet class for testing the MossFormer MaskNet implementation in MLX.
///
/// This class builds a model that integrates the MossFormer_MaskNet_MLX
/// for processing input audio and generating masks for source separation.
public class TestNet: Module {
    public let nLayers: Int
    
    // MossFormer MaskNet instance
    public var mossformer: MossFormerMaskNet
    
    // Compiled forward function for better performance
    private var forwardCompiled: ((MLXArray) -> MLXArray)!
    
    /// Initialize TestNet
    /// - Parameter nLayers: The number of layers in the model (currently unused, kept for compatibility)
    public init(nLayers: Int = 18) {
        self.nLayers = nLayers
        
        // Initialize the MossFormer MaskNet with specified input and output channels
        self.mossformer = MossFormerMaskNet(
            inChannels: 180,
            outChannels: 512,
            outChannelsFinal: 961
        )
        
        super.init()
        
        // Compile the forward function for better performance
        self.forwardCompiled = MLX.compile(self.forwardCore)
    }
    
    /// Internal forward pass implementation
    /// - Parameter input: Input tensor of dimension [B, N, S], where B is the batch size,
    ///                    N is the number of input channels (180), and S is the sequence length
    /// - Returns: List containing the mask tensor predicted by the MossFormer_MaskNet
    private func forward(_ input: MLXArray) -> [MLXArray] {
        var outList: [MLXArray] = []
        
        
        // Transpose input to match expected shape for MaskNet
        // Input is [B, time, channels] but MaskNet expects [B, channels, time]
        let x = input.transposed(0, 2, 1)  // Change shape from [B, T, C] to [B, C, T]
        
        // Get the mask from the MossFormer MaskNet
        var mask = mossformer(x)  // Forward pass through the MossFormer_MaskNet
        
        // Stop gradient for inference mode (no backprop needed)
        mask = MLX.stopGradient(mask)
        outList.append(mask)  // Append the mask to the output list
        
        return outList  // Return the list containing the mask
    }
    
    /// Core forward pass for compilation (returns single mask)
    /// - Parameter input: Input tensor of dimension [B, N, S], where B is the batch size,
    ///                    N is the number of input channels (180), and S is the sequence length
    /// - Returns: Mask tensor predicted by the MossFormer_MaskNet
    private func forwardCore(_ input: MLXArray) -> MLXArray {
        // Transpose input to match expected shape for MaskNet
        // Input is [B, time, channels] but MaskNet expects [B, channels, time]
        let x = input.transposed(0, 2, 1)  // Change shape from [B, T, C] to [B, C, T]
        
        // Get the mask from the MossFormer MaskNet
        var mask = mossformer(x)  // Forward pass through the MossFormer_MaskNet
        
        // Stop gradient for inference mode (no backprop needed)
        mask = MLX.stopGradient(mask)
        
        return mask  // Return the mask directly
    }
    
    /// Forward pass through the TestNet model (compiled for performance)
    /// - Parameter input: Input tensor of dimension [B, N, S], where B is the batch size,
    ///                    N is the number of input channels (180), and S is the sequence length
    /// - Returns: List containing the mask tensor predicted by the MossFormer_MaskNet
    public func callAsFunction(_ input: MLXArray) -> [MLXArray] {
        let mask = forwardCompiled(input)
        return [mask]  // Wrap in array for compatibility
    }
}

/// The MossFormer2_SE_48K model for speech enhancement in MLX.
///
/// This class encapsulates the functionality of the MossFormer MaskNet
/// within a higher-level model. It processes input audio data to produce
/// enhanced outputs and corresponding masks.
public class MossFormer2SE48K: Module {
    // TestNet model instance
    public var model: TestNet
    
    /// Initialize MossFormer2_SE_48K
    /// - Parameter args: Configuration arguments (currently unused but kept for future flexibility)
    public init(args: PipelineConfiguration.Args? = nil) {
        // Initialize the TestNet model, which contains the MossFormer MaskNet
        self.model = TestNet()
        
        super.init()
    }
    
    /// Forward pass through the model
    /// - Parameter x: Input tensor of dimension [B, N, S], where B is the batch size,
    ///                N is the number of channels (180 in this case), and S is the
    ///                sequence length (e.g., time frames)
    /// - Returns: List containing the mask tensor predicted by the model for speech separation
    public func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        // Get outputs from TestNet (returns a list with one mask)
        let outList = model(x)
        
        // Return the list to match PyTorch behavior
        return outList
    }
}