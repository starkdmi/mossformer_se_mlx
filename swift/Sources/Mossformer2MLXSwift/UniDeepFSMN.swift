import Foundation
import MLX
import MLXNN

/// MLX Swift implementation of UniDeepFsmn with mathematical equivalence to Python MLX.
///
/// UniDeepFsmn is a neural network module that implements a single-deep feedforward
/// sequence memory network (FSMN) for temporal sequence modeling.
///
/// Args:
///     inputDim (int): Dimension of the input features
///     outputDim (int): Dimension of the output features
///     lorder (int): Length order for convolution layers (memory span)
///     hiddenSize (int): Number of hidden units in the linear layer
///
/// Inputs: input
///     - **input** (batch, time, inputDim): Tensor containing input sequences (MLX format)
///
/// Returns: output
///     - **output** (batch, time, outputDim): Enhanced tensor with temporal memory
public class UniDeepFSMN: Module {
    public let inputDim: Int
    public let outputDim: Int
    public let lorder: Int?
    public let hiddenSize: Int?
    public let kernelSize: (Int, Int)?
    
    // Layers
    @ModuleInfo var linear: Linear
    @ModuleInfo var project: Linear
    @ModuleInfo var conv1: Conv2d
    
    /// Initialize UniDeepFsmn
    /// - Parameters:
    ///   - inputDim: Dimension of the input features
    ///   - outputDim: Dimension of the output features
    ///   - lorder: Length order for convolution layers (memory span)
    ///   - hiddenSize: Number of hidden units in the linear layer
    public init(
        inputDim: Int,
        outputDim: Int,
        lorder: Int? = nil,
        hiddenSize: Int? = nil
    ) {
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.lorder = lorder
        
        guard let lorder = lorder else {
            fatalError("UniDeepFSMN requires lorder parameter")
        }
        
        self.hiddenSize = hiddenSize ?? outputDim
        
        // Initialize layers (matching PyTorch architecture)
        self.linear = Linear(inputDim, self.hiddenSize!)
        self.project = Linear(self.hiddenSize!, outputDim, bias: false)
        
        // Conv matching PyTorch exactly
        // The actual operation is a 1D convolution over time dimension
        // PyTorch uses Conv2d with kernel (39, 1) but it's really a 1D operation
        let kernelSize = lorder + lorder - 1  // 39
        self.kernelSize = (kernelSize, 1)
        
        // Use MLX's Conv2d to match Python implementation exactly
        // MLX expects NHWC format while PyTorch uses NCHW
        self.conv1 = Conv2d(
            inputChannels: outputDim,  // 256
            outputChannels: outputDim,  // 256
            kernelSize: IntOrPair((kernelSize, 1)),  // (39, 1)
            stride: IntOrPair((1, 1)),
            padding: IntOrPair((0, 0)),  // We'll pad manually
            groups: outputDim,  // Depthwise convolution
            bias: false
        )
        
        super.init()
    }
    
    /// Forward pass for the UniDeepFsmn model
    /// - Parameter inputTensor: Input tensor of shape (batch, time, inputDim)
    /// - Returns: Output tensor of shape (batch, time, outputDim)
    public func callAsFunction(_ inputTensor: MLXArray) -> MLXArray {
        guard let lorder = lorder else {
            fatalError("UniDeepFSMN not properly initialized")
        }
        
        // let batchSize = inputTensor.shape[0]
        // let timeSteps = inputTensor.shape[1]
        // let inputDim = inputTensor.shape[2]
        
        // 1. Linear transformation with ReLU activation
        let f1 = MLXNN.relu(linear(inputTensor))  // ReLU using MLX's native compiled function
        
        // 2. Project to output dimension
        let p1 = project(f1)  // Shape: (batch, time, outputDim)
        
        // 3. Apply grouped convolution
        // Match Python exactly: uses Conv2d with kernel (39, 1)
        
        // Add width dimension to match PyTorch's 4D tensor
        // Python: (batch, time, outputDim) -> (batch, time, 1, outputDim)
        let x = MLX.expandedDimensions(p1, axis: 2)  // Shape: (batch, time, 1, outputDim)
        
        // Causal padding: pad with (lorder-1) on both sides for the time dimension
        let paddingLeft = lorder - 1
        let paddingRight = lorder - 1
        
        // Pad the H dimension (axis=1) which is the time dimension in our NHWC format
        let y = MLX.padded(x, widths: [IntOrPair((0, 0)), IntOrPair((paddingLeft, paddingRight)), IntOrPair((0, 0)), IntOrPair((0, 0))])
        
        // Apply Conv2d (MLX expects NHWC)
        let out = conv1(y)
        
        // Add residual connection: x + conv_output
        let outWithResidual = x + out
        
        // Remove the width dimension to return to 3D
        let enhancedFeatures = MLX.squeezed(outWithResidual, axis: 2)  // Shape: (batch, time, outputDim)
        
        // 5. Final residual connection with original input
        // This only works if inputDim == outputDim, as in the PyTorch version
        if inputDim == outputDim {
            let result = inputTensor + enhancedFeatures
            return result
        } else {
            // If dimensions don't match, return enhanced features only
            return enhancedFeatures
        }
    }
}
