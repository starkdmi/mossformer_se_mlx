import MLX
import MLXFast

enum DepthwiseConv1d {
    
    /// Custom depthwise 1D convolution kernel specialized for stride 1 and groups == channels.
    private static let kernel = MLXFast.metalKernel(
        name: "custom_kernel_depthwise_conv1d",
        inputNames: ["inp", "weight", "params"],
        outputNames: ["out"],
        source: """
        const int chan = int(thread_position_in_grid.x);
        const int time = int(thread_position_in_grid.y);
        const int batch = int(thread_position_in_grid.z);

        const int B = params[0];
        const int L = params[1];
        const int C = params[2];
        const int K = params[3];
        const int pad = params[4];

        if (batch >= B || time >= L || chan >= C) {
            return;
        }

        const int out_index = ((batch * L) + time) * C + chan;
        const int weight_base = chan * K;

        T acc = T(0);
        for (int k = 0; k < K; ++k) {
            const int in_time = time + k - pad;
            if (in_time < 0 || in_time >= L) {
                continue;
            }
            const int in_index = ((batch * L) + in_time) * C + chan;
            acc += inp[in_index] * weight[weight_base + k];
        }

        out[out_index] = acc;
        """
    )
    
    /// Applies a depthwise 1-D convolution, using the custom Metal kernel when supported.
    ///
    /// - Parameters:
    ///   - x: Input tensor of shape [B, T, C].
    ///   - weight: Weight tensor with shape [C, K, 1] for depthwise convolutions.
    ///   - stride: Convolution stride (only stride == 1 uses the custom kernel).
    ///   - padding: Symmetric padding applied to both sides of the sequence.
    ///   - groups: Number of groups to pass to the fallback path.
    ///   - stream: Optional stream.
    /// - Returns: Output tensor of shape [B, T, C].
    static func apply(
        _ x: MLXArray,
        weight: MLXArray,
        stride: Int,
        padding: Int,
        groups: Int,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        guard stride == 1,
              x.ndim == 3,
              weight.shape.count == 3,
              weight.shape[2] == 1,
              weight.shape[0] == x.shape[2],
              (x.dtype == .float32 || x.dtype == .float16) else {
            return MLX.conv1d(
                x,
                weight,
                stride: stride,
                padding: padding,
                groups: groups,
                stream: stream
            )
        }
        
        let xContiguous = MLX.contiguous(x, allowColMajor: false, stream: stream)
        let weightContiguous = MLX.contiguous(weight, allowColMajor: false, stream: stream)
        let kernelSize = weightContiguous.shape[1]
        
        // This specialized kernel assumes "same" padding, i.e. padding * 2 == kernelSize - 1,
        // and will otherwise fall back to the generic implementation.
        guard padding * 2 == kernelSize - 1 else {
            return MLX.conv1d(
                xContiguous,
                weightContiguous,
                stride: stride,
                padding: padding,
                groups: groups,
                stream: stream
            )
        }
        
        let shape = xContiguous.shape  // [B, T, C]
        let params = MLXArray([
            Int32(shape[0]),
            Int32(shape[1]),
            Int32(shape[2]),
            Int32(kernelSize),
            Int32(padding)
        ])
        
        let outputs = kernel(
            [xContiguous, weightContiguous, params],
            template: [("T", xContiguous.dtype)],
            grid: (shape[2], shape[1], shape[0]),
            threadGroup: (1, 1, 1),
            outputShapes: [shape.map { Int($0) }],
            outputDTypes: [xContiguous.dtype],
            stream: stream
        )
        
        guard let result = outputs.first else {
            return MLX.conv1d(
                x,
                weight,
                stride: stride,
                padding: padding,
                groups: groups,
                stream: stream
            )
        }
        
        return result
    }
}
