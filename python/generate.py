import os
import time
import argparse
import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from mossformer2_se_wrapper import MossFormer2_SE_48K
from stft import stft as stft_func, ISTFTCache, create_window
from fbank import compute_fbank
from deltas import compute_deltas

# Constants
MAX_WAV_VALUE = 32768.0
MODEL_REPO = "starkdmi/MossFormer2_SE_48K_MLX"

# Model configuration
MODEL_CONFIG = argparse.Namespace(
    sampling_rate=48000,
    win_len=1920,
    win_inc=384,
    fft_len=1920,
    num_mels=60,
    win_type='hamming',
    one_time_decode_length=20,
    decode_window=4,
    preemphasis=0.97
)

# Global caches for optimal performance
istft_cache = ISTFTCache()


def load_model(precision="fp32"):
    """Load MossFormer2 model from HuggingFace Hub.
    
    Args:
        precision: One of "fp32", "fp16", "int8", "int6", "int4"
    """
    print(f"Loading MossFormer2 SE 48K model ({precision})...")

    # Enable fast LayerNorm optimization
    nn.LayerNorm.__call__ = lambda self, x: mx.fast.layer_norm(x, self.weight, self.bias, self.eps)

    # Initialize model
    model_wrapper = MossFormer2_SE_48K(MODEL_CONFIG)
    
    # Check if this is a quantized precision
    quantized_precisions = {"int4": 4, "int6": 6, "int8": 8}
    is_quantized = precision in quantized_precisions
    
    if is_quantized:
        bits = quantized_precisions[precision]
        model_file = f"model_{precision}.safetensors"
        weights_path = hf_hub_download(repo_id=MODEL_REPO, filename=model_file)
        
        # Quantize the model structure before loading weights
        nn.quantize(model_wrapper, group_size=64, bits=bits)
    else:
        # Standard fp16/fp32 loading
        model_file = f"model_{precision}.safetensors"
        weights_path = hf_hub_download(repo_id=MODEL_REPO, filename=model_file)

    weights = mx.load(weights_path)
    model_wrapper.update(tree_unflatten(list(weights.items())))

    total_params = sum(v.size for v in weights.values() if hasattr(v, 'size'))
    print(f"Model loaded: {total_params:,} parameters")

    return model_wrapper.model


def compute_fbank_optimized(audio_input, config):
    """Compute mel-filterbank features"""
    if hasattr(audio_input, 'numpy'):
        audio_np = audio_input.numpy()
    elif isinstance(audio_input, mx.array):
        audio_np = np.array(audio_input)
    else:
        audio_np = audio_input
    return compute_fbank(audio_np, config)


def optimized_compute_deltas(features: mx.array) -> mx.array:
    """Compute delta features"""
    return compute_deltas(features, win_length=5)


def decode_one_audio(model, inputs, config):
    """
    Speech enhancement inference with optimized processing pipeline
    """
    # Convert to numpy
    if hasattr(inputs, 'numpy'):
        inputs_np = inputs.numpy()
    else:
        inputs_np = inputs

    if inputs_np.ndim == 2:
        inputs_np = inputs_np[0, :]

    input_len = inputs_np.shape[0]
    original_len = input_len
    inputs_np = inputs_np * MAX_WAV_VALUE

    window = create_window(config.win_type, config.win_len, periodic=False)

    # Check if segmented processing is needed for long audio
    if input_len > config.sampling_rate * config.one_time_decode_length:
        print(f"  Using segmented processing for {input_len / config.sampling_rate:.1f}s audio")

        window_size = int(config.sampling_rate * config.decode_window)
        stride = int(window_size * 0.75)
        t = inputs_np.shape[0]

        # Pad input if necessary
        if t < window_size:
            inputs_np = np.concatenate([inputs_np, np.zeros(window_size - t)], 0)
        elif t < window_size + stride:
            padding = window_size + stride - t
            inputs_np = np.concatenate([inputs_np, np.zeros(padding)], 0)
        else:
            if (t - window_size) % stride != 0:
                padding = t - (t - window_size) // stride * stride
                inputs_np = np.concatenate([inputs_np, np.zeros(padding)], 0)

        audio = mx.array(inputs_np)
        t = audio.shape[0]
        output_segments = []
        output_ranges = []
        give_up_length = (window_size - stride) // 2
        current_idx = 0

        # Process each segment
        while current_idx + window_size <= t:
            audio_segment = audio[current_idx:current_idx + window_size]

            # Feature extraction
            fbanks = compute_fbank_optimized(audio_segment, config)
            fbank_transposed = mx.transpose(fbanks, [1, 0])
            fbank_delta = optimized_compute_deltas(fbank_transposed)
            fbank_delta_delta = optimized_compute_deltas(fbank_delta)
            fbank_delta = mx.transpose(fbank_delta, [1, 0])
            fbank_delta_delta = mx.transpose(fbank_delta_delta, [1, 0])
            fbanks = mx.concatenate([fbanks, fbank_delta, fbank_delta_delta], axis=1)
            fbanks = mx.expand_dims(fbanks, axis=0)

            # Model inference
            Out_List = model(fbanks)
            pred_mask = Out_List[-1][0]

            # STFT
            real_part, imag_part = stft_func(
                audio_segment.reshape(1, -1),
                config.fft_len, config.win_inc, config.win_len,
                window, center=False
            )

            # Apply mask
            pred_mask = mx.transpose(pred_mask, [1, 0])
            pred_mask = mx.expand_dims(pred_mask, axis=-1)
            spectrum_real = real_part[0] * pred_mask[:, :, 0]
            spectrum_imag = imag_part[0] * pred_mask[:, :, 0]

            # iSTFT
            output_segment = istft_cache.istft(
                spectrum_real.reshape(1, *spectrum_real.shape),
                spectrum_imag.reshape(1, *spectrum_imag.shape),
                config.fft_len, config.win_inc, config.win_len,
                window, center=False, audio_length=len(audio_segment)
            )
            mx.eval(output_segment)

            # Store segment with overlap handling
            if current_idx == 0:
                output_segments.append(output_segment[0][:-give_up_length])
                output_ranges.append((current_idx, current_idx + window_size - give_up_length))
            else:
                output_segments.append(output_segment[0][give_up_length:-give_up_length])
                output_ranges.append((current_idx + give_up_length, current_idx + window_size - give_up_length))

            current_idx += stride

        # Reconstruct full output
        outputs_np = np.zeros(t)
        for segment, (start, end) in zip(output_segments, output_ranges):
            segment_np = np.array(segment)
            outputs_np[start:end] = segment_np

        outputs_np = outputs_np[:original_len]
        outputs = mx.array(outputs_np)

    else:
        # Process entire audio at once
        audio = mx.array(inputs_np)

        # Feature extraction
        fbanks = compute_fbank_optimized(audio, config)
        fbank_transposed = mx.transpose(fbanks, [1, 0])
        fbank_delta = optimized_compute_deltas(fbank_transposed)
        fbank_delta_delta = optimized_compute_deltas(fbank_delta)
        fbank_delta = mx.transpose(fbank_delta, [1, 0])
        fbank_delta_delta = mx.transpose(fbank_delta_delta, [1, 0])
        fbanks = mx.concatenate([fbanks, fbank_delta, fbank_delta_delta], axis=1)
        fbanks = mx.expand_dims(fbanks, axis=0)
        mx.eval(fbanks)

        # Model inference
        Out_List = model(fbanks)
        pred_mask = Out_List[-1][0]
        mx.eval(pred_mask)

        # STFT
        real_part, imag_part = stft_func(
            audio.reshape(1, -1),
            config.fft_len, config.win_inc, config.win_len,
            window, center=False
        )
        mx.eval(real_part, imag_part)

        # Apply mask
        pred_mask = mx.transpose(pred_mask, [1, 0])
        pred_mask = mx.expand_dims(pred_mask, axis=-1)
        spectrum_real = real_part[0] * pred_mask[:, :, 0]
        spectrum_imag = imag_part[0] * pred_mask[:, :, 0]
        mx.eval(spectrum_real, spectrum_imag)

        # iSTFT
        outputs = istft_cache.istft(
            spectrum_real.reshape(1, *spectrum_real.shape),
            spectrum_imag.reshape(1, *spectrum_imag.shape),
            config.fft_len, config.win_inc, config.win_len,
            window, center=False, audio_length=len(audio)
        )
        mx.eval(outputs)
        outputs = outputs[0]

    return np.array(outputs) / MAX_WAV_VALUE


def warmup_model(model, config):
    """
    Warm up model and MLX operations for optimal performance.
    First inference is typically 2-3x slower without warmup.
    """
    print("Warming up model...")
    warmup_start = time.time()

    # Create small random audio (0.2 seconds)
    warmup_samples = 9600
    warmup_audio = mx.random.uniform(-0.1, 0.1, shape=(1, warmup_samples))

    # Run full inference pipeline to compile all operations
    _ = decode_one_audio(model, warmup_audio, config)

    warmup_time = time.time() - warmup_start
    print(f"Warmup complete: {warmup_time:.2f}s\n")


def process_chunk(model, audio_segment, config, window, chunk_length):
    """Process a single audio chunk through the model."""
    # Feature extraction
    fbanks = compute_fbank_optimized(audio_segment, config)
    fbank_transposed = mx.transpose(fbanks, [1, 0])
    fbank_delta = optimized_compute_deltas(fbank_transposed)
    fbank_delta_delta = optimized_compute_deltas(fbank_delta)
    fbank_delta = mx.transpose(fbank_delta, [1, 0])
    fbank_delta_delta = mx.transpose(fbank_delta_delta, [1, 0])
    fbanks = mx.concatenate([fbanks, fbank_delta, fbank_delta_delta], axis=1)
    fbanks = mx.expand_dims(fbanks, axis=0)
    
    # Model inference
    Out_List = model(fbanks)
    pred_mask = Out_List[-1][0]
    
    # STFT
    real_part, imag_part = stft_func(
        audio_segment.reshape(1, -1),
        config.fft_len, config.win_inc, config.win_len,
        window, center=False
    )
    
    # Apply mask
    pred_mask = mx.transpose(pred_mask, [1, 0])
    pred_mask = mx.expand_dims(pred_mask, axis=-1)
    spectrum_real = real_part[0] * pred_mask[:, :, 0]
    spectrum_imag = imag_part[0] * pred_mask[:, :, 0]
    
    # iSTFT
    output_segment = istft_cache.istft(
        spectrum_real.reshape(1, *spectrum_real.shape),
        spectrum_imag.reshape(1, *spectrum_imag.shape),
        config.fft_len, config.win_inc, config.win_len,
        window, center=False, audio_length=chunk_length
    )
    mx.eval(output_segment)
    
    return np.array(output_segment[0])


def decode_one_audio_chunked(model, inputs, config, chunk_seconds=4.0, overlap=0.25):
    """
    Chunked speech enhancement with discard-edges reassembly.
    
    Uses sequential processing (batch=1) which is memory-efficient
    and nearly as fast as batched processing for this model size.
    
    Args:
        model: The model to use for inference
        inputs: Input audio array
        config: Model configuration
        chunk_seconds: Chunk duration in seconds (default: 4.0)
        overlap: Overlap ratio between chunks (default: 0.25)
    
    Returns:
        Enhanced audio as numpy array
    """
    # Convert to numpy
    if hasattr(inputs, 'numpy'):
        inputs_np = inputs.numpy()
    else:
        inputs_np = inputs

    if inputs_np.ndim == 2:
        inputs_np = inputs_np[0, :]

    original_len = inputs_np.shape[0]
    inputs_np = inputs_np * MAX_WAV_VALUE

    window = create_window(config.win_type, config.win_len, periodic=False)
    
    # Calculate chunk parameters
    chunk_samples = int(config.sampling_rate * chunk_seconds)
    overlap_samples = int(chunk_samples * overlap)
    stride = chunk_samples - overlap_samples
    give_up = overlap_samples // 2
    
    # For short audio, process directly without chunking
    if original_len <= chunk_samples:
        audio = mx.array(inputs_np)
        result = process_chunk(model, audio, config, window, original_len)
        return result / MAX_WAV_VALUE
    
    # Count chunks
    num_full_chunks = (original_len - chunk_samples) // stride + 1
    remaining = original_len - (num_full_chunks - 1) * stride - chunk_samples
    has_partial = remaining > 0
    
    print(f"  Chunked: {num_full_chunks} x {chunk_seconds}s" + 
          (f" + partial" if has_partial else ""))
    
    # Process chunks sequentially
    chunks = []
    chunk_starts = []
    current_idx = 0
    
    while current_idx + chunk_samples <= original_len:
        audio_segment = mx.array(inputs_np[current_idx:current_idx + chunk_samples])
        chunk_result = process_chunk(model, audio_segment, config, window, chunk_samples)
        chunks.append(chunk_result)
        chunk_starts.append(current_idx)
        current_idx += stride
    
    # Handle last partial chunk if any
    if current_idx < original_len:
        remaining = original_len - current_idx
        audio_segment = mx.array(inputs_np[current_idx:])
        chunk_result = process_chunk(model, audio_segment, config, window, remaining)
        chunks.append(chunk_result)
        chunk_starts.append(current_idx)
    
    # Reassemble using discard-edges strategy (vectorized)
    output = np.zeros(original_len)
    num_chunks = len(chunks)
    
    for idx, (chunk, start_idx) in enumerate(zip(chunks, chunk_starts)):
        chunk_len = len(chunk)
        is_first = idx == 0
        is_last = idx == num_chunks - 1
        
        # Calculate what range of the chunk to keep
        if is_last and chunk_len < chunk_samples:
            keep_start = give_up if not is_first else 0
            keep_end = chunk_len
        else:
            keep_start = 0 if is_first else give_up
            keep_end = chunk_len - give_up
        
        # Copy to output (vectorized)
        output_start = start_idx + keep_start
        output_end = min(start_idx + keep_end, original_len)
        chunk_slice = chunk[keep_start:keep_start + (output_end - output_start)]
        output[output_start:output_end] = chunk_slice
    
    return output / MAX_WAV_VALUE


def enhance_audio(model, audio_path, config, chunked=None):
    """Load and enhance audio file
    
    Args:
        model: The model to use
        audio_path: Path to input audio file
        config: Model configuration
        chunked: Force chunked processing mode. If None, auto-selects based on duration:
                 - < 60s: Full mode (faster, best quality)
                 - >= 60s: Chunked mode (4s chunks, 25% overlap, lower RAM)
    """
    # Load audio
    audio_np, sr = sf.read(audio_path, dtype='float32')
    
    duration = len(audio_np) / sr
    print(f"  Input: {audio_path}")
    print(f"  Sample rate: {sr} Hz, Duration: {duration:.2f}s")

    # Ensure correct shape
    if audio_np.ndim == 1:
        audio_np = audio_np.reshape(1, -1)
    elif audio_np.ndim == 2:
        audio_np = audio_np.T
        audio_np = audio_np[0:1]

    # Resample if needed
    if sr != 48000:
        print(f"  Resampling from {sr} Hz to 48000 Hz...")
        from scipy import signal
        resample_factor = 48000 / sr
        num_samples = int(audio_np.shape[1] * resample_factor)
        audio_np = signal.resample(audio_np, num_samples, axis=1)
        duration = num_samples / 48000

    # Auto-select mode based on duration (60s threshold from MossFormer2 paper)
    use_chunked = chunked if chunked is not None else (duration >= 60)
    
    # Perform enhancement
    if use_chunked:
        enhanced_audio = decode_one_audio_chunked(model, audio_np, config)
    else:
        enhanced_audio = decode_one_audio(model, audio_np, config)

    return enhanced_audio


def main():
    parser = argparse.ArgumentParser(
        description="MossFormer2 Speech Enhancement for MLX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python generate.py --input noisy.wav --output clean.wav
  python generate.py --input noisy.wav --output clean.wav --precision fp16
  python generate.py --input noisy.wav --output clean.wav --precision int4
  python generate.py --input noisy.wav --output clean.wav --chunked
        """
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input audio file path")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output audio file path")
    parser.add_argument("--precision", "-p", type=str, default="fp32",
                        choices=["fp16", "fp32", "int4", "int6", "int8"],
                        help="Model precision (default: fp32)")
    parser.add_argument("--chunked", action="store_true",
                        help="Force chunked processing (auto-enabled for 60s+ audio)")
    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    try:
        print("=" * 60)
        print("MossFormer2 Speech Enhancement - MLX")
        print("=" * 60)

        # Load model
        load_start = time.time()
        model = load_model(args.precision)
        load_time = time.time() - load_start
        print(f"Model loading time: {load_time:.2f}s\n")

        # Warm up model (optional)
        warmup_model(model, MODEL_CONFIG)

        # Process audio
        print("Processing audio...")

        # Clean run
        # gc.collect()
        # mx.clear_cache()
        # Benchmark begin
        mx.reset_peak_memory()
        process_start = time.time()

        enhanced_audio = enhance_audio(model, args.input, MODEL_CONFIG, chunked=args.chunked or None)

        # Benchmark finish
        process_time = time.time() - process_start
        peak_mem_bytes = mx.get_peak_memory()

        # Save output
        if enhanced_audio.ndim == 2 and enhanced_audio.shape[0] == 1:
            enhanced_audio = enhanced_audio[0]

        sf.write(args.output, enhanced_audio, 48000)

        # Print summary
        audio_duration = enhanced_audio.shape[0] / 48000
        rtf = audio_duration / process_time
        peak_mem_mb = peak_mem_bytes / (1024 * 1024)

        print("\n" + "=" * 60)
        print("Processing complete!")
        print("=" * 60)
        print(f"Audio duration: {audio_duration:.2f}s")
        print(f"Processing time: {process_time:.2f}s")
        print(f"Real-time factor: {rtf:.2f}x")
        print(f"Peak Memory: {peak_mem_mb:.2f} MB")
        print("-" * 60)
        print(f"Output saved to: {args.output}")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
