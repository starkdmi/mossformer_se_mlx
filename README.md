# MossFormer2 Speech Enhancement

Speaker enhancement models for extracting speech from noisy audio using MLX.

## Usage

### Python

```bash
pip install -r requirements.txt
python demo.py --input noisy.wav --output clean.wav --precision fp32
```

See [`Python/README.md`](python/README.md) and [`demo.py`](python/demo.py) for details.

### Swift

See [`MossFormer2Demos.swift`](swift/Tests/MossFormer2Demos.swift) for implementation details.

## Models

MLX weights: [starkdmi/MossFormer2_SE_48K_MLX](https://huggingface.co/starkdmi/MossFormer2_SE_48K_MLX)

Source: [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)

## License

See [LICENSE](LICENSE).
