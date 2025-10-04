# MossFormer2 Speech Enhancement

Speaker enhancement models for extracting speech from noisy audio using MLX.

## Usage

### Python

```bash
cd python
pip install -r requirements.txt
python demo.py --input noisy.wav --output clean.wav --precision fp32
```

### Swift

```bash
cd swift
xcodebuild build -scheme generate -configuration Release -destination 'platform=macOS' -derivedDataPath .build/DerivedData -quiet
.build/DerivedData/Build/Products/Release/generate noisy.wav --precision fp32
```

## Models

MLX weights: [starkdmi/MossFormer2_SE_48K_MLX](https://huggingface.co/starkdmi/MossFormer2_SE_48K_MLX)

Source: [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)

## License

See [LICENSE](LICENSE).
