# MossFormer2 Speech Enhancement

Speaker enhancement models for extracting speech from noisy audio using MLX.

## Usage

### Python

```bash
cd python
pip install -r requirements.txt
python generate.py --input noisy.wav --output clean.wav --precision fp32
```

### Swift

```bash
cd swift
xcodebuild build -scheme generate -configuration Release -destination 'platform=macOS' -derivedDataPath .build/DerivedData -quiet
.build/DerivedData/Build/Products/Release/generate noisy.wav --precision fp32
```

## Performance

| Framework  | Speed (× faster than input) |
| ---------- | --------------------------- |
| Python MLX | **25×**                     |
| Swift MLX  | **30×**                     |

## Models

| Precision | Model Size |
| --------- | ---------- |
| [FP32](https://huggingface.co/starkdmi/MossFormer2_SE_48K_MLX/resolve/main/model_fp32.safetensors)      | **221 MB** |
| [FP16](https://huggingface.co/starkdmi/MossFormer2_SE_48K_MLX/resolve/main/model_fp16.safetensors)      | **111 MB** |

HuggingFace: [starkdmi/MossFormer2_SE_48K_MLX](https://huggingface.co/starkdmi/MossFormer2_SE_48K_MLX)

Source: [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)

## License

See [LICENSE](LICENSE).
