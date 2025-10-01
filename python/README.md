# MossFormer2 Speech Enhancement - MLX

High-performance speech enhancement model optimized for Apple Silicon using MLX. About 20x faster than real-time, FP32/FP16 support.

## Usage

```bash
pip install -r requirements.txt
python demo.py --input noisy.wav --output clean.wav --precision fp32
```

## Model

The model is automatically downloaded from HuggingFace: [`starkdmi/MossFormer2_SE_48K_MLX`](https://huggingface.co/starkdmi/MossFormer2_SE_48K_MLX)

## License

Apache 2.0
