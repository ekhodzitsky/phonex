# Zipformer Training on RunPod L40S

End-to-end pipeline for training high-quality Zipformer ASR models for phonex.

## Quick Start

1. **Create Pod** on RunPod:
   - GPU: **RTX A6000 Ada** or **L40S** (48 GB VRAM)
   - Template: PyTorch 2.2 + CUDA 12.1
   - Volume: at least **100 GB** (Common Voice + models)

2. **SSH into pod** and run:
   ```bash
   cd /workspace
   git clone <your-repo>
   cd <your-repo>/training/runpod
   bash setup.sh          # one-time
   ```

3. **Train single language**:
   ```bash
   bash train_language.sh ar large    # Arabic, best quality
   bash train_language.sh he small    # Hebrew, low-resource
   ```

4. **Train all 7 languages** (sequential, ~7 days total):
   ```bash
   bash run_all_languages.sh
   ```

## Model Size → Language Mapping

| Lang | Code | Est. CV Hours | Size   | Epochs | Est. Time | Est. Cost (@$0.79/hr) |
|------|------|---------------|--------|--------|-----------|----------------------|
| Arabic    | `ar` | ~1000h | **large**  | 50 | ~40h | ~$32 |
| Hindi     | `hi` | ~200h  | **medium** | 50 | ~25h | ~$20 |
| Turkish   | `tr` | ~400h  | **medium** | 50 | ~30h | ~$24 |
| Indonesian| `id` | ~150h  | **medium** | 50 | ~20h | ~$16 |
| Persian   | `fa` | ~200h  | **medium** | 50 | ~25h | ~$20 |
| Hebrew    | `he` | ~80h   | **small**  | 60 | ~15h | ~$12 |
| Swahili   | `sw` | ~30h   | **small**  | 60 | ~12h | ~$10 |
| **Total** |      |        |            |    | **~167h** | **~$134** |

## Quality Optimizations Applied

- **SpecAugment**: time-warp (80), time-mask (2×5%), freq-mask (2×15%)
- **Auxiliary CTC loss**: scale 0.2 (stabilizes transducer training)
- **Label smoothing**: 0.1
- **Model averaging**: last 10 epochs for decode & ONNX export
- **FP16 training**: ~30% faster on L40S, no quality loss
- **Bucketing sampler**: 50 buckets for efficient batching
- **BPE tokenizer**: 500 units, character coverage 99.95%

## Output Structure

After training each language:
```
/workspace/icefall_training/exp/zipformer_large_ar/
├── checkpoint-*.pt          # per-epoch checkpoints
├── avg-10.pt                # averaged model (best for inference)
├── train.log                # full training log
├── decode.log               # WER on test set
├── model_info.txt           # metadata
└── (encoder|decoder|joiner).onnx  # after export_onnx.sh
```

## Integrating with phonex

Once ONNX files are ready, place them in your phonex model directory:
```
models/
├── ar-zipformer/
│   ├── encoder.onnx
│   ├── decoder.onnx
│   ├── joiner.onnx
│   └── tokens.txt
```

phonex `discover_model_files()` will auto-detect them if filenames start with `encoder`, `decoder`, `joiner`.

## Notes

- **OOM?** Lower `--max-duration` in `train_language.sh` by 50-100.
- **Resume training**: `train.py` auto-resumes from latest checkpoint in `--exp-dir`.
- **Multi-GPU**: Change `--world-size 1` to `--world-size 8` and use `torchrun`. Not needed for these languages.
- **Common Voice versions**: If 16.1 is unavailable for a language, the script auto-falls back to 16.1-2022-12-07.
