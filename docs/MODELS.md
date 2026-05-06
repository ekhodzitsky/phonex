# Supported Models

phonex works with any Sherpa-ONNX Zipformer encoder-decoder-joiner model. On first load it probes ONNX sessions to auto-detect shapes and tensor names, then writes a `model.json` manifest for instant subsequent starts.

## Built-in Languages (Auto-download)

Specify `--language <lang>` and the model is downloaded automatically on first use.

```bash
phonex transcribe audio.wav --language english
phonex transcribe audio.wav --language japanese
phonex transcribe audio.wav --language cantonese
```

## Full Model Matrix

| Language | Offline | Streaming | CLI Flag | Model |
|----------|---------|-----------|----------|-------|
| **English** | ✅ | ✅ | `--language english` | `sherpa-onnx-zipformer-en-2023-06-26` (offline) / `sherpa-onnx-streaming-zipformer-en-2023-06-21` (streaming) |
| **Cantonese** | ✅ | — | `--language cantonese` | `sherpa-onnx-zipformer-cantonese-2024-03-13` |
| **Chinese + English** | ✅ | — | `--language chinese` | `sherpa-onnx-zipformer-zh-en-2023-11-22` |
| **French** | — | ✅ | `--language fr20230414` | `sherpa-onnx-streaming-zipformer-fr-2023-04-14` |
| **German** | — | ✅ | `--language de-kroko-20250806` | `sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06` |
| **Japanese** | ✅ | — | `--language japanese` | `sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01` |
| **Korean** | ✅ | ✅ | `--language korean` | `sherpa-onnx-zipformer-korean-2024-06-24` (offline) / `sherpa-onnx-streaming-zipformer-korean-2024-06-16` (streaming) |
| **Russian** | ✅ | — | `--language russian` | `sherpa-onnx-small-zipformer-ru-2024-09-18` |
| **Spanish** | — | ✅ | `--language es-kroko-20250806` | `sherpa-onnx-streaming-zipformer-es-kroko-2025-08-06` |
| **Thai** | ✅ | — | `--language thai` | `sherpa-onnx-zipformer-thai-2024-06-20` |
| **Vietnamese** | ✅ | — | `--language vietnamese` | `sherpa-onnx-zipformer-vi-30M-int8-2026-02-09` |

> **Streaming** models only work with the `streaming` binary. They require cached-state Zipformer encoders. Offline models work with `phonex transcribe`, `phonex serve`, and the HTTP API.

## Manual Download

If you prefer to download models manually, place them in `models/`:

```bash
cd models

# English offline
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-06-26.tar.bz2
tar xf sherpa-onnx-zipformer-en-2023-06-26.tar.bz2

# Cantonese
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-cantonese-2024-03-13.tar.bz2
tar xf sherpa-onnx-zipformer-cantonese-2024-03-13.tar.bz2

# Chinese + English bilingual
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-zh-en-2023-11-22.tar.bz2
tar xf sherpa-onnx-zipformer-zh-en-2023-11-22.tar.bz2

# Japanese
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2
tar xf sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2

# Korean
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-korean-2024-06-24.tar.bz2
tar xf sherpa-onnx-zipformer-korean-2024-06-24.tar.bz2

# Russian small
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-small-zipformer-ru-2024-09-18.tar.bz2
tar xf sherpa-onnx-small-zipformer-ru-2024-09-18.tar.bz2

# Thai
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-thai-2024-06-20.tar.bz2
tar xf sherpa-onnx-zipformer-thai-2024-06-20.tar.bz2

# Vietnamese small int8
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2
tar xf sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2
```

## Custom Models

Drop any Sherpa-ONNX Zipformer model into a directory and point `--model-dir` at it. Required files:

| File | Pattern |
|------|---------|
| Encoder | `encoder*.onnx` |
| Decoder | `decoder*.onnx` |
| Joiner | `joiner*.onnx` |
| Tokenizer | `*.model` (e.g. `bpe.model`) |
| Tokens | `tokens.txt` |
| VAD (optional) | `silero_vad.onnx` |
