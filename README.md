# phonex

[![CI](https://github.com/ekhodzitsky/phonex/actions/workflows/ci.yml/badge.svg)](https://github.com/ekhodzitsky/phonex/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/ekhodzitsky/phonex)](https://github.com/ekhodzitsky/phonex/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://github.com/ekhodzitsky/phonex/blob/main/Dockerfile)

> **Local speech-to-text in 10 languages. One binary. Zero cloud. Works offline.**

phonex is a Rust CLI + server that transcribes speech using Sherpa-ONNX Zipformer models — fully offline, no API keys, no data leaves your machine.

```bash
# Transcribe a file
phonex transcribe interview.wav --language english
# → "hello world"

# Start a local API server
phonex serve --language chinese --port 8080

# Real-time streaming from microphone
streaming --wav audio.wav --language english
```

## 🚀 Try it in 30 seconds

```bash
# With Docker
docker run -p 8080:8080 ghcr.io/ekhodzitsky/phonex:latest \
  phonex serve --language english --port 8080

# Or install from source
cargo install --git https://github.com/ekhodzitsky/phonex
phonex transcribe audio.wav --language english
```

First run downloads the model automatically (~30–600 MB depending on language).

## 🌍 Languages

| Language | Offline | Streaming | Flag |
|----------|---------|-----------|------|
| 🇺🇸 English | ✅ | ✅ | `--language english` |
| 🇨🇳 Chinese + English | ✅ | — | `--language chinese` |
| 🇯🇵 Japanese | ✅ | — | `--language japanese` |
| 🇰🇷 Korean | ✅ | ✅ | `--language korean` |
| 🇷🇺 Russian | ✅ | — | `--language russian` |
| 🇹🇭 Thai | ✅ | — | `--language thai` |
| 🇻🇳 Vietnamese | ✅ | — | `--language vietnamese` |
| 🇭🇰 Cantonese | ✅ | — | `--language cantonese` |
| 🇫🇷 French | — | ✅ | `--language fr20230414` |
| 🇩🇪 German | — | ✅ | `--language de-kroko-20250806` |
| 🇪🇸 Spanish | — | ✅ | `--language es-kroko-20250806` |

> **Streaming** = real-time chunk-by-chunk transcription via the `streaming` binary.  
> **Offline** = end-to-end file transcription via `phonex transcribe` or the HTTP API.

## 📊 Why phonex?

| | phonex | whisper.cpp | Google Cloud STT |
|---|---|---|---|
| **Offline** | ✅ | ✅ | ❌ |
| **Streaming** | ✅ | ❌ | ✅ |
| **Languages** | 10+ | 99 (Whisper) | 125+ |
| **Model size** | ~30–600 MB | 150 MB – 3 GB | Cloud |
| **Latency** | ~70 ms / 5s clip | ~1–3 s | ~200–500 ms |
| **Price** | Free | Free | $0.024/min |
| **Deployment** | Single binary | Single binary | API keys + network |

## 🎤 Quick Start

### 1. Transcribe a file

```bash
phonex transcribe podcast.wav
phonex transcribe meeting.mp3 --language russian --format json
```

### 2. Start the HTTP server

```bash
phonex serve --language english --port 8080 --pool-size 2
```

**Endpoints:**
- `POST /v1/transcribe` — upload audio, get text
- `POST /v1/transcribe/batch` — multiple files at once
- `POST /v1/transcribe/stream` — SSE streaming
- `GET /health` — health check

### 3. Real-time streaming

```bash
streaming --wav audio.wav --language english --chunk-ms 500
```

### 4. Use any Sherpa-ONNX model

```bash
phonex transcribe audio.wav --model-dir models/my-custom-model
```

## 📦 Installation

### Docker

```bash
docker run -p 8080:8080 ghcr.io/ekhodzitsky/phonex:latest
```

### From source

Requires Rust 1.93+ and ONNX Runtime:

```bash
# macOS
brew install onnxruntime
export ORT_PREFER_DYNAMIC_LINK=1
export ORT_LIB_PATH=$(brew --prefix onnxruntime)/lib
export DYLD_LIBRARY_PATH=$ORT_LIB_PATH

cargo install --git https://github.com/ekhodzitsky/phonex
```

See [docs/BUILD.md](docs/BUILD.md) for Linux, Windows, and CoreML setup.

## ⚡ Performance

| Mode | Model | Size | Latency (M1 Pro) | RTF |
|------|-------|------|------------------|-----|
| Offline | English Zipformer | ~70 MB | ~70 ms / 5 s clip | 0.19× |
| Offline | Vietnamese int8 | ~30 MB | ~60 ms / 5 s clip | 0.15× |
| Streaming | English Zipformer | ~78 MB | ~11 ms / 500 ms chunk | 0.03× |
| Streaming | French Zipformer | ~80 MB | ~12 ms / 500 ms chunk | 0.03× |

RTF < 1.0 means faster than real-time. CPU is recommended for all workloads — CoreML is currently slower for Zipformer streaming on Apple Silicon.

## 🏗️ Architecture

phonex is a thin Rust wrapper around Sherpa-ONNX Zipformer models:

```
Audio → kaldi-native-fbank (80-bin mel) → Zipformer Encoder →
Greedy RNNT Decoder → SentencePiece → Text
```

- **Auto-detection**: ONNX shapes, tensor names, and model params are probed automatically — drop in any Zipformer model and it works.
- **Streaming**: Stateful encoder with 35 cached tensors, chunk-by-chunk greedy decode.
- **Session pool**: Parallel ONNX sessions for concurrent requests.

Read more in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 📚 Documentation

- [docs/API.md](docs/API.md) — HTTP & WebSocket API reference
- [docs/MODELS.md](docs/MODELS.md) — Supported models & manual download
- [docs/BUILD.md](docs/BUILD.md) — Build from source, CoreML, Linux/Windows
- [docs/FAQ.md](docs/FAQ.md) — Frequently asked questions
- [docs/SECURITY.md](docs/SECURITY.md) — Authentication, CORS, rate limiting

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📜 License

MIT
