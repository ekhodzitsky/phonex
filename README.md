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
| 🇺🇸 English | ✅ | ✅ | `--language english` / `--language english-streaming` |
| 🇨🇳 Chinese + English | ✅ | ✅ | `--language chinese` / `--language chinese-streaming` |
| 🇯🇵 Japanese | ✅ | — | `--language japanese` |
| 🇰🇷 Korean | ✅ | ✅ | `--language korean` / `--language korean-streaming` |
| 🇷🇺 Russian | ✅ | — | `--language russian` |
| 🇹🇭 Thai | ✅ | — | `--language thai` |
| 🇻🇳 Vietnamese | ✅ | — | `--language vietnamese` |
| 🇭🇰 Cantonese | ✅ | — | `--language cantonese` |
| 🇫🇷 French | — | ✅ | `--language french` |
| 🇩🇪 German | — | ✅ | `--language german` |
| 🇪🇸 Spanish | — | ✅ | `--language spanish` |
| 🇧🇩 Bengali | — | ✅ | `--language bengali` |

> **Streaming** = real-time WebSocket transcription (true Zipformer streaming for streaming models, VAD-chunked for offline).  
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
- `POST /v1/transcribe` — upload audio, get text (`?diarize=true` for speaker diarization)
- `POST /v1/transcribe/batch` — multiple files at once
- `POST /v1/transcribe/stream` — SSE streaming
- `GET /health` — health check
- `GET /docs` — Swagger UI (OpenAPI)
- `GET /openapi.json` — OpenAPI spec
- `POST /v1/admin/reload` — hot-swap model without restart

**gRPC** (optional, `--grpc-port 50051`):
- `Transcribe` — offline transcription with word-level timestamps
- `StreamTranscribe` — bidirectional streaming with per-word timing
- API key authentication is enforced on every gRPC method when `--api-key` is set

See `proto/phonex.proto` for the service definition.

### 3. Real-time streaming

```bash
streaming --wav audio.wav --language english --chunk-ms 500
```

### 4. Use any Sherpa-ONNX model

```bash
phonex transcribe audio.wav --model-dir models/my-custom-model
```

> **Model integrity**: phonex supports SHA-256 checksum verification for downloaded models. You can add an expected hash to `ModelSpec` and the engine will verify the archive before extraction.

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

### Configuration file

```bash
cp phonex.yaml.example phonex.yaml
phonex serve --config phonex.yaml
```

### Environment variables

| Variable | Purpose |
|----------|---------|
| `PHONEX_API_KEY` | Bearer token required for all API endpoints |
| `PHONEX_ADMIN_API_KEY` | Bearer token required for admin endpoints (`/v1/admin/reload`, `/metrics`) |
| `PHONEX_TRUST_PROXY` | Set to `true` to trust `X-Forwarded-For` / `X-Real-IP` for rate limiting (only behind a trusted proxy) |

### Docker Compose (with Prometheus + Grafana)

```bash
docker compose up
# Open Grafana at http://localhost:3000 (admin/admin)
# Open Prometheus at http://localhost:9090
```

## ⚡ Performance

| Mode | Model | Size | Latency (M1 Pro) | RTF |
|------|-------|------|------------------|-----|
| Offline | English Zipformer | ~70 MB | ~70 ms / 5 s clip | 0.19× |
| Offline | Vietnamese int8 | ~30 MB | ~60 ms / 5 s clip | 0.15× |
| Streaming | English Zipformer | ~78 MB | ~11 ms / 500 ms chunk | 0.03× |
| Streaming | French Zipformer | ~80 MB | ~12 ms / 500 ms chunk | 0.03× |

RTF < 1.0 means faster than real-time. CPU is recommended for all workloads — CoreML is currently slower for Zipformer streaming on Apple Silicon.

### GPU Acceleration

| Feature | Platform | Command |
|---------|----------|---------|
| CoreML | macOS (Apple Silicon) | `cargo build --features coreml` |
| CUDA | Linux (NVIDIA) | `cargo build --features cuda` |

CUDA requires NVIDIA drivers and cuDNN. See [docs/BUILD.md](docs/BUILD.md) for setup.

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

## 🔒 Security

- **API key authentication**: Set `--api-key` (or `PHONEX_API_KEY`) to require `Authorization: Bearer <key>` on all endpoints. Set `--admin-api-key` (or `PHONEX_ADMIN_API_KEY`) to restrict `/v1/admin/reload` and `/metrics` separately.
- **Rate limiting**: Enable with `--rate-limit-per-minute` to protect against abuse. If running behind a trusted reverse proxy, set `--trust-proxy` (or `PHONEX_TRUST_PROXY`) so rate limiting uses the real client IP from `X-Forwarded-For`.
- **Path validation**: Model reload (`POST /v1/admin/reload`) validates that paths are absolute and rejects `..` traversal attempts.
- **gRPC auth**: gRPC endpoints also require the Bearer API key when authentication is enabled.

See [SECURITY.md](SECURITY.md) and [docs/SECURITY.md](docs/SECURITY.md) for full details.

## 📚 Documentation

- [docs/API.md](docs/API.md) — HTTP & WebSocket API reference
- [docs/MODELS.md](docs/MODELS.md) — Supported models & manual download
- [docs/BUILD.md](docs/BUILD.md) — Build from source, CoreML, Linux/Windows
- [docs/FAQ.md](docs/FAQ.md) — Frequently asked questions
- [docs/SECURITY.md](docs/SECURITY.md) — Authentication, CORS, rate limiting

## 🔌 FFI / Embedded Use

phonex can be embedded into Android, iOS, Python, or any C-compatible runtime via the `ffi` feature. No HTTP server, no JNI boilerplate — just a shared library and C headers.

```bash
cargo build --release --features ffi --no-default-features
```

### C API

```c
void* engine = phonex_engine_new("models/sherpa-onnx-zipformer-thai-2024-06-20");
char* text = phonex_transcribe_file(engine, "audio.wav");
printf("%s\n", text);
phonex_string_free(text);
phonex_engine_free(engine);
```

See [docs/FFI.md](docs/FFI.md) for full reference and Python examples.

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📜 License

MIT
