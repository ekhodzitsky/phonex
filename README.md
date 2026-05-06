# phonex

[![CI](https://github.com/ekhodzitsky/phonex/actions/workflows/ci.yml/badge.svg)](https://github.com/ekhodzitsky/phonex/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/ekhodzitsky/phonex)](https://github.com/ekhodzitsky/phonex/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.93%2B-blue.svg)](https://www.rust-lang.org)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://github.com/ekhodzitsky/phonex/blob/main/Dockerfile)
[![GitHub stars](https://img.shields.io/github/stars/ekhodzitsky/phonex?style=social)](https://github.com/ekhodzitsky/phonex)

> Generic on-device speech-to-text. Local inference, no cloud APIs, full privacy.

## Overview

**phonex** is a Rust library + CLI + HTTP/WebSocket server that performs speech-to-text using **any Sherpa-ONNX Zipformer-transducer model**. It auto-detects model parameters (`d_model`, `vocab_size`, `context_size`, tensor names) from ONNX metadata, so you can drop in any language model and it just works.

**Two inference modes:**
- **Offline** — transcribe audio files end-to-end. Works with **any language** that has a Sherpa-ONNX Zipformer model (Thai, English, Chinese, German, etc.).
- **Streaming** — real-time chunk-by-chunk transcription. Currently works with **English streaming Zipformer models** only, because Sherpa-ONNX has not yet released streaming variants for other languages.

The default model is English (`sherpa-onnx-zipformer-en-2023-06-26`). You can switch languages with the `--language` flag or point `--model-dir` to any Sherpa-ONNX Zipformer model.

## Status

**Production-ready for offline and streaming inference.** The pipeline transcribes audio files with results matching Sherpa-ONNX Python greedy decode. RTF ~0.19x offline, ~0.03x per 500ms chunk streaming on Apple Silicon (M-series).

### What's working
- **Generic model support** — works with any Sherpa-ONNX Zipformer encoder-decoder-joiner model
- **True streaming inference** — stateful Zipformer encoder with 35 cached tensors, chunk-by-chunk greedy RNNT decode
- **WebSocket real-time API** — `/v1/transcribe/stream` with binary audio frames, partial/final transcripts
- **Streaming VAD** — Silero VAD with hysteresis, filters silence and auto-flushes on speech end
- ONNX encoder + decoder + joiner inference via `ort` 2.0
- Kaldi-native-fbank feature extraction (80-bin log-mel, 25 ms / 10 ms frames)
- Greedy RNNT decode with auto-detected context size
- SentencePiece BPE tokenization
- VAD-based segmentation (Silero VAD) with word-level timing and speaker labels
- HTTP server with REST API, batch transcription, and SSE streaming
- Session pool for parallel inference
- **API key authentication** — Bearer token middleware
- **CORS restriction** — configurable origin allowlist
- Per-IP rate limiting, Prometheus metrics
- CLI: `transcribe`, `serve`, and `streaming` commands
- Multi-format audio input: WAV, MP3, OGG, FLAC, AAC (via symphonia)
- macOS ARM64 with dynamic ONNX Runtime linking

### Language support matrix

| Language | Offline | Streaming | CLI flag | Example model |
|----------|---------|-----------|----------|---------------|
| **English** | ✅ | ✅ | `--language english` | `sherpa-onnx-zipformer-en-2023-06-26` (offline) / `sherpa-onnx-streaming-zipformer-en-2023-06-21` (streaming) |
| **Chinese** | ✅ | — | `--language chinese` | `sherpa-onnx-zipformer-zh-en-2023-11-22` (Chinese+English bilingual) |
| **French** | — | ✅ | `--language fr20230414` | `sherpa-onnx-streaming-zipformer-fr-2023-04-14` |
| **German** | — | ✅ | `--language de-kroko-20250806` | `sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06` |
| **Japanese** | ✅ | — | `--language japanese` | `sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01` |
| **Korean** | ✅ | ✅ | `--language korean` | `sherpa-onnx-zipformer-korean-2024-06-24` (offline) / `sherpa-onnx-streaming-zipformer-korean-2024-06-16` (streaming) |
| **Russian** | ✅ | — | `--language russian` | `sherpa-onnx-small-zipformer-ru-2024-09-18` |
| **Spanish** | — | ✅ | `--language es-kroko-20250806` | `sherpa-onnx-streaming-zipformer-es-kroko-2025-08-06` |
| **Thai** | ✅ | — | `--language thai` | `sherpa-onnx-zipformer-thai-2024-06-20` |
| **Vietnamese** | ✅ | — | `--language vietnamese` | `sherpa-onnx-zipformer-vi-30M-int8-2026-02-09` |
| **Other** | ✅ | — | — | Any Sherpa-ONNX Zipformer offline model |

> **Streaming vs offline:** Offline модели транскрибируют файл целиком. Streaming модели работают в реальном времени (chunk-by-chunk) и используются только в бинарнике `streaming`. Не все языки имеют streaming-модели — их наличие зависит от upstream (Sherpa-ONNX).

### Known limitations
- Streaming требует специальной streaming Zipformer модели. На данный момент streaming есть для English, French, German, Spanish и Korean.
- CoreML EP is ~6× slower than CPU for the streaming encoder (measured on M-series). CPU is recommended for streaming.
- First run with a new `--language` will download the model archive (30–600 MB depending on language).
- CoreML EP is ~6× slower than CPU for the streaming encoder (measured on M-series). CPU is recommended for streaming.
- First run with a new `--language` will download the model archive (30–600 MB depending on language).

## Why phonex?

| | phonex | [whisper.cpp](https://github.com/ggerganov/whisper.cpp) | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) Python |
|---|--------|----------|--------------|
| **Streaming** | ✅ Native | ❌ No | ✅ Yes |
| **Model size** | ~78–148 MB | ~150 MB–3 GB | ~78–148 MB |
| **Languages** | Any Sherpa-ONNX Zipformer | Whisper-only | Any Sherpa-ONNX |
| **Runtime** | Rust + ONNX Runtime | C++ | C++ + Python bindings |
| **Server** | Built-in REST + WebSocket | Separate | Separate |
| **Self-contained** | ✅ Single binary | ✅ Single binary | ❌ Python env |

**phonex** sits between whisper.cpp and sherpa-onnx: it gives you whisper.cpp's deployment simplicity (single Rust binary, no Python) with sherpa-onnx's model ecosystem and native streaming support.

## Model files

Place model files in any directory (e.g. `models/my-model/`). The server auto-discovers files by pattern:

| File | Pattern | Example |
|------|---------|---------|
| Encoder | `encoder*.onnx` | `encoder.onnx`, `encoder-epoch-12-avg-5.int8.onnx` |
| Decoder | `decoder*.onnx` | `decoder.onnx`, `decoder-epoch-12-avg-5.int8.onnx` |
| Joiner | `joiner*.onnx` | `joiner.onnx`, `joiner-epoch-12-avg-5.int8.onnx` |
| Tokenizer | `*.model` | `bpe.model`, `tokenizer.model` |
| Tokens | `tokens.txt` | `tokens.txt` |
| VAD (optional) | `silero_vad.onnx` | `silero_vad.onnx` |

On first load, `phonex` probes the ONNX sessions to detect shapes and tensor names, then writes a `model.json` manifest for instant subsequent starts.

### Built-in languages (auto-download)

phonex can automatically download models for supported languages on first use:

```bash
# English (default)
cargo run --release --bin phonex -- transcribe audio.wav

# Chinese + English bilingual
cargo run --release --bin phonex -- transcribe audio.wav --language chinese

# Japanese
cargo run --release --bin phonex -- transcribe audio.wav --language japanese

# Korean
cargo run --release --bin phonex -- transcribe audio.wav --language korean

# Russian (small model, ~86 MB)
cargo run --release --bin phonex -- transcribe audio.wav --language russian

# Thai
cargo run --release --bin phonex -- transcribe audio.wav --language thai

# Vietnamese (small int8 model, ~30 MB)
cargo run --release --bin phonex -- transcribe audio.wav --language vietnamese
```

### Manual download

If you prefer to download models manually, place them in `models/`:

```bash
cd models

# English offline
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-06-26.tar.bz2
tar xf sherpa-onnx-zipformer-en-2023-06-26.tar.bz2

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

## Docker

```bash
# Build and run with docker-compose
docker-compose up --build

# Or build manually
docker build -t phonex .
docker run -p 8080:8080 -v ./models:/app/models phonex
```

## Build

Requires Rust 1.93+ and ONNX Runtime (Homebrew on macOS):

```bash
export ORT_PREFER_DYNAMIC_LINK=1
export ORT_LIB_PATH=/opt/homebrew/Cellar/onnxruntime/1.25.1/lib
export DYLD_LIBRARY_PATH=$ORT_LIB_PATH

cargo build --release
```

The `ORT_PREFER_DYNAMIC_LINK=1` environment variable is required to avoid a protobuf symbol collision between ONNX Runtime and SentencePiece.

### Apple Silicon (CoreML)

You can enable the CoreML execution provider on macOS to use the Apple Neural Engine:

```bash
cargo build --release --features coreml
```

> **Note:** CoreML support is experimental. For the Sherpa-ONNX Zipformer Thai INT8 model, CPU inference is currently faster (~70-80ms/request) than CoreML (~140-170ms/request) on M1 Pro. Your mileage may vary with other models.

## Installation

### From source (recommended)

```bash
git clone https://github.com/ekhodzitsky/phonex.git
cd phonex

# macOS: install ONNX Runtime first
brew install onnxruntime
export ORT_PREFER_DYNAMIC_LINK=1
export ORT_LIB_PATH=$(brew --prefix onnxruntime)/lib
export DYLD_LIBRARY_PATH=$ORT_LIB_PATH

# Linux: see Dockerfile for ONNX Runtime setup

cargo build --release

# Binaries will be in target/release/
# - phonex      (main CLI)
# - server      (HTTP/WebSocket server)
# - streaming   (real-time streaming CLI)
```

### Docker

```bash
docker run -p 8080:8080 -v ./models:/app/models ghcr.io/ekhodzitsky/phonex:latest
```

> **Note:** Pre-built binaries and crates.io publishing coming soon. Track progress in [Roadmap](#roadmap).

## Quick Start

### English — offline transcription (default)

```bash
# Transcribe an English audio file
cargo run --release --bin phonex -- transcribe audio.wav
# → "hello world"
```

### English — real-time streaming

```bash
# Stream from a WAV file (simulates microphone)
cargo run --release --bin streaming -- \
  --wav audio.wav \
  --language en20230621 \
  --chunk-ms 500

# Or start the server with WebSocket streaming
cargo run --release --bin server -- \
  --language english \
  --port 8080
```

### Other streaming languages

```bash
# French streaming
cargo run --release --bin streaming -- --wav audio.wav --language fr20230414

# German streaming
cargo run --release --bin streaming -- --wav audio.wav --language de-kroko-20250806

# Spanish streaming
cargo run --release --bin streaming -- --wav audio.wav --language es-kroko-20250806

# Korean streaming
cargo run --release --bin streaming -- --wav audio.wav --language ko20240616
```

### Generic CLI commands

```bash
# Offline transcription of any supported audio format (English default)
cargo run --release --bin phonex -- transcribe audio.mp3 --format json

# Use a specific language
cargo run --release --bin phonex -- transcribe audio.wav --language russian

# Use a custom model directory
cargo run --release --bin phonex -- transcribe audio.wav --model-dir models/my-model

# Start the HTTP server with Thai model
cargo run --release --bin phonex -- serve --port 8080 --pool-size 2 --language thai
```

## Server Usage

```bash
# Start the HTTP/WebSocket server with default English model
cargo run --release --bin server -- --port 8080 --pool-size 2

# Use a specific language
cargo run --release --bin server -- --language japanese --port 8080
cargo run --release --bin server -- --language chinese --port 8080
cargo run --release --bin server -- --language korean --port 8080

# Production: enable auth and restrict CORS
cargo run --release --bin server \
  -- --api-key "sk-123456" \
  --cors-origins "https://myapp.com,https://app.myapp.com" \
  --port 8080 --pool-size 4
```

Options:
- `--model-dir` — model directory (overrides `--language`)
- `--language` — built-in language model (`chinese`, `english`, `japanese`, `korean`, `russian`, `thai`, `vietnamese`)
- `--bind` — bind address (default: `127.0.0.1`)
- `--port` — listen port (default: `8080`)
- `--pool-size` — parallel ONNX session pool size for offline inference (default: `1`)
- `--api-key` — optional Bearer token for authentication (also via `PHONEX_API_KEY` env var)
- `--cors-origins` — comma-separated allowed CORS origins (default: `http://localhost:3000,http://localhost:5173`)

### HTTP API

**Health check:**
```bash
curl http://localhost:8080/health
```
Response:
```json
{"status":"ok","model":"sherpa-onnx-zipformer-thai-2024-06-20","version":"0.2.0"}
```

**Model info:**
```bash
curl http://localhost:8080/v1/models
```

**Transcribe audio (multipart upload):**
```bash
# Convert WAV to raw f32 LE (or upload any supported format via CLI)
curl -X POST http://localhost:8080/v1/transcribe \
  -F "audio=@audio.raw" \
  -F "sample_rate=16000"

# With VAD segmentation (speaker diarization + timing)
curl -X POST 'http://localhost:8080/v1/transcribe?vad=true' \
  -F "audio=@audio.raw" \
  -F "sample_rate=16000"
```
Response:
```json
{"text":"สวัสดีครับ","words":[],"duration":3.52}
```

**Batch transcribe multiple files:**
```bash
curl -X POST http://localhost:8080/v1/transcribe/batch \
  -F "audio=@1.raw" \
  -F "audio=@2.raw" \
  -F "sample_rate=16000"
```

**SSE streaming:**
```bash
curl -N -X POST http://localhost:8080/v1/transcribe/stream \
  -F "audio=@audio.raw" \
  -F "sample_rate=16000"
```
Emits `final` event when transcription completes.

**Prometheus metrics:**
```bash
curl http://localhost:8080/metrics
```

### WebSocket Streaming

**Primary endpoint:** `ws://localhost:8080/v1/transcribe/stream`

Send:
- **Binary frames**: raw f32 LE audio samples at 16kHz mono (any chunk size)
- **JSON commands**:
  - `{"cmd":"Clear"}` — reset pipeline (new utterance)
  - `{"cmd":"Stop"}` — finalize and close connection
  - `{"cmd":"Configure","sample_rate":8000}` — configure input rate

Server replies with JSON:
```json
{"type":"ready","model":"sherpa-onnx-streaming-zipformer-en-2023-06-21","sample_rate":16000,"version":"0.2.0"}
{"type":"partial","text":"AFTER EARLY","timestamp":1234567890.0}
{"type":"partial","text":"AFTER EARLY NIGHTFALL","timestamp":1234567890.0}
{"type":"final","text":"AFTER EARLY NIGHTFALL THE YELLOW LAMPS...","timestamp":1234567890.0}
```

**Legacy endpoint:** `ws://localhost:8080/stream` — same protocol but uses offline engine with accumulation.

**Connection limits:** max 100 concurrent WebSocket connections (configurable). Idle connections close after 60 seconds.

## Streaming CLI

```bash
# Real-time streaming transcription from a WAV file
cargo run --release --bin streaming -- \
  --wav audio.wav \
  --model-dir models/sherpa-onnx-streaming-zipformer-en-2023-06-21 \
  --chunk-ms 500

# With VAD (filters silence, auto-segments)
cargo run --release --bin streaming -- \
  --wav audio.wav \
  --model-dir models/sherpa-onnx-streaming-zipformer-en-2023-06-21 \
  --chunk-ms 500 \
  --vad
```

## Streaming API (Rust)

For real-time chunked inference, use the `StreamingPipeline` API directly:

```rust
use phonex::{ModelInfo, StreamingPipeline};

let info = ModelInfo::from_model_dir("models/sherpa-onnx-streaming-zipformer-en-2023-06-21")?;
let mut pipeline = StreamingPipeline::from_model_dir(
    "models/sherpa-onnx-streaming-zipformer-en-2023-06-21",
    &info,
    Some("models/silero_vad.onnx"), // optional VAD
)?;

// Feed audio in chunks (e.g. from a microphone stream)
pipeline.accept_audio(&samples)?;

// Finalize and get the full transcription
let text = pipeline.flush()?;
```

The pipeline maintains encoder state across chunks and performs greedy RNNT decode. Call `reset()` between utterances to clear internal caches.

## Performance

| Mode | Model | Size | Latency (M1 Pro) | RTF |
|------|-------|------|------------------|-----|
| Offline | Sherpa-ONNX Zipformer Thai INT8 | 148 MB | ~70-80ms / 5s clip | ~0.19x |
| Streaming (per chunk) | Sherpa-ONNX Zipformer English | 78 MB | ~11ms / 500ms chunk | ~0.03x |

**Streaming:** encoder latency ~11ms per 500ms audio chunk (CPU). End-to-end latency (audio → partial text) is dominated by chunk size — 500ms chunks yield ~500ms latency. Real-time factor 0.03x means the engine processes audio 30× faster than real-time.

**CoreML note:** On Apple Silicon, CPU inference is faster than CoreML for the streaming encoder (~11ms vs ~66ms per chunk). CPU is recommended for streaming workloads.

## Security

- **Authentication:** Set `--api-key` or `PHONEX_API_KEY` env var to require `Authorization: Bearer <key>` on all endpoints.
- **CORS:** By default only `localhost:3000` and `localhost:5173` are allowed. Use `--cors-origins` to customize.
- **Rate limiting:** Per-IP token bucket. Enable with `--rate-limit-per-minute` (not exposed in CLI yet — configure via code).
- **Input validation:** WebSocket binary frames are validated for NaN/Inf. Malformed audio is rejected with error code `invalid_audio_samples`.
- **Resource limits:** Max 100 concurrent WS connections, 500MB request body limit, 30s audio buffer backpressure limit.

## Development

```bash
# Run all tests
cargo test

# Run integration tests (real ONNX inference)
cargo test --test integration
cargo test --test server_inference -- --ignored

# Run HTTP server tests (fast, no model load)
cargo test --test server

# Format and lint
cargo fmt
cargo clippy
```

## FAQ

**Q: Why not just use Whisper / whisper.cpp?**
A: Whisper models are 150 MB–3 GB and don't support native streaming (you have to buffer the entire audio). phonex uses Zipformer-transducer models (~78 MB for English streaming) with native chunk-by-chunk inference — ~500ms latency vs Whisper's multi-second latency.

**Q: Can I use my own model?**
A: Yes. Drop any Sherpa-ONNX Zipformer encoder-decoder-joiner model into a directory and point `--model-dir` at it. phonex auto-detects all parameters from ONNX metadata.

**Q: Does it run on Raspberry Pi / embedded?**
A: Should work on any Linux ARM64 with ONNX Runtime. Not yet tested on Raspberry Pi — if you try it, open an issue with results.

**Q: Will non-English streaming work soon?**
A: phonex уже поддерживает streaming для English, French, German, Spanish и Korean. Для Japanese, Russian, Thai и Vietnamese пока нет streaming моделей в Sherpa-ONNX. Когда они появятся — заработают без изменений кода.

**Q: How does phonex compare to running sherpa-onnx Python?**
A: phonex is a single static binary with no Python environment. Easier to deploy, harder to break, lower memory overhead. Feature parity for inference; phonex adds built-in REST/WebSocket server.

## Roadmap

- [ ] Pre-built binaries for macOS ARM64 and Linux x64
- [ ] Publish to crates.io
- [ ] CUDA execution provider support
- [ ] gRPC API
- [ ] OpenAPI spec + Swagger UI
- [ ] Distributed tracing (OpenTelemetry)
- [ ] Model hot-swap without restart
- [ ] Streaming models for Japanese, Russian, Thai, Vietnamese (blocked by upstream model availability)
- [ ] Raspberry Pi / embedded ARM32 support
- [ ] Real-time microphone input example

## License

MIT
