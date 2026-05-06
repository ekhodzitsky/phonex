# phonex

> Generic on-device speech-to-text. Local inference, no cloud APIs, full privacy.

## Overview

`phonex` started as a Thai STT project (originally named *siamstt*, from *Siam* = Thailand) and evolved into a **generic multilingual STT engine**.

Today it is a Rust library + CLI + HTTP/WebSocket server that performs speech-to-text using **any Sherpa-ONNX Zipformer-transducer model**. It auto-detects model parameters (`d_model`, `vocab_size`, `context_size`, tensor names) from ONNX metadata, so you can drop in any language model and it just works.

**Two inference modes:**
- **Offline** — transcribe audio files end-to-end. Works with **any language** that has a Sherpa-ONNX Zipformer model (Thai, English, Chinese, German, etc.).
- **Streaming** — real-time chunk-by-chunk transcription. Currently works with **English streaming Zipformer models** only, because Sherpa-ONNX has not yet released streaming variants for other languages.

The default model directory points to the Sherpa-ONNX Zipformer Thai model (GigaSpeech2, INT8 quantized, ~148 MB). Download it below or point `--model-dir` to any Sherpa-ONNX Zipformer model.

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

| Language | Offline | Streaming | Example model |
|----------|---------|-----------|---------------|
| **Thai** | ✅ | ❌ (no model yet) | `sherpa-onnx-zipformer-thai-2024-06-21` |
| **English** | ✅ | ✅ | `sherpa-onnx-streaming-zipformer-en-2023-06-21` |
| **Chinese, German, etc.** | ✅ | ❌ (no model yet) | Any Sherpa-ONNX Zipformer model |

> **Why English only for streaming?** Streaming requires a special "streaming Zipformer" encoder with cached states (35 tensors in our case). Sherpa-ONNX has released streaming variants for English, but not yet for Thai or other languages. When streaming models become available for other languages, they will work out of the box — no code changes needed.

### Known limitations
- Streaming requires a streaming Zipformer model. Only English has one in Sherpa-ONNX today.
- CoreML EP is ~6× slower than CPU for the streaming encoder (measured on M-series). CPU is recommended for streaming.

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

### Thai model (default)

```bash
cd models
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-thai-2024-06-20.tar.bz2
tar xf sherpa-onnx-zipformer-thai-2024-06-20.tar.bz2
rm sherpa-onnx-zipformer-thai-2024-06-20.tar.bz2
```

### English model example

```bash
cd models
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2
tar xf sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2
```

Then run:
```bash
cargo run --release --bin phonex -- transcribe audio.wav --model-dir models/sherpa-onnx-streaming-zipformer-en-2023-06-21
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

## Quick Start

### Thai — offline transcription (default)

```bash
# Download Thai model
cd models
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-thai-2024-06-20.tar.bz2
tar xf sherpa-onnx-zipformer-thai-2024-06-20.tar.bz2

# Transcribe a Thai audio file
cargo run --release --bin phonex -- transcribe audio.wav
# → สวัสดีครับ
```

### English — real-time streaming

```bash
# Download English streaming model
cd models
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2
tar xf sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2

# Stream from a WAV file (simulates microphone)
cargo run --release --bin streaming -- \
  --wav audio.wav \
  --model-dir models/sherpa-onnx-streaming-zipformer-en-2023-06-21 \
  --chunk-ms 500

# Or start the server with WebSocket streaming
cargo run --release --bin server -- \
  --model-dir models/sherpa-onnx-streaming-zipformer-en-2023-06-21 \
  --port 8080
```

### Generic CLI commands

```bash
# Offline transcription of any supported audio format
cargo run --release --bin phonex -- transcribe audio.mp3 --format json

# Use a non-default model directory
cargo run --release --bin phonex -- transcribe audio.wav --model-dir models/my-model

# Start the HTTP server with Thai model
cargo run --release --bin phonex -- serve --port 8080 --pool-size 2
```

## Server Usage

```bash
# Start the HTTP/WebSocket server with default Thai model
cargo run --release --bin server -- --port 8080 --pool-size 2

# Use an English model with streaming support
cargo run --release --bin server -- --model-dir models/sherpa-onnx-streaming-zipformer-en-2023-06-21 --port 8080

# Production: enable auth and restrict CORS
cargo run --release --bin server \
  -- --api-key "sk-123456" \
  --cors-origins "https://myapp.com,https://app.myapp.com" \
  --port 8080 --pool-size 4
```

Options:
- `--model-dir` — model directory (default: `models/sherpa-onnx-zipformer-thai-2024-06-20`)
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

## License

MIT
