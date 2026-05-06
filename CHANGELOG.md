# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-05-05

### Added
- **Streaming inference** — stateful Zipformer encoder with 35 cached tensors, chunk-by-chunk greedy RNNT decode
- **WebSocket real-time API** — `/v1/transcribe/stream` with binary audio frames, partial/final transcripts
- **Streaming VAD** — Silero VAD with hysteresis, filters silence and auto-flushes on speech end
- **API key authentication** — Bearer token middleware (`--api-key` / `PHONEX_API_KEY`)
- **CORS restriction** — configurable origin allowlist (`--cors-origins`)
- **Rate limiting** — per-IP token bucket
- **Prometheus metrics** — `requests_total`, `request_duration_seconds`, `ws_connections_total`
- **Graceful shutdown** — `CancellationToken` drain for in-flight requests
- **WebSocket backpressure** — max 480k samples (30s @ 16kHz), idle timeout (60s)
- **Input validation** — NaN/Inf rejection on WebSocket binary frames
- **Docker support** — multi-stage Dockerfile for linux/amd64 and linux/arm64
- **GitHub Actions CI** — build, test, integration tests, Docker image build

### Changed
- **Project renamed** from `siamstt` to `phonex`
- Generic multilingual STT engine (was Thai-only)
- Auto-detect model parameters from ONNX metadata (`d_model`, `vocab_size`, `context_size`, tensor names)
- CPU fallback for streaming encoder (CoreML is ~6× slower for streaming)

### Fixed
- **Critical state accumulation bug** — `strip_prefix("new_cached_")` → `replacen("new_", "", 1)` fixed all-blank predictions in streaming
- **Decoder context** — `blank_id` → `1i64` (BPE `<s>`) for correct logits confidence
- **Error handling** — resample/batch no longer silent-fail
- **Hardcoded model name** — now uses `ModelInfo.model_id/model_name` from config or directory basename

## [0.1.0] - 2025-04-20

### Added
- Offline transcription with Sherpa-ONNX Zipformer models
- Kaldi-native-fbank feature extraction (80-bin log-mel)
- Greedy RNNT decode with auto-detected context size
- SentencePiece BPE tokenization
- HTTP server with REST API, batch transcription, SSE streaming
- Session pool for parallel inference
- VAD-based segmentation with word-level timing
- Multi-format audio input: WAV, MP3, OGG, FLAC, AAC

[unreleased]: https://github.com/ekhodzitsky/phonex/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ekhodzitsky/phonex/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ekhodzitsky/phonex/releases/tag/v0.1.0
