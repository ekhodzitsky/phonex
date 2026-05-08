# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.6] - 2026-05-08

### Security
- **CodeQL static analysis** — new GitHub Actions workflow (`codeql.yml`) runs weekly and on every push/PR.
- **SHA-256 model manifest** — all 18 model entries in `models/manifest.json` now have verified SHA-256 hashes for reproducible downloads.

### Fixed
- `shutdown_signal()` now compiles on Windows (`#[cfg(windows)]` Ctrl-C fallback).
- Python wheels CI: ONNX Runtime installed for all platforms (Linux x64/ARM64, macOS x64/ARM64, Windows x64).
- macOS CI tests exclude `python` feature to avoid PyO3 linking failures.

## [0.2.5] - 2026-05-07

### Security
- Path traversal validation for `POST /v1/admin/reload` (`validate_model_dir`).
- Admin API key auth (`admin_api_key`) for `/v1/admin/reload` and `/metrics`.
- Rate limiting `trust_proxy` flag (disabled by default) to prevent IP spoofing.
- gRPC `Authorization: Bearer` enforcement and 500 MB body cap.
- CORS explicit `allow_headers` whitelist (`Content-Type`, `Authorization`, `X-Request-Id`).
- SHA-256 infrastructure for model verification.

### Fixed
- All `cargo fmt` and `cargo clippy` errors resolved across 40 files.
- `result_large_err` and `large_enum_variant` clippy lints fixed (box large variants).
- `derivable_impls` fixed: `PhonexConfig` and `AuthConfig` now derive `Default`.
- FFI `missing_safety_doc` and `nonminimal_bool` lints fixed.

### Added
- **15 new unit and regression tests** — pool exhaustion, pool close, guard drop, auth bypass, CORS, path traversal, VAD NaN/inf rejection, error display.
- **5 benchmarks** — pool cycle, chunked latency, WS streaming latency, streaming encoder latency, audio preprocessor latency (all run without ONNX models via mocks).
- **FFI smoke tests** (`tests/ffi_smoke.py`) — double-free safety and new→free cycle.
- **rustdoc coverage** — zero warnings; public API items now have doc comments.
- **CI safety** — all integration tests skip gracefully when ONNX model files are absent.
- **cargo-deny** — `.cargo/deny.toml` with allowed licenses and ignored advisories.

## [0.2.4] - 2026-05-05

### Security
- ONNX inference in WebSocket and chunked streaming now runs inside `spawn_blocking` to prevent blocking the async runtime and potential denial-of-service.
- `CheckoutGuard` RAII wrapper ensures inference session pool items are always returned on panic or task cancellation, preventing pool exhaustion.
- FFI handles use `AcqRel` memory ordering and atomic `disposed` checks to prevent use-after-free and double-free across the C boundary.
- Model reload endpoint (`POST /v1/admin/reload`) validates that paths are absolute and rejects parent-directory (`..`) traversal attempts.
- Model archive extraction uses `tar::Entry::unpack_in` to prevent archive path traversal attacks.
- Session pool checkin uses blocking send with warnings on channel closure during shutdown.
- VAD model loader returns `Result` instead of panicking when the model file is missing.
- Streaming flush preserves the audio buffer on error instead of silently dropping buffered audio.
- Dynamic ONNX dimension probes clamp negative or invalid values to safe defaults instead of casting to `usize::MAX`, preventing out-of-memory crashes.
- WebSocket handler uses `try_checkout` per operation with proper error handling instead of holding sessions indefinitely.
- gRPC body size limited to 500 MB and concurrent streaming connections bounded by a semaphore.
- Admin endpoints (`/v1/admin/reload` and `/metrics`) require a separate `admin_api_key`; they no longer fall back to the regular API key.
- Per-field multipart upload size limits are enforced before audio processing begins.
- Rate-limiting no longer trusts `X-Forwarded-For` / `X-Real-IP` by default; the `trust_proxy` flag must be explicitly enabled behind a known reverse proxy.
- gRPC endpoints require `Authorization: Bearer <key>` authentication on every RPC method.
- CORS layer uses an explicit allowed-headers whitelist (`Content-Type`, `Authorization`, `X-Request-Id`) instead of a wildcard.
- SHA-256 checksum verification infrastructure added for model downloads; hashes can be declared in `ModelSpec` and are validated after download.

### Fixed
- Audio resample ratio bounded to a safe range (1/16 to 16×) to prevent excessive memory allocation on malformed input.
- Streaming encoder handles missing cached states and missing input tensors gracefully without panicking.
- Speech buffer in chunked streaming capped at 30 seconds to prevent unbounded memory growth during long utterances.
- Model archive extraction validated against directory traversal during auto-download.
- Model configuration auto-generated only when the target path is a valid directory.
- FFI errors use structured `tracing` logging instead of raw `eprintln` output.
- Streaming pipeline includes a compile-time `Send` safety assertion.
- Encoder zero-state tensor creation handles allocation failures gracefully instead of panicking.
- Explicit CORS allowed-headers whitelist replaces wildcard `Any`.
- Session pool `PoolGuard` drop warns when an item cannot be returned to the pool.
- `OwnedReservation` checkin uses blocking send for reliable session return.

### Added
- **gRPC API** — `phonex.TranscriptionService` with `Transcribe` (offline) and `StreamTranscribe` (bi-di streaming) RPCs. Enabled via `--grpc-port` and `grpc` feature.
- **OpenAPI / Swagger UI** — auto-generated docs at `/docs` and `/openapi.json` (REST API). Powered by `utoipa`.
- **Word-level timestamps in streaming** — `DecodeToken` carries `start`/`end` seconds per word, exposed through gRPC streaming.
- **CUDA execution provider** — `cuda` feature enables NVIDIA GPU acceleration via ONNX Runtime CUDA EP. Automatically selected at session load time on Linux.
- **Model hot-swap** — `POST /v1/admin/reload` replaces the loaded model without server restart. New requests immediately use the new model; in-flight requests finish on the old engine before the pool is recycled.
- **Speaker diarization** — integrated `polyvoice` for speaker diarization. Add `?diarize=true` to `/v1/transcribe` or `--diarize` to CLI. Requires `diarization` feature and a speaker embedding ONNX model (e.g. WeSpeaker ResNet34).
- **Streaming for all languages** — unified WebSocket endpoint auto-detects streaming vs offline models. Streaming models use true Zipformer streaming (~320 ms latency). Offline models use VAD-triggered chunked pseudo-streaming (~500–800 ms utterance latency). Added French, German, Spanish, Bengali, Chinese streaming, Korean streaming to the server Language enum.
- **YAML configuration** — `phonex.yaml` config file with `--config` flag. Load order: defaults → config → env → CLI.
- **TLS** — HTTPS and gRPC-over-TLS via `--features tls` and `tls.cert` / `tls.key` in config.
- **Docker Compose** — bundled Prometheus + Grafana stack with pre-loaded dashboard.
- **`CheckoutGuard`** — safe RAII guard for session pool checkouts that auto-returns items on drop.
- **`admin_api_key`** — separate API key for privileged endpoints (`/v1/admin/reload`, `/metrics`).
- **`trust_proxy`** — configuration flag to trust `X-Forwarded-For` / `X-Real-IP` headers for rate-limiting (only enable behind a trusted reverse proxy).
- **gRPC API key authentication** — Bearer token validation on every gRPC method via request metadata.
- **SHA-256 model verification infrastructure** — `ModelSpec` supports an optional `sha256` field; downloaded archives are verified against the declared hash before extraction.

## [0.2.3] - 2025-05-07

### Fixed
- **FFI build** — gate server/protocol modules behind `server` feature for clean `--no-default-features` builds

## [0.2.1] - 2025-05-06

### Added
- **FFI layer** — C-ABI for embedding into Android, iOS, Python, C/C++
- **Python bindings** — `bindings/python/phonex.py` with `Engine` and `Stream` classes
- **C header** — `phonex.h` with full API reference
- **Pre-built binaries** — GitHub Actions workflow builds for linux/amd64, linux/arm64, macOS/arm64
- **Benchmark docs** — comparison with whisper.cpp
- **10 language models** support — docs for Thai, English, Chinese, German, French, Spanish, Portuguese, Russian, Japanese, Korean
- **Docs site** — `docs/API.md`, `docs/MODELS.md`, `docs/BUILD.md`, `docs/FAQ.md`, `docs/SECURITY.md`, `docs/FFI.md`, `docs/BENCHMARK.md`

### Changed
- `Cargo.lock` committed for reproducible builds
- Dockerfile: Rust 1.82 → 1.95, added `cmake` and `build-essential`

### Fixed
- **CI: Docker build** — fixed `Cargo.lock` and `cmake` missing in Dockerfile
- **CI: VAD tests** — download `silero_vad.onnx` in CI workflow

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

[unreleased]: https://github.com/ekhodzitsky/phonex/compare/v0.2.4...HEAD
[0.2.4]: https://github.com/ekhodzitsky/phonex/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/ekhodzitsky/phonex/compare/v0.2.1...v0.2.3
[0.2.1]: https://github.com/ekhodzitsky/phonex/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/ekhodzitsky/phonex/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ekhodzitsky/phonex/releases/tag/v0.1.0
