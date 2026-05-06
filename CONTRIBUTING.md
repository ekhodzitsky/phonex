# Contributing to phonex

Thank you for your interest in contributing! This document will help you get started.

## Development setup

### Prerequisites

- Rust 1.93+ (install via [rustup](https://rustup.rs/))
- ONNX Runtime (macOS: `brew install onnxruntime`, Linux: see Dockerfile)
- Git LFS (if working with large test audio files)

### Environment variables

```bash
export ORT_PREFER_DYNAMIC_LINK=1
export ORT_LIB_PATH=/opt/homebrew/Cellar/onnxruntime/1.25.1/lib  # macOS
export DYLD_LIBRARY_PATH=$ORT_LIB_PATH                            # macOS
```

### Download models for testing

```bash
cd models

# Thai model (offline)
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-thai-2024-06-20.tar.bz2
tar xf sherpa-onnx-zipformer-thai-2024-06-20.tar.bz2

# English streaming model
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2
tar xf sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2
```

### Build

```bash
cargo build --release
```

### Run tests

```bash
# Unit tests (fast, no model load)
cargo test --lib

# Integration tests (require models)
cargo test --test integration
cargo test --test server_inference -- --ignored

# HTTP server tests (fast, no model load)
cargo test --test server

# All tests
cargo test --all-targets
```

### Lint and format

```bash
cargo fmt
cargo clippy --all-targets --all-features
```

## Project structure

```
src/
  main.rs              # phonex CLI (transcribe, serve)
  lib.rs               # Library exports
  bin/
    server.rs          # Standalone HTTP/WebSocket server
    streaming.rs       # Real-time streaming CLI
  inference/
    engine.rs          # Offline inference engine
    features.rs        # FBANK feature extraction
    pool.rs            # ONNX session pool
    decode.rs          # Greedy RNNT decoder
    tokenizer.rs       # BPE tokenization
    streaming.rs       # Streaming inference logic
  server/
    http.rs            # HTTP handlers (REST API)
    ws.rs              # WebSocket handlers
    mod.rs             # Router builder
    metrics.rs         # Prometheus metrics
    rate_limit.rs      # Token bucket rate limiter
  streaming_encoder.rs # Stateful Zipformer encoder
  streaming_decoder.rs # Streaming decoder
  streaming_pipeline.rs # End-to-end streaming pipeline
  vad.rs               # Silero VAD + streaming VAD
  model_config.rs      # Auto-detect model parameters
```

## How to contribute

1. **Fork** the repository
2. **Create a branch** (`git checkout -b feature/my-feature`)
3. **Make your changes** with clear commit messages
4. **Run tests and clippy** (`cargo test --all-targets && cargo clippy --all-targets --all-features`)
5. **Open a Pull Request** with a clear description

## Reporting bugs

Please include:
- OS and architecture (e.g., macOS ARM64, Ubuntu x64)
- Rust version (`rustc --version`)
- ONNX Runtime version
- Model name and size
- Steps to reproduce
- Expected vs actual behavior
- Logs (with `RUST_LOG=debug` if possible)

## Adding a new language model

phonex auto-detects model parameters from ONNX metadata. To add a new language:

1. Download a Sherpa-ONNX Zipformer-transducer model
2. Place `encoder*.onnx`, `decoder*.onnx`, `joiner*.onnx`, `*.model`, `tokens.txt` in a directory
3. Run `phonex transcribe audio.wav --model-dir models/my-model`
4. If auto-detection fails, open an issue with the model link

## Code style

- Follow `rustfmt` and `clippy` without warnings
- Use `tracing` for logging, not `println!`
- Prefer `Result` over panics in library code
- Document public APIs with rustdoc comments

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
