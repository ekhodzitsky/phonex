# Build from Source

## Requirements

- Rust 1.93+
- ONNX Runtime (dynamic linking recommended)

## macOS

```bash
brew install onnxruntime
export ORT_PREFER_DYNAMIC_LINK=1
export ORT_LIB_PATH=$(brew --prefix onnxruntime)/lib
export DYLD_LIBRARY_PATH=$ORT_LIB_PATH

cargo build --release
```

> **Note:** `ORT_PREFER_DYNAMIC_LINK=1` avoids a protobuf symbol collision between ONNX Runtime and SentencePiece.

## Apple Silicon (CoreML)

```bash
cargo build --release --features coreml
```

CoreML support is experimental. For Zipformer streaming models, CPU is currently faster on M-series Macs (~11 ms vs ~66 ms per chunk).

## Linux

See the [Dockerfile](../Dockerfile) for a complete Linux build environment.

```bash
# Ubuntu/Debian example
sudo apt-get install libonnxruntime-dev
cargo build --release --features server
```

## Windows

Not yet tested. If you try it, please open an issue with results.

## Development

```bash
# Run all tests
cargo test --features server

# Run integration tests (real ONNX inference)
cargo test --test integration
cargo test --test server_inference -- --ignored

# Format and lint
cargo fmt
cargo clippy --all-targets --features server
```

## Binaries

After build, binaries are in `target/release/`:
- `phonex` — main CLI (transcribe + serve)
- `server` — standalone HTTP/WebSocket server
- `streaming` — real-time streaming CLI
