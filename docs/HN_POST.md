# Hacker News Post Draft

**Title:** Show HN: phonex — on-device speech-to-text in Rust, now with FFI and Python bindings

**Body:**

phonex is a generic on-device speech-to-text engine built in Rust. It runs any Sherpa-ONNX Zipformer-transducer model locally — no cloud APIs, no API keys, full privacy.

**What makes it different:**

- **True streaming inference** — stateful Zipformer encoder with 35 cached tensors, real-time chunk-by-chunk decode. Latency ~11ms per 500ms audio chunk (CPU).
- **Generic model support** — auto-detects model parameters from ONNX metadata. Drop in any Sherpa-ONNX Zipformer model (Thai, English, Chinese, German, etc.).
- **FFI + Python bindings** — embed into Android, iOS, Python, or any C runtime. No HTTP server needed.
- **Single binary** — CLI + HTTP/WebSocket server in one Rust crate.

**Quick demo:**

```bash
cargo install phonex --git https://github.com/ekhodzitsky/phonex
phonex transcribe audio.wav
```

**Benchmarks** (M1 Pro, 11s JFK speech):

| Engine | Model | Size | Latency | Streaming |
|--------|-------|------|---------|-----------|
| phonex | Zipformer EN INT8 | ~68 MB | ~2.5s | ✅ Native |
| whisper.cpp | tiny.en | ~80 MB | ~0.9s | ❌ No |

whisper.cpp is faster for offline batch (GPU-accelerated), but phonex is the only option with native streaming in this size class.

**Links:**

- Repo: https://github.com/ekhodzitsky/phonex
- FFI docs: https://github.com/ekhodzitsky/phonex/blob/main/docs/FFI.md
- Python bindings: https://github.com/ekhodzitsky/phonex/tree/main/bindings/python

**Tech stack:** Rust, ONNX Runtime, kaldi-native-fbank, SentencePiece BPE.

Feedback welcome!
