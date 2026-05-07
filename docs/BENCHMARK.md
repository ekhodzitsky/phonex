# Benchmarks

All measurements taken on Apple Silicon M1 Pro (macOS ARM64).

## Offline Transcription

### JFK Inaugural Address (11 seconds, English)

| Engine | Model | Size | Latency | RTF |
|--------|-------|------|---------|-----|
| **phonex** | Sherpa-ONNX Zipformer EN INT8 | ~68 MB | ~2.5s | ~0.23x |
| **whisper.cpp** | Whisper tiny.en | ~80 MB | ~0.9s | ~0.08x |

**Notes:**
- whisper.cpp uses Metal GPU acceleration on M1 Pro.
- phonex uses CPU-only inference (ONNX Runtime CPU EP).
- phonex supports **streaming** — whisper.cpp does not.

## Streaming Real-Time

| Engine | Model | Latency per 500ms chunk | RTF |
|--------|-------|------------------------|-----|
| **phonex** | Sherpa-ONNX Streaming Zipformer EN | ~78 MB | ~11ms | ~0.03x |

**Notes:**
- End-to-end latency (audio → partial text) is dominated by chunk size.
- 500ms chunks yield ~500ms latency.
- phonex processes audio **30× faster** than real-time per chunk.

## Model Size Comparison

| Engine | Model | Size | Languages | Streaming |
|--------|-------|------|-----------|-----------|
| phonex | Zipformer Thai INT8 | ~148 MB | Thai (offline) | ❌ |
| phonex | Zipformer English INT8 | ~68 MB | English (offline) | ❌ |
| phonex | Streaming Zipformer EN | ~78 MB | English | ✅ |
| whisper.cpp | tiny.en | ~80 MB | English | ❌ |
| whisper.cpp | base.en | ~150 MB | English | ❌ |
| whisper.cpp | small.en | ~500 MB | English | ❌ |

## Key Takeaways

- **whisper.cpp** is faster for offline batch transcription on Apple Silicon (GPU-accelerated).
- **phonex** is the only option with **native streaming** support in this size class.
- **phonex** models are smaller and support more languages via Sherpa-ONNX ecosystem.
- For real-time applications (live captions, voice assistants), phonex is the better fit.
