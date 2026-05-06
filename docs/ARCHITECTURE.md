# Architecture

## Pipeline

```
Audio (WAV/MP3/OGG/FLAC/AAC)
  → symphonia decoder
  → resample to 16 kHz
  → kaldi-native-fbank (80-bin log-mel, 25 ms / 10 ms frames)
  → Zipformer Encoder (ONNX)
  → Greedy RNNT Decoder + Joiner (ONNX)
  → SentencePiece BPE tokenizer
  → Text
```

## Streaming Inference

For real-time streaming, the `StreamingPipeline` maintains encoder state across chunks:

```rust
use phonex::{ModelInfo, StreamingPipeline};

let info = ModelInfo::from_model_dir("models/sherpa-onnx-streaming-zipformer-en-2023-06-21")?;
let mut pipeline = StreamingPipeline::from_model_dir(
    "models/sherpa-onnx-streaming-zipformer-en-2023-06-21",
    &info,
    Some("models/silero_vad.onnx"), // optional VAD
)?;

// Feed audio in chunks (e.g. from a microphone)
pipeline.accept_audio(&samples)?;

// Finalize and get the full transcription
let text = pipeline.flush()?;
```

The pipeline maintains encoder caches (35 cached tensors) and performs greedy RNNT decode. Call `reset()` between utterances to clear internal caches.

## Offline Inference

The `Engine` uses a session pool of ONNX triplets (encoder + decoder + joiner) for parallel inference. Each request checks out a triplet, runs transcription, and returns it to the pool.

## Performance Notes

- **Offline**: RTF ~0.15–0.19× on Apple Silicon (M-series). A 5-second clip is transcribed in ~70 ms.
- **Streaming**: ~11 ms per 500 ms audio chunk (CPU). End-to-end latency is dominated by chunk size.
- **CoreML**: Currently ~6× slower than CPU for the streaming encoder on M-series. CPU is recommended.
