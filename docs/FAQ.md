# FAQ

## Why not just use Whisper / whisper.cpp?

Whisper models are 150 MB–3 GB and don't support native streaming (you have to buffer the entire audio). phonex uses Zipformer-transducer models (~78 MB for English streaming) with native chunk-by-chunk inference — ~500 ms latency vs Whisper's multi-second latency.

## Can I use my own model?

Yes. Drop any Sherpa-ONNX Zipformer encoder-decoder-joiner model into a directory and point `--model-dir` at it. phonex auto-detects all parameters from ONNX metadata.

## Does it run on Raspberry Pi / embedded?

Should work on any Linux ARM64 with ONNX Runtime. Not yet tested on Raspberry Pi — if you try it, open an issue with results.

## Will more languages get streaming support?

phonex is ready — we just need Sherpa-ONNX to release streaming Zipformer models for Japanese, Russian, Thai, Vietnamese, and Cantonese. When they do, they will work with zero code changes.

## How does phonex compare to running sherpa-onnx Python?

phonex is a single static binary with no Python environment. Easier to deploy, harder to break, lower memory overhead. Feature parity for inference; phonex adds a built-in REST/WebSocket server.

## Can I use multiple languages at the same time?

Currently each phonex process loads one model. To serve multiple languages, run multiple server instances on different ports, or use a reverse proxy (e.g. nginx) to route by language.
