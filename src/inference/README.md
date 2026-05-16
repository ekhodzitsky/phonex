# inference

Owns local speech-to-text execution after the application has selected a model
and provided audio bytes or samples.

Primary responsibilities:

- decode supported audio formats into mono f32 samples;
- resample audio to the model sample rate;
- compute Kaldi-compatible mel features;
- manage reusable ONNX encoder, decoder, and joiner sessions;
- run greedy RNNT decoding;
- return `TranscribeResult`, `WordInfo`, and transcript segments to callers.

Public surfaces are re-exported through `src/inference/mod.rs` and partly
through `src/lib.rs`. Server handlers consume `Engine` through shared state.

Do not add network behavior here. Model downloads, server protocols,
authentication, and external API response shapes belong to neighboring
workcells.

Useful verification:

```bash
cargo check --all-targets
cargo test
coad check .
```
