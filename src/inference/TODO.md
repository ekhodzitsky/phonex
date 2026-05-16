# inference TODO

- Add focused unit tests for resampling edge cases and finite sample handling.
- Add pool lifecycle tests for checkout, close, and owned reservation behavior.
- Decide whether model-backed integration tests should use optional fixtures or
  remain manual because ONNX model files are large.
- Keep server-facing result shape changes synchronized with HTTP, gRPC, FFI,
  Python bindings, docs, and examples.
