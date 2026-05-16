---
schema_version: 1
kind: module_contract
module: inference
level: subsystem
layer: runtime
purpose: "Own offline and streaming ONNX inference: audio decoding, feature extraction, session pooling, greedy decode, and transcription result assembly."
status: pilot
owners:
  - phonex-maintainers
workcell:
  type: leaf
  context_path: .
  children: []
  owns_paths:
    - src/inference/
  context_budget:
    max_files: 16
    max_source_lines: 1600
    max_contract_lines: 180
    max_readme_lines: 120
    max_todo_lines: 80
    max_surfaces: 6
    max_invariants: 6
authority:
  write_policy: single_active_write_lease
  orchestrator: project
  read_agents: many_allowed
  migration_lease_required:
    - public inference API changes
    - cross-workcell model lifecycle changes
    - server protocol response shape changes
surface:
  - name: Engine
    kind: rust-type
    visibility: public
    contract: Loads ONNX session triplets, computes mel features, runs greedy RNNT decode, and returns structured transcription results.
    proof:
      kind: static-check
      target: src/inference/engine.rs
      command: cargo check --all-targets
  - name: SessionPool
    kind: rust-type
    visibility: public
    contract: Owns checkout and return semantics for reusable ONNX session triplets.
    proof:
      kind: static-check
      target: src/inference/pool.rs
      command: cargo check --all-targets
  - name: resample
    kind: rust-function
    visibility: public
    contract: Converts mono f32 audio between sample rates while preserving finite sample output.
    proof:
      kind: static-check
      target: src/inference/audio.rs
      command: cargo check --all-targets
dependencies:
  internal:
    - module: model_config
      scope: model metadata and discovered ONNX file paths
      reason: Engine construction depends on model shape, tokenizer, and session file discovery.
    - module: decoder
      scope: greedy RNNT decode
      reason: Inference uses decoder and joiner sessions to produce text tokens.
    - module: server
      scope: HTTP, SSE, WebSocket, and gRPC request handling
      reason: Server handlers call inference surfaces and expose their result shape.
  external:
    - name: ort
      scope: ONNX Runtime sessions
      reason: Inference execution is delegated to ONNX Runtime.
    - name: kaldi-native-fbank
      scope: mel feature extraction
      reason: Zipformer models expect Kaldi-compatible 80-bin fbank features.
    - name: rubato
      scope: audio resampling
      reason: Input audio can arrive at multiple sample rates.
consumers:
  - path: src/lib.rs
    uses:
      - Engine
  - path: src/inference/mod.rs
    uses:
      - SessionPool
      - resample
  - path: src/server/http.rs
    uses:
      - Engine
  - path: src/server/ws.rs
    uses:
      - Engine
  - path: src/server/grpc.rs
    uses:
      - Engine
  - path: tests/server_inference.rs
    uses:
      - Engine
invariants:
  - id: target-sample-rate
    rule: Inference-facing audio must target 16 kHz unless an explicit model configuration says otherwise.
    proof:
      kind: static-check
      target: src/inference/mod.rs
      command: cargo check --all-targets
  - id: pool-checkout-return
    rule: Checked-out session triplets must be returned to the pool when guards are dropped or checked back in explicitly.
    proof:
      kind: static-check
      target: src/inference/pool.rs
      command: cargo check --all-targets
  - id: no-cloud-side-effects
    rule: Inference must remain local and must not send audio, features, or transcripts to network services.
    proof:
      kind: manual
      target: src/inference/
      command: review inference changes for network clients and external side effects
verification:
  pre_change:
    - cargo check --all-targets
  full:
    - cargo test
    - coad check .
agent_policy:
  allowed_mutations:
    - Change inference internals while preserving public result types and server-facing behavior.
    - Add tests or benchmarks that exercise inference without requiring cloud services.
  forbidden_mutations:
    - Introduce network calls from inference code.
    - Change public transcription response semantics without updating server, FFI, Python, docs, and tests.
    - Hide model loading or ONNX Runtime failures behind empty transcripts.
  escalation:
    - Public API changes to Engine, SessionPool, TranscribeResult, WordInfo, or TranscriptSegment.
    - Changes that require model files in normal unit tests.
    - Changes that alter authentication, server response shape, or FFI/Python compatibility.
---

# inference

The inference workcell owns local ONNX speech-to-text execution.
