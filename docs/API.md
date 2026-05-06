# API Reference

## Server Options

```bash
phonex serve \
  --language english          # Built-in language model
  --model-dir models/my-model # Override with custom path
  --bind 127.0.0.1            # Bind address
  --port 8080                 # Listen port
  --pool-size 2               # Parallel ONNX sessions
  --api-key "sk-xxx"          # Bearer token auth
  --cors-origins "https://a.com,https://b.com"
```

## HTTP Endpoints

### Health check

```bash
curl http://localhost:8080/health
```

```json
{"status":"ok","model":"sherpa-onnx-zipformer-en-2023-06-26","version":"0.2.0"}
```

### Model info

```bash
curl http://localhost:8080/v1/models
```

### Transcribe audio (multipart upload)

Accepts raw mono f32 LE bytes and optional `sample_rate` (default 16000 Hz).

```bash
curl -X POST http://localhost:8080/v1/transcribe \
  -F "audio=@audio.raw" \
  -F "sample_rate=16000"
```

Response:
```json
{"text":"hello world","words":[],"duration":3.52}
```

### Transcribe with VAD

```bash
curl -X POST 'http://localhost:8080/v1/transcribe?vad=true' \
  -F "audio=@audio.raw" \
  -F "sample_rate=16000"
```

### Batch transcribe

```bash
curl -X POST http://localhost:8080/v1/transcribe/batch \
  -F "audio=@1.raw" \
  -F "audio=@2.raw" \
  -F "sample_rate=16000"
```

### SSE streaming

```bash
curl -N -X POST http://localhost:8080/v1/transcribe/stream \
  -F "audio=@audio.raw" \
  -F "sample_rate=16000"
```

Emits `partial` and `final` events:
```json
{"type":"partial","text":"hello","timestamp":1234567890.0}
{"type":"final","text":"hello world","timestamp":1234567890.0}
```

## WebSocket Streaming

**Endpoint:** `ws://localhost:8080/v1/transcribe/stream`

Send:
- **Binary frames**: raw f32 LE audio samples at 16 kHz mono
- **JSON commands**:
  - `{"cmd":"Clear"}` — reset pipeline
  - `{"cmd":"Stop"}` — finalize and close

Server replies:
```json
{"type":"ready","model":"...","sample_rate":16000}
{"type":"partial","text":"HELLO","timestamp":1234567890.0}
{"type":"final","text":"HELLO WORLD","timestamp":1234567890.0}
```

**Legacy endpoint:** `ws://localhost:8080/stream` — same protocol with offline engine.

**Limits:** max 100 concurrent WebSocket connections, idle timeout 60s.
