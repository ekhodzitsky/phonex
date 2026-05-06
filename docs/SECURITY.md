# Security

## Authentication

Set `--api-key` or the `PHONEX_API_KEY` env var to require `Authorization: Bearer <key>` on all endpoints:

```bash
phonex serve --api-key "sk-123456"
```

## CORS

By default only `localhost:3000` and `localhost:5173` are allowed. Customize with:

```bash
phonex serve --cors-origins "https://myapp.com,https://app.myapp.com"
```

## Rate Limiting

Per-IP token bucket. Enable via code (not yet exposed in CLI):

```rust
RuntimeLimits {
    rate_limit_per_minute: Some(60),
    ..RuntimeLimits::default()
}
```

## Input Validation

- WebSocket binary frames are validated for NaN/Inf
- Malformed audio is rejected with error code `invalid_audio_samples`
- Max request body: 500 MB
- Max concurrent WebSocket connections: 100
- Audio buffer backpressure limit: 30 seconds
