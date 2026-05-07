# Security Policy

## Supported Versions

The following versions of phonex receive security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.2.4+  | :white_check_mark: |
| < 0.2.4 | :x:                |

Versions prior to 0.2.4 contain unpatched security findings. Please upgrade to 0.2.4 or later.

## Reporting a Vulnerability

If you discover a security vulnerability in phonex, please report it responsibly:

- **Email**: e@khodzitsky.ru
- **GitHub Security Advisory**: [github.com/ekhodzitsky/phonex/security/advisories](https://github.com/ekhodzitsky/phonex/security/advisories)

Please do not open public issues for security vulnerabilities. We aim to respond within 48 hours and will coordinate a disclosure timeline with you.

## Known Security Considerations

### API Key Authentication

phonex uses Bearer token authentication. Set `PHONEX_API_KEY` (or `--api-key`) to require `Authorization: Bearer <key>` on all endpoints.

### Admin Endpoints

Privileged endpoints (`/v1/admin/reload`, `/metrics`) require a separate `admin_api_key`. Set `PHONEX_ADMIN_API_KEY` (or `--admin-api-key`) independently from the regular API key. If no admin key is configured, admin endpoints fall back to the regular API key (not recommended for production).

### Rate Limiting and Reverse Proxies

Rate limiting is per-IP. If phonex runs behind a reverse proxy (nginx, Traefik, etc.), enable the `trust_proxy` flag so the rate limiter reads the client IP from `X-Forwarded-For` or `X-Real-IP`. **Only enable `trust_proxy` when phonex is behind a trusted reverse proxy** — otherwise clients can spoof their IP and bypass rate limits.

```bash
phonex serve --trust-proxy
```

### Model Reload Path Validation

The `POST /v1/admin/reload` endpoint accepts an optional `model_dir` query parameter. The server validates that:

1. The path is absolute.
2. The path does not contain `..` parent-directory components.
3. The path exists and is a directory.

This prevents path-traversal attacks that could load arbitrary ONNX models from the filesystem.

### ONNX Runtime Dynamic Linking

phonex links against ONNX Runtime dynamically. When building from source, ensure `ORT_PREFER_DYNAMIC_LINK=1` is set in your environment so the build system links against the system ONNX Runtime instead of vendoring a static copy. This reduces binary size and ensures you receive ONNX Runtime security updates through your system package manager.

### FFI Memory Safety

When using the C-ABI FFI layer (`phonex.h`), callers **must** call `phonex_string_free` exactly once for every string returned by the API, and `phonex_engine_free` / `phonex_stream_free` exactly once per handle. The library includes internal `disposed` guards to mitigate double-free and use-after-free, but correct caller behavior is required for full safety.

### Model Integrity

phonex supports SHA-256 checksum verification for downloaded models. Known models can declare an expected `sha256` in `ModelSpec`; after download the archive hash is validated before extraction. You can also verify models manually before loading them.

## Security Audit History

A comprehensive security audit was conducted in May 2026. All Critical and High severity findings were resolved in v0.2.4. The audit covered:

- Authentication and authorization
- Path traversal and archive extraction
- Async runtime safety and resource exhaustion
- FFI memory safety
- Input validation and rate limiting
- Dependency and build chain security

See [CHANGELOG.md](CHANGELOG.md) for the detailed list of security fixes included in v0.2.4.
