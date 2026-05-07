# Contributing to phonex

Thank you for considering contributing to phonex! This document outlines the process and guidelines for contributing.

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally.
3. Ensure you have Rust 1.75+ and ONNX Runtime installed.

## Development Setup

```bash
# Clone your fork
git clone https://github.com/<your-username>/phonex.git
cd phonex

# Build
cargo build --all-features

# Run tests
cargo test --all-features

# Run clippy and fmt
cargo fmt -- --check
cargo clippy --all-features -- -D warnings

# Run benchmarks (mock mode — no ONNX models required)
cargo bench
```

## Making Changes

1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make your changes.
3. Add tests for any new functionality.
4. Update `CHANGELOG.md` under the `[Unreleased]` section.
5. Ensure all tests pass and clippy is clean.

## Code Style

- Format code with `cargo fmt`.
- Address all `cargo clippy --all-features -- -D warnings` lints.
- Write doc comments for all public API items.
- Avoid `unwrap()` in production code paths; use `?` or `match` instead.
- Add `// SAFETY:` comments for any `unsafe` blocks.

## Testing

- Unit tests go in `src/<module>/mod.rs` inside `#[cfg(test)]` modules.
- Integration tests go in `tests/`.
- Tests that require ONNX model files should be marked with `#[ignore = "..."]`.
- Run `cargo test --all-features` before submitting.

## Security

If you discover a security vulnerability, please follow our [Security Policy](SECURITY.md) instead of opening a public issue.

## Pull Request Process

1. Push your branch to your fork.
2. Open a Pull Request against `main`.
3. Ensure CI passes (fmt, clippy, test, audit, cargo-deny).
4. Request review from maintainers.
5. Once approved, your PR will be squash-merged.

## Release Process

Maintainers handle releases. Do not bump version numbers in PRs.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
