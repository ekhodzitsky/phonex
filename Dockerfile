# Multi-stage Dockerfile for phonex
# Supports linux/amd64 and linux/arm64

ARG RUST_VERSION=1.95
ARG ONNXRUNTIME_VERSION=1.25.1
ARG TARGETARCH

# ---------------------------------------------------------------------------
# Builder stage
# ---------------------------------------------------------------------------
FROM rust:${RUST_VERSION}-bookworm AS builder

ARG ONNXRUNTIME_VERSION
ARG TARGETARCH

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    curl \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Download ONNX Runtime libraries
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        ONNX_ARCH="aarch64"; \
    else \
        ONNX_ARCH="x64"; \
    fi && \
    curl -fsSL \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-${ONNX_ARCH}-${ONNXRUNTIME_VERSION}.tgz" \
        -o /tmp/onnxruntime.tgz && \
    tar -xzf /tmp/onnxruntime.tgz -C /opt && \
    mv /opt/onnxruntime-linux-${ONNX_ARCH}-${ONNXRUNTIME_VERSION} /opt/onnxruntime

WORKDIR /usr/src/phonex
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY benches ./benches
COPY tests ./tests

ENV ORT_PREFER_DYNAMIC_LINK=1
ENV ORT_LIB_PATH=/opt/onnxruntime/lib

RUN cargo build --release --bin server

# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim

ARG ONNXRUNTIME_VERSION
ARG TARGETARCH

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Download ONNX Runtime libraries for runtime
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        ONNX_ARCH="aarch64"; \
    else \
        ONNX_ARCH="x64"; \
    fi && \
    curl -fsSL \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-${ONNX_ARCH}-${ONNXRUNTIME_VERSION}.tgz" \
        -o /tmp/onnxruntime.tgz && \
    tar -xzf /tmp/onnxruntime.tgz -C /opt && \
    mv /opt/onnxruntime-linux-${ONNX_ARCH}-${ONNXRUNTIME_VERSION} /opt/onnxruntime && \
    rm /tmp/onnxruntime.tgz

WORKDIR /app

COPY --from=builder /usr/src/phonex/target/release/server ./server

ENV ORT_PREFER_DYNAMIC_LINK=1
ENV ORT_LIB_PATH=/opt/onnxruntime/lib
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib
ENV MODEL_DIR=/app/models

# Create models directory; the app will auto-download on first start if empty
RUN mkdir -p /app/models

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -fs http://localhost:8080/health || exit 1

ENTRYPOINT ["./server"]
CMD ["--bind", "0.0.0.0", "--port", "8080", "--pool-size", "2"]
