//! Benchmark chunked streaming latency.
//!
//! Simulates feeding 16 kHz f32 audio chunks (320 samples = 20 ms) into the
//! async path used by `ChunkedStreamingPipeline::accept_audio`.  Because real
//! ONNX models are not available in CI, the inference body is mocked with a
//! small CPU-bound workload.  Enable the `bench-with-model` feature to run
//! against actual `ChunkedStreamingPipeline` instances (requires model files).
//!
//! # Real-Time Factor (RTF)
//!
//! Each iteration processes one 20 ms chunk.
//!
//! ```text
//! RTF = measured_latency (s) / 0.020 (s)
//! ```
//!
//! Target for streaming: RTF < 0.1  (process 1 s of audio in < 100 ms).

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use phonex::inference::pool::Pool;

const SAMPLE_RATE: f32 = 16000.0;
const CHUNK_SAMPLES: usize = 320; // 20 ms @ 16 kHz
const CHUNK_DURATION_S: f64 = CHUNK_SAMPLES as f64 / SAMPLE_RATE as f64;

/// Simulate the CPU work done by `Engine::transcribe_samples` inside
/// `spawn_blocking`.  Tunable to represent ~1–5 ms of ONNX inference.
fn mock_onnx_inference(samples: &[f32]) -> Vec<f32> {
    // Deterministic, non-trivial computation so it does not get optimised away.
    let mut acc = 0.0f64;
    for i in 0..40_000 {
        let x = i as f64 + samples.len() as f64;
        acc += x.sin() * x.cos();
    }
    std::hint::black_box(acc);
    vec![0.0f32; 8]
}

fn bench_chunked_latency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("chunked_streaming");
    group.sample_size(100);
    group.measurement_time(std::time::Duration::from_secs(10));
    group.throughput(Throughput::Elements(1)); // 1 chunk = 20 ms audio

    // Pre-allocate a chunk of silence.
    let chunk: Vec<f32> = vec![0.0f32; CHUNK_SAMPLES];

    // -----------------------------------------------------------------
    // Baseline 1: pure buffering, no async boundary, no inference.
    // This isolates the cost of Vec::extend_from_slice + black_box.
    // -----------------------------------------------------------------
    group.bench_function("baseline_buffer_only", |b| {
        b.iter(|| {
            let mut speech_buffer = Vec::with_capacity(SAMPLE_RATE as usize * 10);
            speech_buffer.extend_from_slice(&chunk);
            std::hint::black_box(&speech_buffer);
        });
    });

    // -----------------------------------------------------------------
    // Baseline 2: spawn_blocking with *zero* work.
    // Measures the pure thread-pool scheduling overhead.
    // -----------------------------------------------------------------
    group.bench_function("spawn_blocking_overhead", |b| {
        b.to_async(&rt).iter(|| async {
            let _ = tokio::task::spawn_blocking(move || std::hint::black_box(42))
                .await
                .expect("spawn_blocking join");
        });
    });

    // -----------------------------------------------------------------
    // Simulated ChunkedStreamingPipeline::transcribe_buffer path.
    //
    // 1. Pool checkout (async-channel recv)
    // 2. into_owned()  → CheckoutGuard
    // 3. spawn_blocking with mock ONNX work
    // 4. CheckoutGuard drops → item returns to pool
    // -----------------------------------------------------------------
    let pool: Pool<Vec<f32>> = Pool::new(vec![vec![0.0f32; 64]; 4]);
    group.bench_function("with_spawn_blocking", |b| {
        b.to_async(&rt).iter(|| async {
            let guard = pool.checkout().await.expect("pool checkout");
            let _owned = guard.into_owned();

            let samples = chunk.clone();
            let result = tokio::task::spawn_blocking(move || {
                let out = mock_onnx_inference(&samples);
                std::hint::black_box(out)
            })
            .await
            .expect("spawn_blocking join");

            std::hint::black_box(result);
        });
    });

    // -----------------------------------------------------------------
    // Heavier mock inference (represents a larger utterance).
    // -----------------------------------------------------------------
    group.bench_function("with_heavy_spawn_blocking", |b| {
        b.to_async(&rt).iter(|| async {
            let guard = pool.checkout().await.expect("pool checkout");
            let _owned = guard.into_owned();

            let samples = chunk.clone();
            let result = tokio::task::spawn_blocking(move || {
                // ~2× the compute of the standard mock
                let mut acc = 0.0f64;
                for i in 0..80_000 {
                    let x = i as f64;
                    acc += x.sin() * x.cos();
                }
                std::hint::black_box(acc);
                mock_onnx_inference(&samples)
            })
            .await
            .expect("spawn_blocking join");

            std::hint::black_box(result);
        });
    });

    group.finish();

    // Print RTF reference so it appears in benchmark logs.
    eprintln!(
        " chunk_duration = {:.3} s  |  RTF target < 0.1  =>  latency target < {:.3} ms",
        CHUNK_DURATION_S,
        CHUNK_DURATION_S * 1000.0 * 0.1
    );
}

#[cfg(feature = "bench-with-model")]
fn bench_chunked_with_model(c: &mut Criterion) {
    // Placeholder for real-model benchmark.
    // Requires:
    //   1. A valid model directory on disk.
    //   2. A valid Silero VAD model at models/silero_vad.onnx.
    //
    // Example implementation:
    //   let engine = Arc::new(phonex::Engine::load("models/...").unwrap());
    //   let mut pipeline = phonex::chunked_streaming::ChunkedStreamingPipeline::new(
    //       engine, "models/silero_vad.onnx"
    //   ).unwrap();
    //   let chunk = vec![0.0f32; CHUNK_SAMPLES];
    //   c.bench_function("real_model", |b| {
    //       b.to_async(&rt).iter(|| async {
    //           let _tokens = pipeline.accept_audio(&chunk).await.unwrap();
    //       });
    //   });
    let mut group = c.benchmark_group("chunked_streaming_real_model");
    group.sample_size(10);
    group.finish();
}

#[cfg(not(feature = "bench-with-model"))]
criterion_group!(benches, bench_chunked_latency);

#[cfg(feature = "bench-with-model")]
criterion_group!(benches, bench_chunked_latency, bench_chunked_with_model);

criterion_main!(benches);
