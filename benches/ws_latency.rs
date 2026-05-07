//! Benchmark WS streaming latency.
//!
//! Simulates the `WsPipeline::Streaming::accept_audio` async boundary:
//! `tokio::task::spawn_blocking` wrapping `StreamingPipeline::accept_audio`.
//! Because real ONNX models are not available in CI, the pipeline body is
//! mocked.  Enable the `bench-with-model` feature to benchmark against
//! actual `StreamingPipeline` instances.
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

const SAMPLE_RATE: f32 = 16000.0;
const CHUNK_SAMPLES: usize = 320; // 20 ms @ 16 kHz
const CHUNK_DURATION_S: f64 = CHUNK_SAMPLES as f64 / SAMPLE_RATE as f64;

/// Simulates the CPU work done by `StreamingPipeline::accept_audio`.
fn mock_streaming_pipeline_accept_audio(samples: &[f32]) -> Vec<u8> {
    let mut acc = 0.0f64;
    for i in 0..30_000 {
        let x = i as f64 + samples.len() as f64;
        acc += x.sin().sqrt();
    }
    std::hint::black_box(acc);
    vec![0u8; 16]
}

/// A dummy pipeline handle that we move in/out of `spawn_blocking`.
struct MockPipeline {
    tokens: Vec<u8>,
}

impl MockPipeline {
    fn accept_audio(&mut self, samples: &[f32]) -> Vec<u8> {
        let out = mock_streaming_pipeline_accept_audio(samples);
        self.tokens.extend_from_slice(&out);
        out
    }
}

fn bench_ws_streaming_latency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("ws_streaming");
    group.sample_size(100);
    group.measurement_time(std::time::Duration::from_secs(10));
    group.throughput(Throughput::Elements(1)); // 1 chunk = 20 ms audio

    let chunk: Vec<f32> = vec![0.0f32; CHUNK_SAMPLES];

    // -----------------------------------------------------------------
    // Baseline: direct call, no spawn_blocking.
    // -----------------------------------------------------------------
    group.bench_function("baseline_direct_call", |b| {
        let mut pipeline = MockPipeline { tokens: Vec::new() };
        b.iter(|| {
            let result = pipeline.accept_audio(&chunk);
            std::hint::black_box(result);
        });
    });

    // -----------------------------------------------------------------
    // With spawn_blocking — exact pattern from ws.rs:
    //   1. take pipeline out of Option
    //   2. move into spawn_blocking
    //   3. run accept_audio
    //   4. return (pipeline, result)
    //   5. put pipeline back into Option
    // -----------------------------------------------------------------
    group.bench_function("with_spawn_blocking", |b| {
        b.to_async(&rt).iter(|| async {
            let mut pipeline_opt = Some(MockPipeline { tokens: Vec::new() });
            let samples = chunk.clone();

            let mut pipeline = pipeline_opt.take().expect("pipeline is Some");
            let (pipeline_back, result) = tokio::task::spawn_blocking(move || {
                let result = pipeline.accept_audio(&samples);
                (pipeline, result)
            })
            .await
            .expect("spawn_blocking join");

            pipeline_opt = Some(pipeline_back);
            std::hint::black_box((pipeline_opt, result));
        });
    });

    // -----------------------------------------------------------------
    // spawn_blocking overhead with zero work (just move/return).
    // -----------------------------------------------------------------
    group.bench_function("spawn_blocking_overhead", |b| {
        b.to_async(&rt).iter(|| async {
            let mut pipeline_opt = Some(MockPipeline { tokens: Vec::new() });
            let samples = chunk.clone();

            let pipeline = pipeline_opt.take().expect("pipeline is Some");
            let (pipeline_back, result) = tokio::task::spawn_blocking(move || {
                std::hint::black_box(&samples);
                (pipeline, 42u8)
            })
            .await
            .expect("spawn_blocking join");

            pipeline_opt = Some(pipeline_back);
            std::hint::black_box((pipeline_opt, result));
        });
    });

    group.finish();

    eprintln!(
        " chunk_duration = {:.3} s  |  RTF target < 0.1  =>  latency target < {:.3} ms",
        CHUNK_DURATION_S,
        CHUNK_DURATION_S * 1000.0 * 0.1
    );
}

#[cfg(feature = "bench-with-model")]
fn bench_ws_with_model(c: &mut Criterion) {
    // Placeholder for real-model WS benchmark.
    // Requires a valid streaming model directory on disk.
    //
    // Example:
    //   let rt = tokio::runtime::Runtime::new().unwrap();
    //   let info = phonex::model_config::ModelInfo::from_model_dir("models/...").unwrap();
    //   let mut pipeline = phonex::streaming_pipeline::StreamingPipeline::from_model_dir(
    //       "models/...", &info, None
    //   ).unwrap();
    //   let chunk = vec![0.0f32; CHUNK_SAMPLES];
    //   c.bench_function("real_model", |b| {
    //       b.to_async(&rt).iter(|| async {
    //           let _ = tokio::task::spawn_blocking({
    //               let samples = chunk.clone();
    //               let mut p = pipeline;
    //               move || {
    //                   let r = p.accept_audio(&samples);
    //                   (p, r)
    //               }
    //           }).await.expect("join");
    //       });
    //   });
    let mut group = c.benchmark_group("ws_streaming_real_model");
    group.sample_size(10);
    group.finish();
}

#[cfg(not(feature = "bench-with-model"))]
criterion_group!(benches, bench_ws_streaming_latency);

#[cfg(feature = "bench-with-model")]
criterion_group!(benches, bench_ws_streaming_latency, bench_ws_with_model);

criterion_main!(benches);
