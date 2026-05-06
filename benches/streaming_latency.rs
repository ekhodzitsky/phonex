use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use ndarray::Array3;
use phonex::streaming_encoder::StreamingEncoder;

const STREAMING_MODEL_DIR: &str = "models/sherpa-onnx-streaming-zipformer-en-2023-06-21";

/// Build a 5-second audio clip into precomputed encoder chunks.
///
/// At 16 kHz with 10 ms frame shift, 5 seconds = 500 frames.
/// With chunk_frames=39 and chunk_shift=32, this yields 15 chunks.
fn make_chunks(encoder: &StreamingEncoder) -> Vec<Array3<f32>> {
    let chunk_frames = encoder.chunk_frames();
    let n_mels = 80;
    let total_frames: usize = 500; // 5 seconds @ 10 ms frame shift
    let chunk_shift = encoder.chunk_shift();

    let n_chunks = (total_frames.saturating_sub(chunk_frames) / chunk_shift) + 1;
    (0..n_chunks)
        .map(|_| Array3::from_elem((1, chunk_frames, n_mels), 0.0f32))
        .collect()
}

fn bench_streaming_encoder(c: &mut Criterion) {
    // Discover encoder path
    let paths = phonex::model_config::discover_model_files(STREAMING_MODEL_DIR)
        .expect("failed to discover model files");
    let encoder_path = paths.encoder.to_str().unwrap();

    // Precompute chunks once using a temporary encoder (values don't matter for timing)
    let probe = StreamingEncoder::new(encoder_path).expect("failed to load probe encoder");
    let chunks = make_chunks(&probe);

    // ---- CPU ----
    let mut group = c.benchmark_group("streaming_encoder");
    group.sample_size(20);
    group.bench_function("cpu", |b| {
        b.iter_batched(
            || {
                let session = phonex::session::load_onnx_session_cpu(encoder_path)
                    .expect("failed to load CPU session");
                StreamingEncoder::from_session(session).expect("failed to create CPU encoder")
            },
            |mut encoder| {
                for chunk in &chunks {
                    encoder.encode_chunk(chunk).expect("encode_chunk failed");
                }
            },
            BatchSize::SmallInput,
        )
    });

    // ---- CoreML (best available) ----
    group.bench_function("coreml", |b| {
        b.iter_batched(
            || {
                let session = phonex::session::load_onnx_session(encoder_path)
                    .expect("failed to load CoreML session");
                StreamingEncoder::from_session(session).expect("failed to create CoreML encoder")
            },
            |mut encoder| {
                for chunk in &chunks {
                    encoder.encode_chunk(chunk).expect("encode_chunk failed");
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(benches, bench_streaming_encoder);
criterion_main!(benches);
