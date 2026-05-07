use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use ndarray::Array2;
use phonex::audio::AudioPreprocessor;

const TEST_WAV_PATH: &str = "models/sherpa-onnx-streaming-zipformer-en-2023-06-21/test_wavs/0.wav";

fn make_dummy_preprocessor() -> AudioPreprocessor {
    // resample() does not touch the mel filterbank, so a zero-filled dummy is fine.
    let dummy_fb = Array2::zeros((80, 257));
    AudioPreprocessor::new(16000, 512, 160, 400, 80, dummy_fb)
}

fn bench_read_wav(c: &mut Criterion) {
    if !std::path::Path::new(TEST_WAV_PATH).is_file() {
        eprintln!("Skipping read_wav benchmark — {} not found", TEST_WAV_PATH);
        return;
    }

    let mut group = c.benchmark_group("audio_preprocessor");
    group.sample_size(50);
    group.throughput(Throughput::Elements(1));
    group.bench_function("read_wav", |b| {
        b.iter(|| {
            let (samples, rate) = AudioPreprocessor::read_wav(TEST_WAV_PATH).expect("read wav");
            std::hint::black_box((samples, rate));
        });
    });
    group.finish();
}

fn bench_resample(c: &mut Criterion) {
    let preprocessor = make_dummy_preprocessor();

    // 1 second of 44.1 kHz audio -> 16000 samples after downsampling
    let samples: Vec<f32> = (0..44100)
        .map(|i| (i as f32 / 44100.0 * 2.0 * std::f32::consts::PI).sin())
        .collect();
    let out_len = (samples.len() as f32 / (44100.0 / 16000.0)) as usize;

    let mut group = c.benchmark_group("audio_preprocessor");
    group.sample_size(100);
    group.throughput(Throughput::Elements(out_len as u64));
    group.bench_function("resample_44k1_to_16k", |b| {
        b.iter(|| {
            let out = preprocessor.resample(&samples, 44100).expect("resample");
            std::hint::black_box(out);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_read_wav, bench_resample);
criterion_main!(benches);
