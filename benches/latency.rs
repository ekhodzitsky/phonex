use criterion::{criterion_group, criterion_main, Criterion};

fn bench_decode_audio(c: &mut Criterion) {
    // Placeholder benchmark
    c.bench_function("noop", |b| b.iter(|| std::hint::black_box(1 + 1)));
}

criterion_group!(benches, bench_decode_audio);
criterion_main!(benches);
