use criterion::{Criterion, criterion_group, criterion_main};

fn bench_decode_audio(c: &mut Criterion) {
    // Placeholder benchmark
    c.bench_function("noop", |b| b.iter(|| std::hint::black_box(1 + 1)));
}

criterion_group!(benches, bench_decode_audio);
criterion_main!(benches);
