//! Benchmark `Pool::checkout` + `into_owned` + drop cycle.
//!
//! This validates that the new `CheckoutGuard` does not add significant
//! overhead compared to a plain `PoolGuard` drop.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use phonex::inference::pool::Pool;

fn bench_pool_cycle(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("pool");
    group.sample_size(100);
    group.measurement_time(std::time::Duration::from_secs(10));

    // -----------------------------------------------------------------
    // 1. checkout → into_owned → drop (CheckoutGuard path)
    //    This is the full cycle used by ChunkedStreamingPipeline:
    //    pool.checkout().await → guard.into_owned() → spawn_blocking
    //    → CheckoutGuard drops inside the blocking task.
    // -----------------------------------------------------------------
    let pool = Pool::new((0..4).collect::<Vec<usize>>());
    group.throughput(Throughput::Elements(1));
    group.bench_function("checkout_into_owned_drop", |b| {
        b.to_async(&rt).iter(|| async {
            let guard = pool.checkout().await.expect("pool checkout");
            let _owned = guard.into_owned();
            // _owned drops here, returning item to pool via async-channel
        });
    });

    // -----------------------------------------------------------------
    // 2. checkout → deref → drop (PoolGuard path, no into_owned)
    //    Baseline: no CheckoutGuard conversion.
    // -----------------------------------------------------------------
    let pool2 = Pool::new((0..4).collect::<Vec<usize>>());
    group.bench_function("checkout_deref_drop", |b| {
        b.to_async(&rt).iter(|| async {
            let guard = pool2.checkout().await.expect("pool checkout");
            std::hint::black_box(*guard);
            // guard drops here, returning item to pool
        });
    });

    // -----------------------------------------------------------------
    // 3. try_checkout → deref → drop (non-blocking)
    // -----------------------------------------------------------------
    let pool3 = Pool::new((0..4).collect::<Vec<usize>>());
    group.bench_function("try_checkout_deref_drop", |b| {
        b.iter(|| {
            let guard = pool3.try_checkout().expect("pool try_checkout");
            std::hint::black_box(*guard);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_pool_cycle);
criterion_main!(benches);
