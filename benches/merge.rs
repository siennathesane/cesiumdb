use std::collections::Bound;

use bytes::Bytes;
use cesiumdb::{
    hlc::{
        HybridLogicalClock,
        HLC,
    },
    keypair::{
        KeyBytes,
        ValueBytes,
        DEFAULT_NS,
    },
    memtable::Memtable,
    merge::MergeIterator,
};
use criterion::{
    black_box,
    criterion_group,
    criterion_main,
    BatchSize,
    BenchmarkId,
    Criterion,
};
use rand::{
    thread_rng,
    Rng,
};

struct TestData {
    tables: Vec<Memtable>,
    _clock: HybridLogicalClock,
}

impl TestData {
    fn new(size: usize, memtable_count: usize) -> Self {
        let clock = HybridLogicalClock::new();
        let mut rng = thread_rng();
        let mut tables = Vec::with_capacity(memtable_count);

        // Create memtables
        for i in 0..memtable_count {
            let table = Memtable::new(i as u64, 1024 * 1024 * 256);
            tables.push(table);
        }

        // Generate data for each memtable with overlapping data
        for entry in 0..size {
            let key = format!("key_{:010}", entry);
            let val = format!("val_{}", entry);

            // Create overlapping data across memtables
            for table_idx in 0..memtable_count {
                if rng.gen_bool(0.3) {
                    // 30% chance of key appearing in each table
                    let ts = clock.time();
                    let ts_offset = rng.gen_range(1..1000);

                    let key = KeyBytes::new(DEFAULT_NS, Bytes::from(key.clone()), ts + ts_offset);
                    let val = ValueBytes::new(DEFAULT_NS, Bytes::from(val.clone()));

                    let _ = tables[table_idx].put(key, val);
                }
            }
        }

        TestData {
            tables,
            _clock: clock,
        }
    }
}

// Benchmark full iteration (simulates full table scan)
fn bench_full_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_iterator_full_scan");

    for size in [10_000, 100_000].iter() {
        for memtables in [2, 4, 8].iter() {
            group.bench_with_input(
                BenchmarkId::new(format!("size_{}_tables_{}", size, memtables), size),
                size,
                |b, &size| {
                    b.iter_batched(
                        || TestData::new(size, *memtables),
                        |test_data| {
                            let iters = test_data
                                .tables
                                .iter()
                                .map(|table| table.scan(Bound::Unbounded, Bound::Unbounded))
                                .collect::<Vec<_>>();

                            let merge_iter = MergeIterator::new(iters);
                            for item in merge_iter {
                                black_box(item);
                            }
                        },
                        BatchSize::SmallInput,
                    )
                },
            );
        }
    }

    group.finish();
}

// Benchmark range scan (simulates typical query pattern)
fn bench_range_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_iterator_range_scan");

    for size in [10_000, 100_000].iter() {
        for memtables in [2, 4, 8].iter() {
            group.bench_with_input(
                BenchmarkId::new(format!("size_{}_tables_{}", size, memtables), size),
                size,
                |b, &size| {
                    b.iter_batched(
                        || TestData::new(size, *memtables),
                        |test_data| {
                            // Scan middle 20% of the data
                            let start_idx = size / 5;
                            let end_idx = (size * 2) / 5;

                            let start_key = format!("key_{:010}", start_idx);
                            let end_key = format!("key_{:010}", end_idx);

                            let iters = test_data
                                .tables
                                .iter()
                                .map(|table| {
                                    table.scan(
                                        Bound::Included(KeyBytes::new(
                                            DEFAULT_NS,
                                            Bytes::from(start_key.clone()),
                                            u128::MAX,
                                        )),
                                        Bound::Excluded(KeyBytes::new(
                                            DEFAULT_NS,
                                            Bytes::from(end_key.clone()),
                                            0,
                                        )),
                                    )
                                })
                                .collect::<Vec<_>>();

                            let merge_iter = MergeIterator::new(iters);
                            for item in merge_iter {
                                black_box(item);
                            }
                        },
                        BatchSize::SmallInput,
                    )
                },
            );
        }
    }

    group.finish();
}

// Benchmark seeking latest versions (simulates get_latest pattern)
fn bench_latest_versions(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_iterator_latest_versions");

    for size in [10_000, 100_000].iter() {
        for memtables in [2, 4, 8].iter() {
            group.bench_with_input(
                BenchmarkId::new(format!("size_{}_tables_{}", size, memtables), size),
                size,
                |b, &size| {
                    b.iter_batched(
                        || TestData::new(size, *memtables),
                        |test_data| {
                            let mut rng = thread_rng();

                            // Get 1000 random latest versions
                            for _ in 0..1000 {
                                let idx = rng.gen_range(0..size);
                                let current_key = format!("key_{:010}", idx);
                                let next_key = format!("key_{:010}", idx + 1);

                                let iters = test_data
                                    .tables
                                    .iter()
                                    .map(|table| {
                                        table.scan(
                                            Bound::Included(KeyBytes::new(
                                                DEFAULT_NS,
                                                Bytes::from(current_key.clone()),
                                                u128::MAX,
                                            )),
                                            Bound::Excluded(KeyBytes::new(
                                                DEFAULT_NS,
                                                Bytes::from(next_key.clone()),
                                                0,
                                            )),
                                        )
                                    })
                                    .collect::<Vec<_>>();

                                let mut merge_iter = MergeIterator::new(iters);
                                if let Some(latest) = merge_iter.next() {
                                    black_box(latest);
                                }
                            }
                        },
                        BatchSize::SmallInput,
                    )
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .measurement_time(std::time::Duration::from_secs(5));
    targets = bench_full_scan, bench_range_scan, bench_latest_versions
);
criterion_main!(benches);
