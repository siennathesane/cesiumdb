use std::fs::OpenOptions;
use std::sync::Arc;
use cesiumdb::fs::{
    CompactionConfig,
    Fs,
};
use criterion::{
    black_box,
    criterion_group,
    criterion_main,
    BatchSize,
    BenchmarkId,
    Criterion,
    Throughput,
};
use memmap2::MmapMut;
use rand::{
    thread_rng,
    Rng,
};
use tempfile::tempdir;

const BENCH_FILE_SIZE: u64 = 1024 * 1024 * 256; // 256MB for benchmarking
const FRAG_BENCH_SIZE: u64 = 1024 * 1024 * 1024; // 1GB for fragmentation tests

fn setup_benchmark_fs() -> (Arc<Fs>, tempfile::TempDir) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bench.db");

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&path)
        .unwrap();

    file.set_len(BENCH_FILE_SIZE).unwrap();
    let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
    let fs = Fs::init(mmap).unwrap();

    (fs, dir)
}

fn setup_fragmented_fs(fragmentation_level: u32) -> (Arc<Fs>, tempfile::TempDir) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bench.db");

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&path)
        .unwrap();

    file.set_len(FRAG_BENCH_SIZE).unwrap();
    let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
    let fs = Fs::init(mmap).unwrap();

    let mut rng = thread_rng();
    let mut ids = Vec::new();

    // Create franges with varying sizes to simulate real-world fragmentation
    for _ in 0..fragmentation_level {
        // Mix of small (4KB-64KB) and large (1MB-4MB) franges
        let size = if rng.gen_bool(0.7) {
            rng.gen_range(4096..65536)
        } else {
            rng.gen_range(1024 * 1024..4 * 1024 * 1024)
        };

        if let Ok(id) = fs.create_frange(size) {
            // Write random amounts of data to simulate partially filled franges
            let handle = fs.open_frange(id).unwrap();
            let write_size = (size as f64 * rng.gen_range(0.3..0.9)) as u64;
            let data = vec![rng.gen::<u8>(); 4096];

            for offset in (0..write_size).step_by(4096) {
                let remaining = write_size - offset;
                let chunk_size = std::cmp::min(remaining, 4096) as usize;
                handle.write_at(offset, &data[..chunk_size]).unwrap();
            }

            fs.close_frange(handle).unwrap();
            ids.push(id);
        }
    }

    // Delete a portion of franges randomly to create fragmentation
    let delete_count = (ids.len() as f64 * rng.gen_range(0.3..0.5)) as usize;
    for _ in 0..delete_count {
        if let Some(idx) = ids.len().checked_sub(1) {
            let remove_idx = rng.gen_range(0..=idx);
            let id = ids.swap_remove(remove_idx);
            fs.delete_frange(id).unwrap();
        }
    }

    (fs, dir)
}

fn benchmark_frange_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("frange_creation");

    // Test different frange sizes
    for size in [1024, 1024 * 1024, 1024 * 1024 * 10].iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_batched(
                || setup_benchmark_fs(),
                |(fs, _dir)| {
                    black_box(fs.create_frange(size as u64)).unwrap();
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn benchmark_sequential_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_write");

    // Test different write sizes
    for size in [1024, 4096, 16384, 65536].iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let (fs, _dir) = setup_benchmark_fs();
            let id = fs.create_frange(size as u64 * 100).unwrap(); // Create frange with room for multiple writes
            let handle = fs.open_frange(id).unwrap();
            let data = vec![0u8; size as usize];

            b.iter(|| {
                black_box(handle.write_at(0, &data)).unwrap();
            });
        });
    }

    group.finish();
}

fn benchmark_random_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_write");
    let mut rng = thread_rng();

    // Test different write sizes with random positions
    for size in [1024, 4096, 16384, 65536].iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let (fs, _dir) = setup_benchmark_fs();
            let frange_size = 1024 * 1024 * 10; // 10MB frange
            let id = fs.create_frange(frange_size).unwrap();
            let handle = fs.open_frange(id).unwrap();
            let data = vec![0u8; size as usize];

            b.iter(|| {
                let pos = rng.gen_range(0..frange_size - size as u64);
                black_box(handle.write_at(pos, &data)).unwrap();
            });
        });
    }

    group.finish();
}

fn benchmark_sequential_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_read");

    // Test different read sizes
    for size in [1024, 4096, 16384, 65536].iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let (fs, _dir) = setup_benchmark_fs();
            let id = fs.create_frange(size as u64 * 100).unwrap();
            let handle = fs.open_frange(id).unwrap();

            // Pre-write data
            let write_data = vec![0u8; size as usize];
            handle.write_at(0, &write_data).unwrap();

            let mut read_buf = vec![0u8; size as usize];
            b.iter(|| {
                black_box(handle.read_at(0, &mut read_buf)).unwrap();
            });
        });
    }

    group.finish();
}

fn benchmark_random_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_read");
    let mut rng = thread_rng();

    // Test different read sizes with random positions
    for size in [1024, 4096, 16384, 65536].iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let (fs, _dir) = setup_benchmark_fs();
            let frange_size = 1024 * 1024 * 10; // 10MB frange
            let id = fs.create_frange(frange_size).unwrap();
            let handle = fs.open_frange(id).unwrap();

            // Pre-write data
            let write_data = vec![0u8; frange_size as usize];
            handle.write_at(0, &write_data).unwrap();

            let mut read_buf = vec![0u8; size as usize];
            b.iter(|| {
                let pos = rng.gen_range(0..frange_size - size as u64);
                black_box(handle.read_at(pos, &mut read_buf)).unwrap();
            });
        });
    }

    group.finish();
}

fn benchmark_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_workload");
    let mut rng = thread_rng();

    // Test mixed read/write workloads
    for ops in [100, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(ops), ops, |b, &ops| {
            let (fs, _dir) = setup_benchmark_fs();
            let frange_size = 1024 * 1024; // 1MB frange
            let id = fs.create_frange(frange_size).unwrap();
            let handle = fs.open_frange(id).unwrap();

            // Pre-write some data
            let write_data = vec![0u8; 4096];
            handle.write_at(0, &write_data).unwrap();

            let mut read_buf = vec![0u8; 4096];
            b.iter(|| {
                for _ in 0..ops {
                    let pos = rng.gen_range(0..frange_size - 4096);
                    match rng.gen_range(0..2) {
                        | 0 => black_box(handle.read_at(pos, &mut read_buf)).unwrap(),
                        | _ => black_box(handle.write_at(pos, &write_data)).unwrap(),
                    }
                }
            });
        });
    }

    group.finish();
}

fn benchmark_fragmentation(c: &mut Criterion) {
    let mut group = c.benchmark_group("fragmentation");

    // Increase initial file size for fragmentation tests
    const FRAG_BENCH_SIZE: u64 = 1024 * 1024 * 512; // 512MB for fragmentation tests

    // Test performance under different fragmentation levels
    for frag_ops in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(frag_ops),
            frag_ops,
            |b, &frag_ops| {
                b.iter_batched(
                    || {
                        // Setup fragmented filesystem with larger size
                        let dir = tempdir().unwrap();
                        let path = dir.path().join("bench.db");

                        let file = OpenOptions::new()
                            .read(true)
                            .write(true)
                            .create(true)
                            .open(&path)
                            .unwrap();

                        file.set_len(FRAG_BENCH_SIZE).unwrap();
                        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
                        let fs = Fs::init(mmap).unwrap();

                        // Create alternating small and large franges
                        let mut ids = Vec::new();
                        for _ in 0..frag_ops {
                            // Create larger franges to ensure we have space
                            if let Ok(id) = fs.create_frange(1024 * 64) {
                                // 64KB small frange
                                ids.push(id);
                            }
                            if let Ok(id) = fs.create_frange(1024 * 1024) {
                                // 1MB large frange
                                ids.push(id);
                            }
                        }

                        // Delete every other frange to create fragmentation
                        for i in (0..ids.len()).step_by(2) {
                            fs.delete_frange(ids[i]).unwrap();
                        }

                        (fs, dir)
                    },
                    |(fs, _dir)| {
                        // Benchmark operation under fragmentation
                        black_box(fs.create_frange(1024 * 512)).unwrap()
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn benchmark_concurrent_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_access");

    for threads in [2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(threads),
            threads,
            |b, &threads| {
                use std::{
                    sync::Arc,
                    thread,
                };

                b.iter_batched(
                    setup_benchmark_fs,
                    |(fs, _dir)| {
                        let fs = Arc::new(fs);
                        let mut handles = Vec::new();

                        for _ in 0..threads {
                            let fs_clone = fs.clone();
                            handles.push(thread::spawn(move || {
                                let id = fs_clone.create_frange(1024 * 1024).unwrap();
                                let handle = fs_clone.open_frange(id).unwrap();
                                let data = vec![0u8; 4096];
                                for i in 0..10 {
                                    handle.write_at(i * 4096, &data).unwrap();
                                }
                                fs_clone.close_frange(handle).unwrap();
                                fs_clone.delete_frange(id).unwrap();
                            }));
                        }

                        for handle in handles {
                            handle.join().unwrap();
                        }
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn benchmark_fragmentation_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("fragmentation_impact");
    group.sample_size(10); // Reduced sample size due to expensive setup

    // Test different fragmentation levels
    for level in [50, 200, 500].iter() {
        group.throughput(Throughput::Bytes(*level as u64 * 1024 * 1024));

        group.bench_with_input(
            BenchmarkId::new("allocation_time", level),
            level,
            |b, &level| {
                b.iter_batched(
                    || setup_fragmented_fs(level),
                    |(fs, _dir)| {
                        // Benchmark allocation time under fragmentation
                        fs.create_frange(1024 * 1024).unwrap() // 1MB allocation
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        // Measure operation latency under fragmentation
        group.bench_with_input(
            BenchmarkId::new("operation_latency", level),
            level,
            |b, &level| {
                b.iter_batched(
                    || {
                        let (fs, dir) = setup_fragmented_fs(level);
                        let id = fs.create_frange(1024 * 1024).unwrap();
                        (fs, dir, id)
                    },
                    |(fs, _dir, id)| {
                        let  handle = fs.open_frange(id).unwrap();
                        let data = vec![0u8; 4096];
                        // Perform mixed operations
                        for i in 0..10 {
                            handle.write_at(i * 4096, &data).unwrap();
                        }
                        fs.close_frange(handle).unwrap();
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

fn benchmark_compaction(c: &mut Criterion) {
    let mut group = c.benchmark_group("compaction");
    group.sample_size(10); // Reduced sample size due to expensive setup

    // Test different fragmentation levels and compaction configurations
    for level in [100, 300, 600].iter() {
        group.throughput(Throughput::Bytes(*level as u64 * 1024 * 1024));

        // Test different compaction configurations
        let configs = [
            (
                "aggressive",
                CompactionConfig {
                    fragmentation_threshold: 0.1,
                    min_fragment_size: 4096,
                    max_compact_batch: 50,
                    block_buffer_size: 4096,
                },
            ),
            (
                "balanced",
                CompactionConfig {
                    fragmentation_threshold: 0.3,
                    min_fragment_size: 16384,
                    max_compact_batch: 20,
                    block_buffer_size: 4096,
                },
            ),
            (
                "conservative",
                CompactionConfig {
                    fragmentation_threshold: 0.5,
                    min_fragment_size: 65536,
                    max_compact_batch: 10,
                    block_buffer_size: 4096,
                },
            ),
            (
                "4k_block",
                CompactionConfig {
                    fragmentation_threshold: 0.3,
                    min_fragment_size: 16384,
                    max_compact_batch: 20,
                    block_buffer_size: 4096,
                },
            ),
            (
                "8k_block",
                CompactionConfig {
                    fragmentation_threshold: 0.3,
                    min_fragment_size: 16384,
                    max_compact_batch: 20,
                    block_buffer_size: 8192,
                },
            ),
            (
                "16k_block",
                CompactionConfig {
                    fragmentation_threshold: 0.3,
                    min_fragment_size: 16384,
                    max_compact_batch: 20,
                    block_buffer_size: 16384,
                },
            ),
            (
                "1m_block",
                CompactionConfig {
                    fragmentation_threshold: 0.3,
                    min_fragment_size: 16384,
                    max_compact_batch: 20,
                    block_buffer_size: 1024 * 1024,
                },
            ),
        ];

        for (config_name, config) in configs.iter() {
            group.bench_with_input(
                BenchmarkId::new(format!("compaction_{}_{}", config_name, level), level),
                &(level, config),
                |b, &(level, config)| {
                    b.iter_batched(
                        || setup_fragmented_fs(*level),
                        |(fs, _dir)| {
                            fs.maybe_compact_with_config(config).unwrap();
                        },
                        BatchSize::SmallInput,
                    )
                },
            );
        }

        // Benchmark post-compaction performance
        group.bench_with_input(
            BenchmarkId::new(format!("post_compaction_{}", level), level),
            level,
            |b, &level| {
                b.iter_batched(
                    || {
                        let (fs, dir) = setup_fragmented_fs(level);
                        // Perform compaction before benchmarking
                        fs.maybe_compact_with_config(&CompactionConfig::default())
                            .unwrap();
                        (fs, dir)
                    },
                    |(fs, _dir)| {
                        // Measure allocation and operation performance after compaction
                        let id = fs.create_frange(1024 * 1024).unwrap();
                        let handle = fs.open_frange(id).unwrap();
                        let data = vec![0u8; 4096];
                        for i in 0..10 {
                            handle.write_at(i * 4096, &data).unwrap();
                        }
                        fs.close_frange(handle).unwrap();
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_frange_creation,
    benchmark_sequential_write,
    benchmark_random_write,
    benchmark_sequential_read,
    benchmark_random_read,
    benchmark_mixed_workload,
    benchmark_fragmentation,
    benchmark_fragmentation_impact,
    benchmark_compaction,
    benchmark_concurrent_access,
);
criterion_main!(benches);
