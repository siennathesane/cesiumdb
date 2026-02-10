use std::{
    ops::Bound,
    sync::Arc,
};

use bytes::Bytes;
use cesiumdb::{
    compact::compact,
    compaction::{
        AdaptiveExecutor,
        CompactionExecutor,
        CompactionJob,
        CompactionQueue,
        ResourceLimits,
    },
    hlc::{
        HLC,
        HybridLogicalClock,
    },
    keypair::{
        DEFAULT_NS,
        KeyBytes,
        ValueBytes,
    },
    memtable::Memtable,
    segment::Segment,
    version::VersionManager,
};
use criterion::{
    BatchSize,
    BenchmarkId,
    Criterion,
    Throughput,
    black_box,
    criterion_group,
    criterion_main,
};
use mimalloc::MiMalloc;
use rand::Rng;
use tempfile::TempDir;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Test data generator for compaction benchmarks
struct CompactionTestData {
    /// Memtables to be flushed/compacted
    memtables: Vec<Arc<Memtable>>,
    /// Output directory for segments
    output_dir: TempDir,
    /// Clock for timestamps
    clock: HybridLogicalClock,
}

impl CompactionTestData {
    /// Creates test data with configurable characteristics
    fn new(
        num_memtables: usize,
        keys_per_table: usize,
        key_overlap_pct: f64,
        tombstone_pct: f64,
    ) -> Self {
        let clock = HybridLogicalClock::new();
        let mut rng = rand::rng();
        let mut memtables = Vec::with_capacity(num_memtables);

        // Create memtables
        for i in 0..num_memtables {
            let table = Memtable::new(i as u64, 1024 * 1024 * 256);
            memtables.push(Arc::new(table));
        }

        // Populate memtables
        for table_idx in 0..num_memtables {
            let start_key = if key_overlap_pct > 0.0 {
                // With overlap, tables share some key ranges
                (table_idx as f64 * keys_per_table as f64 * (1.0 - key_overlap_pct)) as usize
            } else {
                // No overlap - disjoint key ranges
                table_idx * keys_per_table
            };

            for i in 0..keys_per_table {
                let key_idx = start_key + i;
                let key = format!("key_{:010}", key_idx);
                let ts = clock.time();

                let key_bytes = KeyBytes::new(DEFAULT_NS, Bytes::from(key), ts);

                // Generate tombstones based on percentage
                let value_bytes = if rng.random_bool(tombstone_pct) {
                    ValueBytes::new_tombstone(DEFAULT_NS)
                } else {
                    let value = format!("value_{}", key_idx);
                    ValueBytes::new(DEFAULT_NS, Bytes::from(value))
                };

                let _ = memtables[table_idx].put(key_bytes, value_bytes);
            }
        }

        Self {
            memtables,
            output_dir: TempDir::new().unwrap(),
            clock,
        }
    }

    /// Flush all memtables to segments for compaction testing
    fn flush_to_segments(&self) -> Vec<Arc<Segment>> {
        use cesiumdb::compact::flush_memtable;

        let mut segments = Vec::new();
        for (idx, memtable) in self.memtables.iter().enumerate() {
            let path = self
                .output_dir
                .path()
                .join("sstables")
                .join(idx.to_string());
            let segment_id = idx as u64 + 1;

            match flush_memtable(memtable.clone(), path, segment_id) {
                | Ok((segment, _, _)) => segments.push(segment),
                | Err(e) => eprintln!("Flush failed: {:?}", e),
            }
        }
        segments
    }
}

/// Benchmark: Full compaction throughput
///
/// Measures how fast we can compact N segments into one
fn bench_full_compaction_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("compaction_throughput");

    for num_segments in [2, 4, 8, 16] {
        for keys_per_segment in [1000, 10_000] {
            group.throughput(Throughput::Elements(
                (num_segments * keys_per_segment) as u64,
            ));

            group.bench_with_input(
                BenchmarkId::new(format!("segments/{}/keys", num_segments), keys_per_segment),
                &(num_segments, keys_per_segment),
                |b, &(num_seg, keys)| {
                    b.iter_batched(
                        || {
                            // Setup: create segments
                            let test_data = CompactionTestData::new(num_seg, keys, 0.0, 0.0); // No overlap, no tombstones
                            let segments = test_data.flush_to_segments();
                            (test_data.output_dir.path().to_path_buf(), segments)
                        },
                        |(output_path, segments)| {
                            // Benchmark: compact all segments
                            let readers: Vec<_> = segments
                                .iter()
                                .filter_map(|seg| seg.reader().ok())
                                .collect();

                            let iterators: Vec<_> = readers
                                .iter()
                                .map(|reader| {
                                    reader
                                        .scan(Bound::Unbounded, Bound::Unbounded)
                                        .filter_map(|r| r.ok())
                                })
                                .collect();

                            let output_dir = output_path.join("output");
                            let result = compact(iterators, output_dir, 999);
                            black_box(result);
                        },
                        BatchSize::LargeInput,
                    )
                },
            );
        }
    }

    group.finish();
}

/// Benchmark: Compaction with key overlap
///
/// Tests performance impact of overlapping key ranges
fn bench_compaction_with_overlap(c: &mut Criterion) {
    let mut group = c.benchmark_group("compaction_overlap");

    let num_segments = 4;
    let keys_per_segment = 10_000;

    for overlap_pct in [0.0, 0.25, 0.5, 0.75] {
        group.throughput(Throughput::Elements(
            (num_segments * keys_per_segment) as u64,
        ));

        group.bench_with_input(
            BenchmarkId::new("overlap_pct", (overlap_pct * 100.0) as u32),
            &overlap_pct,
            |b, &overlap| {
                b.iter_batched(
                    || {
                        let test_data =
                            CompactionTestData::new(num_segments, keys_per_segment, overlap, 0.0);
                        let segments = test_data.flush_to_segments();
                        (test_data.output_dir.path().to_path_buf(), segments)
                    },
                    |(output_path, segments)| {
                        let readers: Vec<_> = segments
                            .iter()
                            .filter_map(|seg| seg.reader().ok())
                            .collect();

                        let iterators: Vec<_> = readers
                            .iter()
                            .map(|reader| {
                                reader
                                    .scan(Bound::Unbounded, Bound::Unbounded)
                                    .filter_map(|r| r.ok())
                            })
                            .collect();

                        let output_dir = output_path.join("output");
                        let result = compact(iterators, output_dir, 999);
                        black_box(result);
                    },
                    BatchSize::LargeInput,
                )
            },
        );
    }

    group.finish();
}

/// Benchmark: Compaction with tombstones
///
/// Tests performance impact of tombstone handling
fn bench_compaction_with_tombstones(c: &mut Criterion) {
    let mut group = c.benchmark_group("compaction_tombstones");

    let num_segments = 4;
    let keys_per_segment = 10_000;

    for tombstone_pct in [0.0, 0.1, 0.25, 0.5] {
        group.throughput(Throughput::Elements(
            (num_segments * keys_per_segment) as u64,
        ));

        group.bench_with_input(
            BenchmarkId::new("tombstone_pct", (tombstone_pct * 100.0) as u32),
            &tombstone_pct,
            |b, &tombstones| {
                b.iter_batched(
                    || {
                        let test_data = CompactionTestData::new(
                            num_segments,
                            keys_per_segment,
                            0.0,
                            tombstones,
                        );
                        let segments = test_data.flush_to_segments();
                        (test_data.output_dir.path().to_path_buf(), segments)
                    },
                    |(output_path, segments)| {
                        let readers: Vec<_> = segments
                            .iter()
                            .filter_map(|seg| seg.reader().ok())
                            .collect();

                        let iterators: Vec<_> = readers
                            .iter()
                            .map(|reader| {
                                reader
                                    .scan(Bound::Unbounded, Bound::Unbounded)
                                    .filter_map(|r| r.ok())
                            })
                            .collect();

                        let output_dir = output_path.join("output");
                        let result = compact(iterators, output_dir, 999);
                        black_box(result);
                    },
                    BatchSize::LargeInput,
                )
            },
        );
    }

    group.finish();
}

/// Benchmark: Adaptive executor performance
///
/// Tests the overhead and scaling of the adaptive executor
fn bench_adaptive_executor(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_executor");

    // Test job submission and processing overhead
    group.bench_function("job_submission_overhead", |b| {
        b.iter_batched(
            || {
                let temp_dir = TempDir::new().unwrap();
                let version_manager = Arc::new(VersionManager::new(7));
                let executor_impl = Arc::new(CompactionExecutor::new(
                    Arc::clone(&version_manager),
                    temp_dir.path().to_path_buf(),
                    None,
                ));
                let queue = Arc::new(CompactionQueue::new());
                let limits = ResourceLimits {
                    min_workers: 1,
                    max_workers: 2,
                    ..Default::default()
                };

                let executor =
                    AdaptiveExecutor::new(executor_impl, queue.clone(), version_manager, limits);

                (executor, queue)
            },
            |(executor, _queue)| {
                // Create dummy jobs and submit them
                for i in 0..100 {
                    let job = create_dummy_job(i);
                    executor.submit(job);
                }

                // Wait a bit for processing
                std::thread::sleep(std::time::Duration::from_millis(100));

                let stats = executor.usage();
                black_box(stats);

                executor.shutdown();
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

/// Benchmark: Queue operations
///
/// Tests compaction queue enqueue/dequeue performance
fn bench_queue_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("compaction_queue");

    group.bench_function("enqueue_dequeue_1000_jobs", |b| {
        b.iter(|| {
            let queue = CompactionQueue::new();

            // Enqueue 1000 jobs
            for i in 0..1000 {
                let job = create_dummy_job(i);
                queue.enqueue(job);
            }

            // Dequeue all
            let mut count = 0;
            while let Some(job) = queue.dequeue() {
                black_box(job);
                queue.mark_completed();
                count += 1;
            }

            black_box(count);
        })
    });

    group.finish();
}

/// Helper: Create a dummy compaction job for testing
fn create_dummy_job(id: u64) -> CompactionJob {
    use cesiumdb::{
        compaction::job::{
            CompactionInput,
            CompactionJobType,
            CompactionOutput,
        },
        levels::KeyRange,
    };

    let input = CompactionInput {
        level: 0,
        segments: vec![],
        key_range: KeyRange::new(vec![], vec![], 0),
        total_size: 0,
    };

    let output = CompactionOutput::new(1, 64 * 1024 * 1024);

    CompactionJob::new(id, CompactionJobType::Flush, input, None, output)
}

criterion_group!(
    benches,
    bench_full_compaction_throughput,
    bench_compaction_with_overlap,
    bench_compaction_with_tombstones,
    bench_adaptive_executor,
    bench_queue_operations
);
criterion_main!(benches);
