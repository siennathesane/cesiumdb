use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, BatchSize, Criterion, Throughput, SamplingMode};
use rand::{Rng};
use std::sync::Arc;
use tempfile::tempdir;
use std::time::Duration;

// Assuming these modules are exposed for benchmarking
use cesiumdb::block::BLOCK_SIZE;
use cesiumdb::map::Map;
use cesiumdb::segment::Segment;
use cesiumdb::segment_reader::SegmentReader;
use cesiumdb::segment_writer::SegmentWriter;

// Helper function remains the same
fn create_test_segment() -> (Arc<Segment>, tempfile::TempDir) {
    // Same implementation as before
    let dir = tempdir().expect("failed to create temp dir");

    // Add a random component to filenames to ensure uniqueness
    let random_id: u64 = rand::random();

    // Create key map and writer with more space for benchmarking
    let key_path = dir.path().join(format!("bench-key-segment-{}", random_id));
    let key_map = Arc::new(
        Map::new(key_path, BLOCK_SIZE as u64 * 100).expect("failed to create key map"),
    );
    let key_writer = SegmentWriter::new(key_map.clone()).expect("failed to create key writer");

    // Create value map and writer with more space
    let val_path = dir.path().join(format!("bench-val-segment-{}", random_id));
    let val_map = Arc::new(
        Map::new(val_path, BLOCK_SIZE as u64 * 100).expect("failed to create val map"),
    );
    let val_writer = SegmentWriter::new(val_map.clone()).expect("failed to create val writer");

    // Create segment reader
    let reader = SegmentReader::new(key_map.clone(), val_map.clone())
        .expect("failed to create segment reader");

    // Create segment with a fixed seed for reproducibility
    let seed = 42i64;
    let segment = Arc::new(Segment::new(1, 2, seed, key_writer, val_writer, reader));

    (segment, dir)
}

// Helper function for key-value pairs remains the same
fn generate_kv_pair(key_size: usize, value_size: usize, ns: u64) -> (Vec<u8>, Vec<u8>) {
    // Same implementation as before
    let mut key = Vec::with_capacity(8 + key_size);
    key.extend_from_slice(&ns.to_le_bytes());
    for _ in 0..key_size {
        key.push(rand::random());
    }

    let mut value = Vec::with_capacity(8 + value_size);
    value.extend_from_slice(&ns.to_le_bytes());
    for _ in 0..value_size {
        value.push(rand::random());
    }

    (key, value)
}

fn bench_write_small_kv(c: &mut Criterion) {
    let mut group = c.benchmark_group("segment_write_small_kv");

    // Configure the group to reduce file descriptor usage
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.sampling_mode(SamplingMode::Flat);

    // Only test with one count to reduce files
    let count = 100;
    group.throughput(Throughput::Elements(count as u64));
    group.bench_function(BenchmarkId::from_parameter(count), |b| {
        b.iter_batched_ref(
            || {
                // Setup: Create segment and generate kv pairs
                let (segment, dir) = create_test_segment();
                let kv_pairs = (0..count)
                    .map(|_| generate_kv_pair(16, 64, 0))
                    .collect::<Vec<_>>();

                (segment, kv_pairs, dir)
            },
            |(segment, kv_pairs, _dir)| {
                // Benchmark: write all kv pairs
                let segment_ref = Arc::get_mut(segment).unwrap();
                for (key, value) in kv_pairs.iter() {
                    black_box(segment_ref.write(key, value).unwrap());
                }

                // Force flush to ensure all writes complete
                black_box(segment_ref.flush().unwrap());
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

fn bench_write_different_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("segment_write_different_sizes");

    // Configure the group to reduce file descriptor usage
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.sampling_mode(SamplingMode::Flat);

    // Just one size to reduce file usage
    let key_size = 64;
    let value_size = 256;
    let label = format!("k{}_v{}", key_size, value_size);

    group.bench_function(BenchmarkId::from_parameter(label), |b| {
        b.iter_batched_ref(
            || {
                // Setup: create segment and generate kv pairs
                let (segment, dir) = create_test_segment();
                let kv_pairs = (0..50)
                    .map(|_| generate_kv_pair(key_size, value_size, 0))
                    .collect::<Vec<_>>();

                (segment, kv_pairs, dir)
            },
            |(segment, kv_pairs, _dir)| {
                // Benchmark: write all kv pairs
                let segment_ref = Arc::get_mut(segment).unwrap();
                for (key, value) in kv_pairs.iter() {
                    black_box(segment_ref.write(key, value).unwrap());
                }

                // Force flush to ensure all writes complete
                black_box(segment_ref.flush().unwrap());
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

fn bench_write_large_kv(c: &mut Criterion) {
    let mut group = c.benchmark_group("segment_write_large_kv");

    // Configure the group to reduce file descriptor usage
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.sampling_mode(SamplingMode::Flat);

    // Just one value size to reduce file usage
    let value_size = 4000;
    group.throughput(Throughput::Bytes(value_size as u64));

    group.bench_function(BenchmarkId::from_parameter(value_size), |b| {
        b.iter_batched_ref(
            || {
                // Setup: create segment and generate kv pairs
                let (segment, dir) = create_test_segment();
                let kv_pairs = vec![
                    generate_kv_pair(32, value_size, 0),
                    generate_kv_pair(32, value_size, 0),
                ];

                (segment, kv_pairs, dir)
            },
            |(segment, kv_pairs, _dir)| {
                // Benchmark: write the large kv pairs
                let segment_ref = Arc::get_mut(segment).unwrap();
                for (key, value) in kv_pairs.iter() {
                    black_box(segment_ref.write(key, value).unwrap());
                }

                // Force flush to ensure all writes complete
                black_box(segment_ref.flush().unwrap());
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

fn bench_namespace_switching(c: &mut Criterion) {
    let mut group = c.benchmark_group("segment_namespace_switching");

    // Configure the group to reduce file descriptor usage
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.sampling_mode(SamplingMode::Flat);

    // Just test single namespace case
    let name = "single_ns";
    let ns_pattern = vec![0; 20]; // Reduce number of entries

    group.bench_function(BenchmarkId::from_parameter(name), |b| {
        b.iter_batched_ref(
            || {
                // Setup: create segment and prepare kv pairs with the namespace pattern
                let (segment, dir) = create_test_segment();
                let kv_pairs = ns_pattern.iter()
                    .map(|&ns| generate_kv_pair(16, 64, ns as u64))
                    .collect::<Vec<_>>();

                (segment, kv_pairs, dir)
            },
            |(segment, kv_pairs, _dir)| {
                // Benchmark: write all kv pairs
                let segment_ref = Arc::get_mut(segment).unwrap();
                for (key, value) in kv_pairs.iter() {
                    black_box(segment_ref.write(key, value).unwrap());
                }

                // Force flush to ensure all writes complete
                black_box(segment_ref.flush().unwrap());
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

fn bench_sync(c: &mut Criterion) {
    let mut group = c.benchmark_group("segment_sync");

    // Configure the group to reduce file descriptor usage
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.sampling_mode(SamplingMode::Flat);

    // Just one test case
    let count = 100;

    group.bench_function(BenchmarkId::from_parameter(count), |b| {
        b.iter_batched_ref(
            || {
                // Setup: create segment and generate kv pairs
                let (segment, dir) = create_test_segment();
                let kv_pairs = (0..count)
                    .map(|_| generate_kv_pair(16, 64, 0))
                    .collect::<Vec<_>>();

                (segment, kv_pairs, dir)
            },
            |(segment, kv_pairs, _dir)| {
                let segment_ref = Arc::get_mut(segment).unwrap();

                // Write all kv pairs before benchmarking sync
                for (key, value) in kv_pairs.iter() {
                    segment_ref.write(key, value).unwrap();
                }

                // Benchmark just the sync operation
                black_box(segment_ref.sync().unwrap());
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

criterion_group!(
    segment_benches,
    bench_write_small_kv,
    bench_write_different_sizes,
    bench_write_large_kv,
    bench_namespace_switching,
    bench_sync,
);
criterion_main!(segment_benches);