use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, BatchSize, Throughput};
use rand::{Rng, thread_rng};
use std::sync::Arc;
use tempfile::tempdir;

// assuming these modules are exposed for benchmarking
use cesiumdb::segment::Segment;
use cesiumdb::map::Map;
use cesiumdb::segment_reader::SegmentReader;
use cesiumdb::segment_writer::SegmentWriter;
use cesiumdb::block::BLOCK_SIZE;

// helper function to create a test segment
fn create_test_segment() -> (Arc<Segment>, tempfile::TempDir) {
    let dir = tempdir().expect("failed to create temp dir");

    // add a random component to filenames to ensure uniqueness
    let random_id: u64 = rand::random();

    // create key map and writer with more space for benchmarking
    let key_path = dir.path().join(format!("bench-key-segment-{}", random_id));
    let key_map = Arc::new(Map::new(key_path, BLOCK_SIZE as u64 * 100).expect("failed to create key map"));
    let key_writer = SegmentWriter::new(key_map.clone()).expect("failed to create key writer");

    // create value map and writer with more space
    let val_path = dir.path().join(format!("bench-val-segment-{}", random_id));
    let val_map = Arc::new(Map::new(val_path, BLOCK_SIZE as u64 * 100).expect("failed to create val map"));
    let val_writer = SegmentWriter::new(val_map.clone()).expect("failed to create val writer");

    // create segment reader
    let reader = SegmentReader::new(key_map.clone(), val_map.clone())
        .expect("failed to create segment reader");

    // create segment with a fixed seed for reproducibility
    let seed = 42i64;
    let segment = Arc::new(Segment::new(1, 2, seed, key_writer, val_writer, reader));

    (segment, dir)
}

// helper function to generate keys and values of different sizes
fn generate_kv_pair(key_size: usize, value_size: usize, ns: u64) -> (Vec<u8>, Vec<u8>) {
    // create key with namespace at beginning
    let mut key = Vec::with_capacity(8 + key_size);
    key.extend_from_slice(&ns.to_le_bytes());
    for _ in 0..key_size {
        key.push(rand::random());
    }

    // create value with namespace at beginning
    let mut value = Vec::with_capacity(8 + value_size);
    value.extend_from_slice(&ns.to_le_bytes());
    for _ in 0..value_size {
        value.push(rand::random());
    }

    (key, value)
}

// benchmark writing small key-value pairs (fits easily in a block)
fn bench_write_small_kv(c: &mut Criterion) {
    let mut group = c.benchmark_group("segment_write_small_kv");

    // different numbers of key-value pairs to write
    for count in [10, 100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, &count| {
            b.iter_batched(
                || {
                    // setup: create segment and generate kv pairs
                    let (segment, _dir) = create_test_segment();

                    let mut kv_pairs = Vec::with_capacity(count);
                    for _ in 0..count {
                        kv_pairs.push(generate_kv_pair(16, 64, 0));
                    }

                    (segment, kv_pairs)
                },
                |(mut segment, kv_pairs)| {
                    // benchmark: write all kv pairs
                    let segment_ref = Arc::get_mut(&mut segment).unwrap();
                    for (key, value) in kv_pairs {
                        black_box(segment_ref.write(&key, &value).unwrap());
                    }

                    // force flush to ensure all writes complete
                    black_box(segment_ref.flush().unwrap());
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// benchmark writing different key-value sizes
fn bench_write_different_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("segment_write_different_sizes");

    // different combinations of key and value sizes
    let kv_sizes = [
        (16, 64),      // small key, small value
        (64, 256),     // medium key, medium value
        (256, 1024),   // large key, large value
        (16, 1024),    // small key, large value
        (256, 64),     // large key, small value
    ];

    for &(key_size, value_size) in kv_sizes.iter() {
        let label = format!("k{}_v{}", key_size, value_size);
        group.bench_with_input(BenchmarkId::from_parameter(label), &(key_size, value_size), |b, &(key_size, value_size)| {
            b.iter_batched(
                || {
                    // setup: create segment and generate 100 kv pairs
                    let (segment, _dir) = create_test_segment();

                    let mut kv_pairs = Vec::with_capacity(100);
                    for _ in 0..100 {
                        kv_pairs.push(generate_kv_pair(key_size, value_size, 0));
                    }

                    (segment, kv_pairs)
                },
                |(mut segment, kv_pairs)| {
                    // benchmark: write all kv pairs
                    let segment_ref = Arc::get_mut(&mut segment).unwrap();
                    for (key, value) in kv_pairs {
                        black_box(segment_ref.write(&key, &value).unwrap());
                    }

                    // force flush to ensure all writes complete
                    black_box(segment_ref.flush().unwrap());
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// benchmark writing large key-value pairs (that span multiple blocks)
fn bench_write_large_kv(c: &mut Criterion) {
    let mut group = c.benchmark_group("segment_write_large_kv");

    // different large value sizes
    for value_size in [4000, 8000, 16000, 32000].iter() {
        // set throughput to bytes written for better comparison
        group.throughput(Throughput::Bytes(*value_size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(value_size), value_size, |b, &value_size| {
            b.iter_batched(
                || {
                    // setup: create segment and generate a few large kv pairs
                    let (segment, _dir) = create_test_segment();

                    let kv_pairs = vec![
                        generate_kv_pair(32, value_size, 0),
                        generate_kv_pair(32, value_size, 0),
                        generate_kv_pair(32, value_size, 0),
                    ];

                    (segment, kv_pairs)
                },
                |(mut segment, kv_pairs)| {
                    // benchmark: write the large kv pairs
                    let segment_ref = Arc::get_mut(&mut segment).unwrap();
                    for (key, value) in kv_pairs {
                        black_box(segment_ref.write(&key, &value).unwrap());
                    }

                    // force flush to ensure all writes complete
                    black_box(segment_ref.flush().unwrap());
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// benchmark switching between namespaces
fn bench_namespace_switching(c: &mut Criterion) {
    let mut group = c.benchmark_group("segment_namespace_switching");

    // different patterns of namespace switching
    // each test writes the same number of kv pairs but with different ns patterns
    let test_cases = [
        ("single_ns", vec![0; 100]),  // single namespace
        ("alternating", (0..100).map(|i| i % 2).collect::<Vec<_>>()), // alternating between 2 namespaces
        ("sequential", (0..100).map(|i| i / 10).collect::<Vec<_>>()),  // 10 sequential writes per namespace
        ("random", (0..100).map(|_| thread_rng().gen_range(0..10)).collect::<Vec<_>>()),  // random namespaces
    ];

    for (name, ns_pattern) in test_cases.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(name), ns_pattern, |b, pattern| {
            b.iter_batched(
                || {
                    // setup: create segment and prepare kv pairs with the namespace pattern
                    let (segment, _dir) = create_test_segment();

                    let mut kv_pairs = Vec::with_capacity(pattern.len());
                    for &ns in pattern {
                        kv_pairs.push(generate_kv_pair(16, 64, ns as u64));
                    }

                    (segment, kv_pairs)
                },
                |(mut segment, kv_pairs)| {
                    // benchmark: write all kv pairs (which will cause namespace switches)
                    let segment_ref = Arc::get_mut(&mut segment).unwrap();
                    for (key, value) in kv_pairs {
                        black_box(segment_ref.write(&key, &value).unwrap());
                    }

                    // force flush to ensure all writes complete
                    black_box(segment_ref.flush().unwrap());
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// benchmark sync overhead
fn bench_sync(c: &mut Criterion) {
    let mut group = c.benchmark_group("segment_sync");

    // benchmark with different numbers of writes before sync
    for count in [10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, &count| {
            b.iter_batched(
                || {
                    // setup: create segment and generate kv pairs
                    let (segment, _dir) = create_test_segment();
                    let kv_pairs = (0..count)
                        .map(|_| generate_kv_pair(16, 64, 0))
                        .collect::<Vec<_>>();

                    // clone to move segment into closure
                    (segment, kv_pairs)
                },
                |(mut segment, kv_pairs)| {
                    let segment_ref = Arc::get_mut(&mut segment).unwrap();

                    // write all kv pairs before benchmarking sync
                    for (key, value) in kv_pairs {
                        segment_ref.write(&key, &value).unwrap();
                    }

                    // benchmark just the sync operation
                    black_box(segment_ref.sync().unwrap());
                },
                BatchSize::SmallInput,
            );
        });
    }

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