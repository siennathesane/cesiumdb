use std::{
    ops::Bound,
    sync::Arc,
    time::Duration,
};

use bytes::Bytes;
use cesiumdb::{
    block::BLOCK_SIZE,
    hlc::{
        HLC,
        HybridLogicalClock,
    },
    keypair::{
        DEFAULT_NS,
        KeyBytes,
        ValueBytes,
    },
    map::Map,
    segment::Segment,
    segment_writer::SegmentWriter,
    utils::Serializer,
};
use criterion::{
    BatchSize,
    BenchmarkId,
    Criterion,
    SamplingMode,
    Throughput,
    black_box,
    criterion_group,
    criterion_main,
};
use rand::Rng;
use tempfile::tempdir;

lazy_static::lazy_static! {
    static ref TEMP_DIR: tempfile::TempDir = tempdir().expect("failed to create global temp dir");
}

fn create_test_segment() -> Arc<Segment> {
    // use global temp dir
    let dir = &*TEMP_DIR;

    // add a random component to filenames to ensure uniqueness
    let random_id: u64 = rand::random();

    // create key map and writer with more space for benchmarking
    let key_path = dir.path().join(format!("bench-key-segment-{}", random_id));
    let key_map =
        Arc::new(Map::new(key_path, BLOCK_SIZE as u64 * 100).expect("failed to create key map"));
    let key_writer = SegmentWriter::new(key_map.clone()).expect("failed to create key writer");

    // create value map and writer with more space
    let val_path = dir.path().join(format!("bench-val-segment-{}", random_id));
    let val_map =
        Arc::new(Map::new(val_path, BLOCK_SIZE as u64 * 100).expect("failed to create val map"));
    let val_writer = SegmentWriter::new(val_map.clone()).expect("failed to create val writer");

    // create segment with a fixed seed for reproducibility
    let seed = 42i64;
    Arc::new(Segment::new(1, 2, seed, key_writer, val_writer))
}

fn create_segment_in_dir() -> Segment {
    // use global temp dir
    let dir = &*TEMP_DIR;

    // add a random component to filenames to ensure uniqueness
    let random_id: u64 = rand::random();

    // create key map and writer with more space for benchmarking
    let key_path = dir.path().join(format!("bench-key-segment-{}", random_id));
    let key_map =
        Arc::new(Map::new(key_path, BLOCK_SIZE as u64 * 100).expect("failed to create key map"));
    let key_writer = SegmentWriter::new(key_map.clone()).expect("failed to create key writer");

    // create value map and writer with more space
    let val_path = dir.path().join(format!("bench-val-segment-{}", random_id));
    let val_map =
        Arc::new(Map::new(val_path, BLOCK_SIZE as u64 * 100).expect("failed to create val map"));
    let val_writer = SegmentWriter::new(val_map.clone()).expect("failed to create val writer");

    // create segment with a fixed seed for reproducibility
    let seed = 42i64;
    Segment::new(1, 2, seed, key_writer, val_writer)
}

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

    // configure the group to reduce file descriptor usage
    group.measurement_time(Duration::from_millis(500));
    group.sampling_mode(SamplingMode::Flat);

    // only test with one count to reduce files
    let count = 100;
    group.throughput(Throughput::Elements(count as u64));
    group.bench_function(BenchmarkId::from_parameter(count), |b| {
        // create segment before benchmark
        let mut segment = Arc::new(create_segment_in_dir());

        // generate kv pairs
        let kv_pairs = (0..count)
            .map(|_| generate_kv_pair(16, 64, 0))
            .collect::<Vec<_>>();

        // the actual benchmark
        b.iter(|| {
            // get a new mutable reference for each iteration
            let segment_ref = Arc::get_mut(&mut segment).unwrap();

            // write all kv pairs
            for (key, value) in &kv_pairs {
                black_box(segment_ref.write(key, value).unwrap());
            }

            // force flush to ensure all writes complete
            black_box(segment_ref.flush().unwrap());
        });
    });

    group.finish();
}

fn bench_write_different_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("segment_write_different_sizes");

    // configure the group to reduce file descriptor usage
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.sampling_mode(SamplingMode::Flat);

    // just one size to reduce file usage
    let key_size = 64;
    let value_size = 256;
    let label = format!("k{}_v{}", key_size, value_size);

    group.bench_function(BenchmarkId::from_parameter(label), |b| {
        b.iter_batched_ref(
            || {
                // setup: create segment and generate kv pairs
                let segment = create_test_segment();
                let kv_pairs = (0..50)
                    .map(|_| generate_kv_pair(key_size, value_size, 0))
                    .collect::<Vec<_>>();

                (segment, kv_pairs)
            },
            |(segment, kv_pairs)| {
                // benchmark: write all kv pairs
                let segment_ref = Arc::get_mut(segment).unwrap();
                for (key, value) in kv_pairs.iter() {
                    black_box(segment_ref.write(key, value).unwrap());
                }

                // force flush to ensure all writes complete
                black_box(segment_ref.flush().unwrap());
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

fn bench_write_large_kv(c: &mut Criterion) {
    let mut group = c.benchmark_group("segment_write_large_kv");

    // configure the group to reduce file descriptor usage
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.sampling_mode(SamplingMode::Flat);

    // just one value size to reduce file usage
    let value_size = 4000;
    group.throughput(Throughput::Bytes(value_size as u64));

    group.bench_function(BenchmarkId::from_parameter(value_size), |b| {
        b.iter_batched_ref(
            || {
                // setup: create segment and generate kv pairs
                let segment = create_test_segment();
                let kv_pairs = vec![
                    generate_kv_pair(32, value_size, 0),
                    generate_kv_pair(32, value_size, 0),
                ];

                (segment, kv_pairs)
            },
            |(segment, kv_pairs)| {
                // benchmark: write the large kv pairs
                let segment_ref = Arc::get_mut(segment).unwrap();
                for (key, value) in kv_pairs.iter() {
                    black_box(segment_ref.write(key, value).unwrap());
                }

                // force flush to ensure all writes complete
                black_box(segment_ref.flush().unwrap());
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

fn bench_sync(c: &mut Criterion) {
    let mut group = c.benchmark_group("segment_sync");

    // configure the group to reduce file descriptor usage
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.sampling_mode(SamplingMode::Flat);

    // just one test case
    let count = 100;

    group.bench_function(BenchmarkId::from_parameter(count), |b| {
        b.iter_batched_ref(
            || {
                // setup: create segment and generate kv pairs
                let segment = create_test_segment();
                let kv_pairs = (0..count)
                    .map(|_| generate_kv_pair(16, 64, 0))
                    .collect::<Vec<_>>();

                (segment, kv_pairs)
            },
            |(segment, kv_pairs)| {
                let segment_ref = Arc::get_mut(segment).unwrap();

                // write all kv pairs before benchmarking sync
                for (key, value) in kv_pairs.iter() {
                    segment_ref.write(key, value).unwrap();
                }

                // benchmark just the sync operation
                black_box(segment_ref.sync().unwrap());
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

fn bench_segment_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("segment_get");

    // configure the group
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.sampling_mode(SamplingMode::Flat);

    // define parameters for the test
    let key_sizes = [16, 64, 256];
    let value_sizes = [16, 64, 256, 1024];

    // create multiple key-value pairs with different sizes
    for &key_size in &key_sizes {
        for &value_size in &value_sizes {
            let label = format!("k{}/v{}", key_size, value_size);

            group.bench_function(BenchmarkId::from_parameter(label), |b| {
                b.iter_batched_ref(
                    || {
                        // setup: create segment and populate with kv pairs
                        let mut segment = create_test_segment();

                        // generate and write a set of key-value pairs
                        let num_pairs = 100;
                        let mut keys = Vec::with_capacity(num_pairs);

                        // get mutable reference and write data
                        {
                            let segment_ref = Arc::get_mut(&mut segment).unwrap();
                            for i in 0..num_pairs {
                                let (key, value) = generate_kv_pair(key_size, value_size, 0);
                                // store the key for later lookups
                                keys.push(key.clone());
                                // write to the segment
                                segment_ref.write(&key, &value).unwrap();
                            }

                            // force flush to ensure all writes complete
                            segment_ref.flush().unwrap();
                        }

                        // return everything needed for the benchmark
                        (segment, keys)
                    },
                    |(segment, keys)| {
                        // create reader at the start of each benchmark iteration
                        let reader = Arc::get_mut(segment).unwrap().new_reader();
                        // benchmark: perform random gets
                        let mut rng = rand::thread_rng();

                        // pick 20 random keys to look up
                        for _ in 0..20 {
                            let idx = rng.gen_range(0..keys.len());
                            // just black_box the result without unwrapping
                            black_box(reader.get(&keys[idx]));
                        }
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }

    group.finish();
}

fn bench_segment_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("segment_scan");

    // Configure the group
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.sampling_mode(SamplingMode::Flat);

    let dataset_sizes = [100, 10_000, 100_000, 1_000_000];
    let scan_ranges = [
        ("small", 0.1),
        ("medium", 0.4),
        ("large", 0.8),
        ("all", 1.0),
    ];

    let clock = HybridLogicalClock::new();

    for &size in &dataset_sizes {
        for (range_name, range_fraction) in &scan_ranges {
            let label = format!("size/{}/range/{}", size, range_name);

            group.bench_function(BenchmarkId::from_parameter(label), |b| {
                b.iter_batched_ref(
                    || {
                        // Setup: create segment with ordered keys
                        let mut segment = create_test_segment();

                        let mut keys: Vec<Vec<u8>> = Vec::with_capacity(size);

                        // Get mutable reference and write data
                        {
                            let segment_ref = Arc::get_mut(&mut segment).unwrap();

                            for i in 0..size {
                                let x = format!("k{:05}", i).as_bytes().to_vec();
                                let key = KeyBytes::new(DEFAULT_NS, Bytes::from(x), clock.time());

                                let y = format!("v{:05}", i).as_bytes().to_vec();
                                let mut value = ValueBytes::new(DEFAULT_NS, Bytes::from(y));

                                let key_bytes = key.serialize_for_memory();

                                // store the raw key for later range bounds
                                keys.push(key_bytes.clone().into());

                                segment_ref
                                    .write(&key_bytes, &value.serialize_for_memory())
                                    .unwrap();
                            }

                            segment_ref.flush().unwrap();
                        }

                        let range_size = (size as f64 * range_fraction) as usize;
                        let start_idx = 5; // Start at a low index to ensure we have data
                        let end_idx = (start_idx + range_size).min(keys.len() - 5); // Ensure we stay in bounds

                        // Get boundary keys
                        let start_key = keys[start_idx].clone();
                        let end_key = keys[end_idx].clone();

                        (segment, start_key, end_key)
                    },
                    |(segment, start_key, end_key)| {
                        let reader = Arc::get_mut(segment).unwrap().new_reader();

                        let lower_bound = Bound::Included(start_key.as_slice());
                        let upper_bound = Bound::Included(end_key.as_slice());

                        let iter = reader.scan(lower_bound, upper_bound);

                        for item in iter {
                            match item {
                                | Ok(entry) => {
                                    black_box(entry);
                                },
                                | Err(_) => {
                                    // Just skip errors in the benchmark
                                    continue;
                                },
                            }
                        }
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }

    group.finish();
}

fn bench_segment_get_cache_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("segment_get_cache");

    // configure the group
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.sampling_mode(SamplingMode::Flat);

    group.bench_function("cold_cache", |b| {
        b.iter_batched_ref(
            || {
                // setup: create segment with many keys
                let mut segment = create_test_segment();

                // generate and write a set of key-value pairs
                let num_pairs = 1000;
                let mut keys = Vec::with_capacity(num_pairs);

                // get mutable reference and write data
                {
                    let segment_ref = Arc::get_mut(&mut segment).unwrap();

                    for i in 0..num_pairs {
                        let (key, value) = generate_kv_pair(32, 64, 0);
                        keys.push(key.clone());
                        segment_ref.write(&key, &value).unwrap();
                    }

                    segment_ref.flush().unwrap();
                }

                (segment, keys)
            },
            |(segment, keys)| {
                // create a fresh reader for each iteration
                let mut reader = Arc::get_mut(segment).unwrap().new_reader();
                // benchmark: perform gets with cold cache
                let mut rng = rand::thread_rng();

                // clear cache before each key lookup
                reader.refresh();

                // look up 10 random keys
                for _ in 0..10 {
                    let idx = rng.gen_range(0..keys.len());
                    // don't unwrap the result
                    black_box(reader.get(&keys[idx]));
                }
            },
            BatchSize::LargeInput,
        );
    });

    group.bench_function("warm_cache", |b| {
        b.iter_batched_ref(
            || {
                // same setup as above
                let mut segment = create_test_segment();

                let num_pairs = 1000;
                let mut keys = Vec::with_capacity(num_pairs);

                // get mutable reference and write data
                {
                    let segment_ref = Arc::get_mut(&mut segment).unwrap();

                    for i in 0..num_pairs {
                        let (key, value) = generate_kv_pair(32, 64, 0);
                        keys.push(key.clone());
                        segment_ref.write(&key, &value).unwrap();
                    }

                    segment_ref.flush().unwrap();
                }

                // pre-warm some keys in the benchmark setup for consistency
                {
                    let segment_ref = Arc::get_mut(&mut segment).unwrap();
                    let reader = segment_ref.new_reader();

                    // pre-warm cache by accessing some keys
                    for i in 0..5 {
                        reader.get(&keys[i]).unwrap();
                    }
                }

                (segment, keys)
            },
            |(segment, keys)| {
                // create a fresh reader for each iteration (but cache remains warmed)
                let reader = Arc::get_mut(segment).unwrap().new_reader();
                // benchmark: perform gets with warm cache
                let mut rng = rand::thread_rng();

                // look up 10 random keys (without clearing cache)
                for _ in 0..10 {
                    let idx = rng.gen_range(0..5); // only look up the pre-warmed keys
                    // don't unwrap the result
                    black_box(reader.get(&keys[idx]));
                }
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

criterion_group!(
    name = segment_benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_millis(500));
    targets = bench_write_small_kv,
    bench_write_different_sizes,
    bench_write_large_kv,
    bench_sync,
    bench_segment_get,
    bench_segment_scan,
    bench_segment_get_cache_comparison,
);
criterion_main!(segment_benches);
