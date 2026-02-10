use std::{
    sync::Arc,
    time,
    time::Instant,
};

use bytes::Bytes;
use cesiumdb::{
    Batch,
    Batch::Put,
    Db,
    DbOptions,
};
use criterion::{
    BenchmarkId,
    Criterion,
    Throughput,
    criterion_group,
    criterion_main,
};
use mimalloc::MiMalloc;
use rand::{
    Rng,
    rngs::ThreadRng,
};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static KB: usize = 1024;
// Test fewer sizes to reduce memory pressure
static PAYLOAD_SIZES: [usize; 4] = [KB, 4 * KB, 16 * KB, 32 * KB];
// Reduced batch sizes significantly to prevent OOM - max 128 items
static BATCH_SIZES: [usize; 4] = [1, 8, 32, 128];

fn db_builder() -> Arc<Db> {
    Db::open(DbOptions::default())
}

fn generate_kvp(db: &Arc<Db>, batch_size: usize, payload_size: usize) {
    let mut rng = rand::rng();
    let mut batches = vec![];
    for batch in 0..batch_size {
        let key = Bytes::from(format!("{}-{}", batch, payload_size).into_bytes());
        let value =
            Bytes::copy_from_slice(rng.random_range(0..payload_size).to_le_bytes().as_ref());
        batches.push(Put(key.clone(), value.clone(), db.time()));
    }

    db.batch(batches.as_slice()).expect("cannot add batches");
}

pub fn db_put(c: &mut Criterion) {
    let db = db_builder();

    let mut put_group = c.benchmark_group("put_value");
    // Limit measurement time to prevent millions of iterations
    put_group.sample_size(10);
    put_group.measurement_time(std::time::Duration::from_secs(2));
    put_group.warm_up_time(std::time::Duration::from_millis(500));

    for size in PAYLOAD_SIZES.iter() {
        put_group.throughput(Throughput::Bytes(*size as u64));
        put_group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let key = format!("key-{}", size);
            let value = vec![0; size];
            b.iter(|| db.put(key.as_ref(), value.as_ref()));
        });
    }
    put_group.finish();
}

pub fn db_put_batch(c: &mut Criterion) {
    let mut put_group = c.benchmark_group("put_batch_value");
    // Aggressive limits to prevent resource exhaustion
    put_group.sample_size(10);
    put_group.measurement_time(std::time::Duration::from_secs(2));
    put_group.warm_up_time(std::time::Duration::from_millis(500));

    for size in PAYLOAD_SIZES.iter() {
        for batch_size in BATCH_SIZES.iter() {
            put_group.throughput(Throughput::Bytes((*size * *batch_size) as u64));
            put_group.bench_with_input(
                BenchmarkId::new(format!("batch/{}/size", batch_size), size),
                size,
                |b, &size| {
                    // Create a fresh database for each sub-benchmark to prevent thread accumulation
                    let db = db_builder();

                    // Create batch with actual batch_size items
                    let mut batch: Vec<Batch<&[u8], &[u8]>> = Vec::with_capacity(*batch_size);
                    let keys: Vec<String> = (0..*batch_size)
                        .map(|i| format!("key-{}-{}", size, i))
                        .collect();
                    let values: Vec<Vec<u8>> = (0..*batch_size).map(|_| vec![0; size]).collect();

                    for i in 0..*batch_size {
                        batch.push(Put(keys[i].as_ref(), values[i].as_ref(), db.time()));
                    }

                    b.iter(|| db.batch(batch.as_slice().as_ref()));

                    // Explicit cleanup
                    drop(db);
                },
            );
        }
    }
    put_group.finish();
}

pub fn db_get(c: &mut Criterion) {
    let db = db_builder();

    // Generate less data to reduce memory pressure
    for size in PAYLOAD_SIZES.iter() {
        for batch_size in BATCH_SIZES.iter() {
            generate_kvp(&db, *batch_size, *size);
        }
    }

    // the largest number of potential keys
    let total_keys = BATCH_SIZES[BATCH_SIZES.len() - 1];
    let rng = rand::rng();

    let random_batch = move |mut x: ThreadRng| x.random_range(0..total_keys);

    let random_size = move |mut x: ThreadRng| PAYLOAD_SIZES[x.random_range(0..PAYLOAD_SIZES.len())];

    let mut get_group = c.benchmark_group("get_value");
    get_group.sample_size(10);
    get_group.measurement_time(std::time::Duration::from_secs(2));
    get_group.warm_up_time(std::time::Duration::from_millis(500));
    get_group.bench_function("random", move |b| {
        b.iter_custom(|iters| {
            let mut durations = Vec::with_capacity(iters as usize);
            for _i in 0..iters {
                let random_key =
                    format!("{}-{}", random_batch(rng.clone()), random_size(rng.clone()));
                let start = Instant::now();
                db.get(random_key.as_ref()).expect("TODO: panic message");
                durations.push(start.elapsed());
            }
            durations.iter().sum::<time::Duration>()
        });
    });
    get_group.finish();
}

criterion_group!(benches, db_get, db_put, db_put_batch);
criterion_main!(benches);
