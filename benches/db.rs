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
    criterion_group,
    criterion_main,
    BenchmarkId,
    Criterion,
    Throughput,
};
use rand::{
    prelude::ThreadRng,
    Rng,
};

static KB: usize = 1024;
static PAYLOAD_SIZES: [usize; 6] = [KB, 2 * KB, 4 * KB, 8 * KB, 16 * KB, 32 * KB];
static BATCH_SIZES: [usize; 6] = [1, 2 << 1, 2 << 2, 2 << 4, 2 << 8, 2 << 16];

fn db_builder() -> Arc<Db> {
    Db::open(DbOptions::default())
}

fn generate_kvp(db: &Db, batch_size: usize, payload_size: usize) {
    let mut rng = rand::thread_rng();
    let mut batches = vec![];
    for batch in 0..batch_size {
        let key = Bytes::from(format!("{}-{}", batch, payload_size).into_bytes());
        let value = Bytes::copy_from_slice(rng.gen_range(0..payload_size).to_le_bytes().as_ref());
        batches.push(Put(key.clone(), value.clone(), db.time()));
    }

    db.batch(batches.as_slice());
}

pub fn db_put(c: &mut Criterion) {
    let db = db_builder();

    let mut put_group = c.benchmark_group("put_value");
    for size in PAYLOAD_SIZES.iter() {
        put_group.throughput(Throughput::Bytes(*size as u64));
        put_group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let key = format!("key-{}", size);
            let value = vec![0; size];
            b.iter(|| db.put(key.as_ref(), value.as_ref()));
        });
    }
}

pub fn db_put_batch(c: &mut Criterion) {
    let db = db_builder();

    let mut put_group = c.benchmark_group("put_batch_value");
    for size in PAYLOAD_SIZES.iter() {
        for batch_size in BATCH_SIZES.iter() {
            put_group.throughput(Throughput::Bytes(*size as u64));
            put_group.bench_with_input(
                BenchmarkId::new(format!("batch/{}", batch_size), size),
                size,
                |b, &size| {
                    let mut batch: Vec<Batch<&[u8], &[u8]>> = Vec::with_capacity(*batch_size);
                    let key = format!("key-{}", size);
                    let value = vec![0; size];
                    batch.push(Put(key.as_ref(), value.as_ref(), db.time()));
                    b.iter(|| db.batch(batch.as_slice().as_ref()));
                },
            );
        }
    }
}

pub fn db_get(c: &mut Criterion) {
    let db = db_builder();

    for size in PAYLOAD_SIZES.iter() {
        for batch_size in BATCH_SIZES.iter() {
            generate_kvp(&db, *batch_size, *size);
        }
    }

    // the largest number of potential keys
    let total_keys = BATCH_SIZES[BATCH_SIZES.len() - 1];
    let rng = rand::thread_rng();

    let random_batch = move |mut x: ThreadRng| x.gen_range(0..total_keys);

    let random_size = move |mut x: ThreadRng| PAYLOAD_SIZES[x.gen_range(0..PAYLOAD_SIZES.len())];

    c.bench_function("get_value", move |b| {
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
}

criterion_group!(benches, db_get, db_put, db_put_batch);
criterion_main!(benches);
