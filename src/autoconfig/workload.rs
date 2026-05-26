//! Benchmark workloads for the autoconfigurator.

use std::{
    sync::{
        Arc,
        atomic::{
            AtomicBool,
            AtomicU64,
            Ordering,
        },
    },
    thread,
    time::{
        Duration,
        Instant,
    },
};

use rand::{
    Rng,
    rngs::ThreadRng,
};

use crate::Db;

/// Metrics collected from a single benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkMetrics {
    pub ops_per_sec: f64,
    pub write_amplification: f64,
}

/// Runs a sustained write benchmark.
///
/// Spins up `num_workers` threads that each write batches of `batch_size`
/// keys with `value_size`-byte values for `duration_secs`.
pub fn run_write_benchmark(
    db: Arc<Db>,
    num_workers: usize,
    batch_size: usize,
    value_size: usize,
    duration_secs: u64,
) -> BenchmarkMetrics {
    let start = Instant::now();
    let ops_written = Arc::new(AtomicU64::new(0));
    let bytes_written = Arc::new(AtomicU64::new(0));
    let shutdown = Arc::new(AtomicBool::new(false));

    let workers: Vec<_> = (0..num_workers)
        .map(|id| {
            let db = db.clone();
            let ops = ops_written.clone();
            let bytes = bytes_written.clone();
            let shutdown = shutdown.clone();
            thread::spawn(move || {
                let _rng = ThreadRng::default();
                let value = vec![0u8; value_size];
                let worker_key_offset = id * 1_000_000;
                while !shutdown.load(Ordering::Relaxed) {
                    for i in 0..batch_size {
                        let key = format!("key_{:010}", worker_key_offset + i).into_bytes();
                        let _ = db.put(&key, &value);
                        ops.fetch_add(1, Ordering::Relaxed);
                        bytes.fetch_add((key.len() + value.len()) as u64, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    thread::sleep(Duration::from_secs(duration_secs));
    shutdown.store(true, Ordering::Relaxed);
    for w in workers {
        let _ = w.join();
    }

    let elapsed = start.elapsed().as_secs_f64();
    let total_ops = ops_written.load(Ordering::Relaxed);
    let total_bytes = bytes_written.load(Ordering::Relaxed);

    let vstats = db.version_stats();
    let disk_bytes = vstats.total_size;
    let write_amp = if total_bytes > 0 {
        disk_bytes as f64 / total_bytes as f64
    } else {
        1.0
    };

    BenchmarkMetrics {
        ops_per_sec: total_ops as f64 / elapsed,
        write_amplification: write_amp,
    }
}

/// Runs a read-heavy benchmark.
///
/// Assumes the database has already been pre-filled.  Each worker does
/// `reads_per_write` point lookups for every write.
pub fn run_read_benchmark(
    db: Arc<Db>,
    num_workers: usize,
    key_space: usize,
    value_size: usize,
    reads_per_write: usize,
    duration_secs: u64,
) -> BenchmarkMetrics {
    let start = Instant::now();
    let ops = Arc::new(AtomicU64::new(0));
    let shutdown = Arc::new(AtomicBool::new(false));

    let workers: Vec<_> = (0..num_workers)
        .map(|id| {
            let db = db.clone();
            let ops = ops.clone();
            let shutdown = shutdown.clone();
            thread::spawn(move || {
                let mut rng = ThreadRng::default();
                let value = vec![0u8; value_size];
                let worker_key_offset = id * 1_000_000;
                while !shutdown.load(Ordering::Relaxed) {
                    // Do reads
                    for _ in 0..reads_per_write {
                        let key_idx = rng.random_range(0..key_space);
                        let key = format!("key_{:010}", key_idx).into_bytes();
                        let _ = db.get(&key);
                        ops.fetch_add(1, Ordering::Relaxed);
                    }
                    // Occasional write
                    let key = format!("key_{:010}", worker_key_offset + rng.random_range(0..1000))
                        .into_bytes();
                    let _ = db.put(&key, &value);
                    ops.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    thread::sleep(Duration::from_secs(duration_secs));
    shutdown.store(true, Ordering::Relaxed);
    for w in workers {
        let _ = w.join();
    }

    let elapsed = start.elapsed().as_secs_f64();
    let total_ops = ops.load(Ordering::Relaxed);

    BenchmarkMetrics {
        ops_per_sec: total_ops as f64 / elapsed,
        write_amplification: 1.0,
    }
}

/// Runs a mixed read/write benchmark (50/50).
pub fn run_mixed_benchmark(
    db: Arc<Db>,
    num_workers: usize,
    key_space: usize,
    value_size: usize,
    duration_secs: u64,
) -> BenchmarkMetrics {
    run_read_benchmark(db, num_workers, key_space, value_size, 1, duration_secs)
}
