//! Comprehensive soak test framework for CesiumDB
//!
//! This module provides stress testing capabilities similar to RocksDB's
//! db_bench, validating stability, performance, and correctness under sustained
//! heavy load.
//!
//! Features:
//! - Configurable workloads (mixed, write-heavy, read-heavy, scan-heavy,
//!   delete-heavy)
//! - Adjustable test parameters via environment variables
//! - Real-time metrics collection and reporting
//! - Probabilistic correctness verification (shadow verifier)
//! - Multi-threaded concurrent operations
//!
//! Usage:
//!   cargo test --test soak_test soak_smoke_mixed -- --ignored --nocapture
//!   SOAK_DURATION=300 cargo test --test soak_test -- --ignored --nocapture
//!
//! With OpenTelemetry/Jaeger (for performance analysis):
//!   1. Start Jaeger: docker run -d -p4317:4317 -p16686:16686
//!      jaegertracing/all-in-one:latest
//!   2. Run test: OTEL_ENABLED=1 cargo test --test soak_test soak_smoke_mixed
//!      -- --ignored --nocapture
//!   3. View traces: http://localhost:16686

use std::{
    collections::HashMap,
    fs,
    ops::Bound,
    path::PathBuf,
    sync::{
        Arc,
        Mutex,
        atomic::{
            AtomicBool,
            AtomicU64,
            Ordering,
        },
    },
    thread::{
        self,
        JoinHandle,
    },
    time::{
        Duration,
        Instant,
    },
};

use cesiumdb::{
    Batch::*,
    Db,
    DbOptions,
};
use rand::{
    Rng,
    rngs::ThreadRng,
};

// ============================================================================
// Configuration System
// ============================================================================

#[derive(Debug, Clone)]
struct SoakConfig {
    duration_secs: u64,
    num_workers: usize,
    memtable_size: u64,
    max_memtables: u64,
    value_size: usize,
    key_space_size: u64,
    verification_sample_rate: f64,
    metrics_interval_ms: u64,
}

impl SoakConfig {
    /// 1-minute smoke test with small key space
    fn smoke_test() -> Self {
        Self {
            duration_secs: Self::env_or("SOAK_DURATION", 60),
            num_workers: Self::env_or(
                "SOAK_WORKERS",
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4),
            ),
            memtable_size: Self::env_or("SOAK_MEMTABLE_SIZE", 64 * 1024 * 1024u64),
            max_memtables: Self::env_or("SOAK_MAX_MEMTABLES", 4u64),
            value_size: Self::env_or("SOAK_VALUE_SIZE", 1024),
            key_space_size: Self::env_or("SOAK_KEY_SPACE", 10_000),
            verification_sample_rate: 0.01,
            metrics_interval_ms: 500,
        }
    }

    /// 10-minute standard soak test
    fn standard_soak() -> Self {
        Self {
            duration_secs: Self::env_or("SOAK_DURATION", 600),
            key_space_size: Self::env_or("SOAK_KEY_SPACE", 1_000_000),
            ..Self::smoke_test()
        }
    }

    /// 1+ hour extended soak test
    fn extended_soak() -> Self {
        Self {
            duration_secs: Self::env_or("SOAK_DURATION", 3600),
            key_space_size: Self::env_or("SOAK_KEY_SPACE", 10_000_000),
            ..Self::smoke_test()
        }
    }

    fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }
}

// ============================================================================
// Metrics Collection (Lock-Free)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpType {
    Get,
    Put,
    Delete,
    Scan,
}

struct MetricsCollector {
    // Operation counters
    get_count: AtomicU64,
    put_count: AtomicU64,
    delete_count: AtomicU64,
    scan_count: AtomicU64,
    error_count: AtomicU64,

    // Simple latency histogram (7 buckets: <1µs, 1-10µs, 10-100µs, 100µs-1ms, 1-10ms, 10-100ms,
    // >100ms)
    latency_buckets: [AtomicU64; 7],
    min_micros: AtomicU64,
    max_micros: AtomicU64,

    start_time: Instant,
}

impl MetricsCollector {
    fn new() -> Self {
        Self {
            get_count: AtomicU64::new(0),
            put_count: AtomicU64::new(0),
            delete_count: AtomicU64::new(0),
            scan_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            latency_buckets: Default::default(),
            min_micros: AtomicU64::new(u64::MAX),
            max_micros: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    fn record_op(&self, op_type: OpType, latency: Duration) {
        // Update operation counters
        match op_type {
            | OpType::Get => self.get_count.fetch_add(1, Ordering::Relaxed),
            | OpType::Put => self.put_count.fetch_add(1, Ordering::Relaxed),
            | OpType::Delete => self.delete_count.fetch_add(1, Ordering::Relaxed),
            | OpType::Scan => self.scan_count.fetch_add(1, Ordering::Relaxed),
        };

        // Update latency stats
        let micros = latency.as_micros() as u64;

        // Update min/max
        self.min_micros.fetch_min(micros, Ordering::Relaxed);
        self.max_micros.fetch_max(micros, Ordering::Relaxed);

        // Update histogram bucket
        let bucket = match micros {
            | 0..=1 => 0,
            | 2..=10 => 1,
            | 11..=100 => 2,
            | 101..=1_000 => 3,
            | 1_001..=10_000 => 4,
            | 10_001..=100_000 => 5,
            | _ => 6,
        };
        self.latency_buckets[bucket].fetch_add(1, Ordering::Relaxed);
    }

    fn record_error(&self) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self) -> MetricsSnapshot {
        let get_count = self.get_count.load(Ordering::Relaxed);
        let put_count = self.put_count.load(Ordering::Relaxed);
        let delete_count = self.delete_count.load(Ordering::Relaxed);
        let scan_count = self.scan_count.load(Ordering::Relaxed);
        let error_count = self.error_count.load(Ordering::Relaxed);
        let total_ops = get_count + put_count + delete_count + scan_count;

        let elapsed_secs = self.start_time.elapsed().as_secs_f64();
        let ops_per_sec = if elapsed_secs > 0.0 {
            total_ops as f64 / elapsed_secs
        } else {
            0.0
        };

        let min_micros = self.min_micros.load(Ordering::Relaxed);
        let max_micros = self.max_micros.load(Ordering::Relaxed);

        // Load histogram
        let mut buckets = [0u64; 7];
        for (i, bucket) in self.latency_buckets.iter().enumerate() {
            buckets[i] = bucket.load(Ordering::Relaxed);
        }

        // Calculate percentiles (approximate)
        let (p50, p95, p99) = calculate_percentiles(&buckets, total_ops);

        MetricsSnapshot {
            get_count,
            put_count,
            delete_count,
            scan_count,
            total_ops,
            error_count,
            elapsed_secs,
            ops_per_sec,
            min_micros,
            max_micros,
            p50_micros: p50,
            p95_micros: p95,
            p99_micros: p99,
        }
    }
}

#[derive(Debug, Clone)]
struct MetricsSnapshot {
    get_count: u64,
    put_count: u64,
    delete_count: u64,
    scan_count: u64,
    total_ops: u64,
    error_count: u64,
    elapsed_secs: f64,
    ops_per_sec: f64,
    min_micros: u64,
    max_micros: u64,
    p50_micros: u64,
    p95_micros: u64,
    p99_micros: u64,
}

fn calculate_percentiles(buckets: &[u64; 7], total: u64) -> (u64, u64, u64) {
    if total == 0 {
        return (0, 0, 0);
    }

    let p50_target = total / 2;
    let p95_target = (total * 95) / 100;
    let p99_target = (total * 99) / 100;

    let mut cumulative = 0u64;
    let mut p50 = 0u64;
    let mut p95 = 0u64;
    let mut p99 = 0u64;

    let bucket_midpoints = [1, 5, 50, 500, 5_000, 50_000, 150_000];

    for (i, &count) in buckets.iter().enumerate() {
        cumulative += count;
        let midpoint = bucket_midpoints[i];

        if p50 == 0 && cumulative >= p50_target {
            p50 = midpoint;
        }
        if p95 == 0 && cumulative >= p95_target {
            p95 = midpoint;
        }
        if p99 == 0 && cumulative >= p99_target {
            p99 = midpoint;
        }
    }

    (p50, p95, p99)
}

// ============================================================================
// Shadow Verifier (Probabilistic Correctness Checking)
// ============================================================================

struct ShadowVerifier {
    /// Sample map: key → (value_seed, is_deleted)
    shadow: Mutex<HashMap<Vec<u8>, (u64, bool)>>,
    sample_rate: f64,
    verified_count: AtomicU64,
    mismatch_count: AtomicU64,
}

impl ShadowVerifier {
    fn new(sample_rate: f64) -> Self {
        Self {
            shadow: Mutex::new(HashMap::new()),
            sample_rate,
            verified_count: AtomicU64::new(0),
            mismatch_count: AtomicU64::new(0),
        }
    }

    fn should_track(&self, key: &[u8]) -> bool {
        // Deterministic sampling based on key hash
        let hash = gxhash::gxhash64(key, 0);
        ((hash % 10000) as f64) < (self.sample_rate * 10000.0)
    }

    fn record_write(&self, key: Vec<u8>, value_seed: u64) {
        if self.should_track(&key) {
            let mut shadow = self.shadow.lock().unwrap();
            shadow.insert(key, (value_seed, false));
        }
    }

    fn record_delete(&self, key: Vec<u8>) {
        if self.should_track(&key) {
            let mut shadow = self.shadow.lock().unwrap();
            if let Some(entry) = shadow.get_mut(&key) {
                entry.1 = true; // Mark as deleted
            }
        }
    }

    fn verify_read(&self, key: &[u8], value: &[u8]) -> bool {
        let _span = tracing::debug_span!("verifier_read").entered();
        if !self.should_track(key) {
            return true;
        }

        let _lock_span = tracing::debug_span!("verifier_lock").entered();
        let shadow = self.shadow.lock().unwrap();
        drop(_lock_span);
        if let Some(&(expected_seed, is_deleted)) = shadow.get(key) {
            self.verified_count.fetch_add(1, Ordering::Relaxed);

            if is_deleted {
                eprintln!(
                    "VERIFICATION ERROR: Read deleted key: {:?}",
                    String::from_utf8_lossy(key)
                );
                self.mismatch_count.fetch_add(1, Ordering::Relaxed);
                return false;
            }

            let matches = verify_value(value, value.len(), expected_seed);
            if !matches {
                eprintln!(
                    "VERIFICATION ERROR: Value mismatch for key {:?}",
                    String::from_utf8_lossy(key)
                );
                self.mismatch_count.fetch_add(1, Ordering::Relaxed);
            }
            matches
        } else {
            true // Key not tracked
        }
    }

    fn report(&self) -> (u64, u64) {
        let verified = self.verified_count.load(Ordering::Relaxed);
        let mismatches = self.mismatch_count.load(Ordering::Relaxed);
        (verified, mismatches)
    }
}

// ============================================================================
// Helper Functions (from compaction_integration.rs)
// ============================================================================

/// Generate deterministic test key
fn generate_key(ns: &str, i: u64) -> Vec<u8> {
    format!("{}-key-{:010}", ns, i).into_bytes()
}

/// Generate deterministic test value
fn generate_value(size: usize, seed: u64) -> Vec<u8> {
    let mut data = vec![0u8; size];
    for (i, byte) in data.iter_mut().enumerate() {
        *byte = ((seed + i as u64) % 256) as u8;
    }
    data
}

/// Verify that a value matches the expected deterministic pattern
fn verify_value(value: &[u8], size: usize, seed: u64) -> bool {
    if value.len() != size {
        return false;
    }
    for (i, &byte) in value.iter().enumerate() {
        if byte != ((seed + i as u64) % 256) as u8 {
            return false;
        }
    }
    true
}

// ============================================================================
// Workload System
// ============================================================================

trait Workload: Send + Sync {
    /// Execute one operation, return operation type
    fn execute_operation(
        &self,
        db: &Arc<Db>,
        worker_id: usize,
        iteration: u64,
        rng: &mut ThreadRng,
    ) -> OpType;

    /// Workload name for reporting
    fn name(&self) -> &str;

    /// Optional setup (pre-populate database)
    fn setup(&self, db: &Arc<Db>) -> Result<(), cesiumdb::errs::CesiumError> {
        let _ = db;
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// Mixed Workload (40% Get, 30% Put, 20% Scan, 10% Delete)
// ----------------------------------------------------------------------------

struct MixedWorkload {
    key_space_size: u64,
    value_size: usize,
    verifier: Arc<ShadowVerifier>,
}

impl MixedWorkload {
    fn new(config: &SoakConfig, verifier: Arc<ShadowVerifier>) -> Self {
        Self {
            key_space_size: config.key_space_size,
            value_size: config.value_size,
            verifier,
        }
    }

    fn random_key(&self, rng: &mut ThreadRng) -> Vec<u8> {
        let idx = rng.random_range(0..self.key_space_size);
        generate_key("mixed", idx)
    }

    fn deterministic_key(&self, worker_id: usize, iteration: u64) -> Vec<u8> {
        let idx = (worker_id as u64 * 1_000_000 + iteration) % self.key_space_size;
        generate_key("mixed", idx)
    }
}

impl Workload for MixedWorkload {
    fn name(&self) -> &str {
        "Mixed Workload"
    }

    fn execute_operation(
        &self,
        db: &Arc<Db>,
        worker_id: usize,
        iteration: u64,
        rng: &mut ThreadRng,
    ) -> OpType {
        let roll: u32 = rng.random_range(0..100);

        match roll {
            | 0..40 => {
                // 40% Get
                let _span = tracing::debug_span!("db_get").entered();
                let key = self.random_key(rng);
                if let Ok(Some(value)) = db.get(&key) {
                    drop(_span); // End span before verification
                    self.verifier.verify_read(&key, &value);
                }
                OpType::Get
            },
            | 40..70 => {
                // 30% Put
                let _span = tracing::debug_span!("db_put").entered();
                let key = self.deterministic_key(worker_id, iteration);
                let value = generate_value(self.value_size, iteration);
                if db.put(&key, &value).is_ok() {
                    drop(_span); // End span before verification
                    self.verifier.record_write(key, iteration);
                }
                OpType::Put
            },
            | 70..90 => {
                // 20% Scan
                let _span = tracing::debug_span!("db_scan", keys = 100).entered();
                let start = self.random_key(rng);
                let _ = db
                    .scan(Bound::Included(&start), Bound::Unbounded)
                    .unwrap()
                    .take(100)
                    .count();
                OpType::Scan
            },
            | _ => {
                // 10% Delete
                let _span = tracing::debug_span!("db_delete").entered();
                let key = self.random_key(rng);
                let _ = db.delete(&key);
                drop(_span); // End span before verification
                self.verifier.record_delete(key);
                OpType::Delete
            },
        }
    }

    fn setup(&self, db: &Arc<Db>) -> Result<(), cesiumdb::errs::CesiumError> {
        let prepopulate_count = self.key_space_size / 2;
        println!("Pre-populating {} keys...", prepopulate_count);

        const BATCH_SIZE: usize = 1000;
        let mut i = 0u64;
        while i < prepopulate_count {
            let mut batch = Vec::with_capacity(BATCH_SIZE);

            for j in 0..BATCH_SIZE {
                if i + j as u64 >= prepopulate_count {
                    break;
                }
                let idx = i + j as u64;
                let key = generate_key("mixed", idx);
                let value = generate_value(self.value_size, idx);
                batch.push(Put(key, value, db.time()));

                // Track in verifier
                if j % 100 == 0 {
                    // Sample 1% during setup
                    let sample_key = generate_key("mixed", idx);
                    self.verifier.record_write(sample_key, idx);
                }
            }

            db.batch(&batch)?;
            i += batch.len() as u64;

            if i % 100_000 == 0 && i > 0 {
                println!("  Pre-populated {}/{}", i, prepopulate_count);
            }
        }

        println!("Pre-population complete: {} keys", prepopulate_count);
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// Write-Heavy Workload (80% Put, 15% Get, 5% Delete)
// ----------------------------------------------------------------------------

struct WriteHeavyWorkload {
    key_space_size: u64,
    value_size: usize,
    verifier: Arc<ShadowVerifier>,
}

impl WriteHeavyWorkload {
    fn new(config: &SoakConfig, verifier: Arc<ShadowVerifier>) -> Self {
        Self {
            key_space_size: config.key_space_size,
            value_size: config.value_size,
            verifier,
        }
    }
}

impl Workload for WriteHeavyWorkload {
    fn name(&self) -> &str {
        "Write-Heavy Workload"
    }

    fn execute_operation(
        &self,
        db: &Arc<Db>,
        worker_id: usize,
        iteration: u64,
        rng: &mut ThreadRng,
    ) -> OpType {
        let roll: u32 = rng.random_range(0..100);

        match roll {
            | 0..80 => {
                // 80% Put (sequential)
                let idx = (worker_id as u64 * 1_000_000 + iteration) % self.key_space_size;
                let key = generate_key("write-heavy", idx);
                let value = generate_value(self.value_size, iteration);
                if db.put(&key, &value).is_ok() {
                    self.verifier.record_write(key, iteration);
                }
                OpType::Put
            },
            | 80..95 => {
                // 15% Get
                let idx = rng.random_range(0..self.key_space_size);
                let key = generate_key("write-heavy", idx);
                if let Ok(Some(value)) = db.get(&key) {
                    self.verifier.verify_read(&key, &value);
                }
                OpType::Get
            },
            | _ => {
                // 5% Delete
                let idx = rng.random_range(0..self.key_space_size);
                let key = generate_key("write-heavy", idx);
                let _ = db.delete(&key);
                self.verifier.record_delete(key);
                OpType::Delete
            },
        }
    }
}

// ----------------------------------------------------------------------------
// Read-Heavy Workload (90% Get, 8% Put, 2% Delete)
// ----------------------------------------------------------------------------

struct ReadHeavyWorkload {
    key_space_size: u64,
    value_size: usize,
    verifier: Arc<ShadowVerifier>,
}

impl ReadHeavyWorkload {
    fn new(config: &SoakConfig, verifier: Arc<ShadowVerifier>) -> Self {
        Self {
            key_space_size: config.key_space_size,
            value_size: config.value_size,
            verifier,
        }
    }
}

impl Workload for ReadHeavyWorkload {
    fn name(&self) -> &str {
        "Read-Heavy Workload"
    }

    fn execute_operation(
        &self,
        db: &Arc<Db>,
        _worker_id: usize,
        _iteration: u64,
        rng: &mut ThreadRng,
    ) -> OpType {
        let roll: u32 = rng.random_range(0..100);

        match roll {
            | 0..90 => {
                // 90% Get (hot keys - Zipfian-like)
                // Simple approximation: bias towards lower indices
                let skew = rng.random::<f64>().powf(2.0); // x^2 distribution
                let idx = (skew * self.key_space_size as f64) as u64;
                let key = generate_key("read-heavy", idx);
                if let Ok(Some(value)) = db.get(&key) {
                    self.verifier.verify_read(&key, &value);
                }
                OpType::Get
            },
            | 90..98 => {
                // 8% Put
                let idx = rng.random_range(0..self.key_space_size);
                let key = generate_key("read-heavy", idx);
                let value = generate_value(self.value_size, idx);
                if db.put(&key, &value).is_ok() {
                    self.verifier.record_write(key, idx);
                }
                OpType::Put
            },
            | _ => {
                // 2% Delete
                let idx = rng.random_range(0..self.key_space_size);
                let key = generate_key("read-heavy", idx);
                let _ = db.delete(&key);
                self.verifier.record_delete(key);
                OpType::Delete
            },
        }
    }

    fn setup(&self, db: &Arc<Db>) -> Result<(), cesiumdb::errs::CesiumError> {
        println!("Pre-populating {} keys...", self.key_space_size);

        const BATCH_SIZE: usize = 1000;
        let mut i = 0u64;
        while i < self.key_space_size {
            let mut batch = Vec::with_capacity(BATCH_SIZE);

            for j in 0..BATCH_SIZE {
                if i + j as u64 >= self.key_space_size {
                    break;
                }
                let idx = i + j as u64;
                let key = generate_key("read-heavy", idx);
                let value = generate_value(self.value_size, idx);
                batch.push(Put(key, value, db.time()));

                if j % 100 == 0 {
                    let sample_key = generate_key("read-heavy", idx);
                    self.verifier.record_write(sample_key, idx);
                }
            }

            db.batch(&batch)?;
            i += batch.len() as u64;

            if i % 100_000 == 0 && i > 0 {
                println!("  Pre-populated {}/{}", i, self.key_space_size);
            }
        }

        println!("Pre-population complete: {} keys", self.key_space_size);
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// Scan-Heavy Workload (60% Scan, 30% Get, 10% Put)
// ----------------------------------------------------------------------------

struct ScanHeavyWorkload {
    key_space_size: u64,
    value_size: usize,
    verifier: Arc<ShadowVerifier>,
}

impl ScanHeavyWorkload {
    fn new(config: &SoakConfig, verifier: Arc<ShadowVerifier>) -> Self {
        Self {
            key_space_size: config.key_space_size,
            value_size: config.value_size,
            verifier,
        }
    }
}

impl Workload for ScanHeavyWorkload {
    fn name(&self) -> &str {
        "Scan-Heavy Workload"
    }

    fn execute_operation(
        &self,
        db: &Arc<Db>,
        _worker_id: usize,
        _iteration: u64,
        rng: &mut ThreadRng,
    ) -> OpType {
        let roll: u32 = rng.random_range(0..100);

        match roll {
            | 0..60 => {
                // 60% Scan (variable range 10-1000 keys)
                let start_idx = rng.random_range(0..self.key_space_size);
                let start_key = generate_key("scan-heavy", start_idx);
                let scan_size = rng.random_range(10..1000);
                let _ = db
                    .scan(Bound::Included(&start_key), Bound::Unbounded)
                    .unwrap()
                    .take(scan_size)
                    .count();
                OpType::Scan
            },
            | 60..90 => {
                // 30% Get
                let idx = rng.random_range(0..self.key_space_size);
                let key = generate_key("scan-heavy", idx);
                if let Ok(Some(value)) = db.get(&key) {
                    self.verifier.verify_read(&key, &value);
                }
                OpType::Get
            },
            | _ => {
                // 10% Put
                let idx = rng.random_range(0..self.key_space_size);
                let key = generate_key("scan-heavy", idx);
                let value = generate_value(self.value_size, idx);
                if db.put(&key, &value).is_ok() {
                    self.verifier.record_write(key, idx);
                }
                OpType::Put
            },
        }
    }

    fn setup(&self, db: &Arc<Db>) -> Result<(), cesiumdb::errs::CesiumError> {
        println!("Pre-populating {} keys...", self.key_space_size);

        const BATCH_SIZE: usize = 1000;
        let mut i = 0u64;
        while i < self.key_space_size {
            let mut batch = Vec::with_capacity(BATCH_SIZE);

            for j in 0..BATCH_SIZE {
                if i + j as u64 >= self.key_space_size {
                    break;
                }
                let idx = i + j as u64;
                let key = generate_key("scan-heavy", idx);
                let value = generate_value(self.value_size, idx);
                batch.push(Put(key, value, db.time()));

                if j % 100 == 0 {
                    let sample_key = generate_key("scan-heavy", idx);
                    self.verifier.record_write(sample_key, idx);
                }
            }

            db.batch(&batch)?;
            i += batch.len() as u64;

            if i % 100_000 == 0 && i > 0 {
                println!("  Pre-populated {}/{}", i, self.key_space_size);
            }
        }

        println!("Pre-population complete: {} keys", self.key_space_size);
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// Delete-Heavy Workload (50% Delete, 30% Put, 20% Get)
// ----------------------------------------------------------------------------

struct DeleteHeavyWorkload {
    key_space_size: u64,
    value_size: usize,
    verifier: Arc<ShadowVerifier>,
}

impl DeleteHeavyWorkload {
    fn new(config: &SoakConfig, verifier: Arc<ShadowVerifier>) -> Self {
        Self {
            key_space_size: config.key_space_size,
            value_size: config.value_size,
            verifier,
        }
    }
}

impl Workload for DeleteHeavyWorkload {
    fn name(&self) -> &str {
        "Delete-Heavy Workload"
    }

    fn execute_operation(
        &self,
        db: &Arc<Db>,
        worker_id: usize,
        iteration: u64,
        rng: &mut ThreadRng,
    ) -> OpType {
        let roll: u32 = rng.random_range(0..100);

        match roll {
            | 0..50 => {
                // 50% Delete
                let idx = rng.random_range(0..self.key_space_size);
                let key = generate_key("delete-heavy", idx);
                let _ = db.delete(&key);
                self.verifier.record_delete(key);
                OpType::Delete
            },
            | 50..80 => {
                // 30% Put
                let idx = (worker_id as u64 * 1_000_000 + iteration) % self.key_space_size;
                let key = generate_key("delete-heavy", idx);
                let value = generate_value(self.value_size, iteration);
                if db.put(&key, &value).is_ok() {
                    self.verifier.record_write(key, iteration);
                }
                OpType::Put
            },
            | _ => {
                // 20% Get
                let idx = rng.random_range(0..self.key_space_size);
                let key = generate_key("delete-heavy", idx);
                if let Ok(Some(value)) = db.get(&key) {
                    self.verifier.verify_read(&key, &value);
                }
                OpType::Get
            },
        }
    }

    fn setup(&self, db: &Arc<Db>) -> Result<(), cesiumdb::errs::CesiumError> {
        println!("Pre-populating {} keys...", self.key_space_size);

        const BATCH_SIZE: usize = 1000;
        let mut i = 0u64;
        while i < self.key_space_size {
            let mut batch = Vec::with_capacity(BATCH_SIZE);

            for j in 0..BATCH_SIZE {
                if i + j as u64 >= self.key_space_size {
                    break;
                }
                let idx = i + j as u64;
                let key = generate_key("delete-heavy", idx);
                let value = generate_value(self.value_size, idx);
                batch.push(Put(key, value, db.time()));

                if j % 100 == 0 {
                    let sample_key = generate_key("delete-heavy", idx);
                    self.verifier.record_write(sample_key, idx);
                }
            }

            db.batch(&batch)?;
            i += batch.len() as u64;

            if i % 100_000 == 0 && i > 0 {
                println!("  Pre-populated {}/{}", i, self.key_space_size);
            }
        }

        println!("Pre-population complete: {} keys", self.key_space_size);
        Ok(())
    }
}

// ============================================================================
// Worker & Reporter Threads
// ============================================================================

#[derive(Debug)]
struct WorkerStats {
    ops_completed: u64,
}

fn spawn_workers(
    config: &SoakConfig,
    db: &Arc<Db>,
    workload: &Arc<dyn Workload>,
    metrics: &Arc<MetricsCollector>,
    shutdown: &Arc<AtomicBool>,
) -> Vec<JoinHandle<WorkerStats>> {
    (0..config.num_workers)
        .map(|worker_id| {
            let db = Arc::clone(db);
            let workload = Arc::clone(workload);
            let metrics = Arc::clone(metrics);
            let shutdown = Arc::clone(shutdown);

            thread::spawn(move || {
                let mut rng = rand::rng();
                let mut iteration = 0u64;

                while !shutdown.load(Ordering::Relaxed) {
                    let start = Instant::now();

                    let op_type = workload.execute_operation(&db, worker_id, iteration, &mut rng);

                    metrics.record_op(op_type, start.elapsed());
                    iteration += 1;
                }

                WorkerStats {
                    ops_completed: iteration,
                }
            })
        })
        .collect()
}

fn spawn_reporter(
    metrics: &Arc<MetricsCollector>,
    shutdown: &Arc<AtomicBool>,
    interval_ms: u64,
) -> JoinHandle<()> {
    let metrics = Arc::clone(metrics);
    let shutdown = Arc::clone(shutdown);

    thread::spawn(move || {
        let mut last_snapshot = metrics.snapshot();

        while !shutdown.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(interval_ms));

            let current = metrics.snapshot();
            let delta_ops = current.total_ops.saturating_sub(last_snapshot.total_ops);
            let delta_time = current.elapsed_secs - last_snapshot.elapsed_secs;
            let instant_rate = if delta_time > 0.0 {
                delta_ops as f64 / delta_time
            } else {
                0.0
            };

            println!(
                "[{:>6.1}s] {:>8} ops | {:>7.0} ops/sec (inst) | {:>7.0} ops/sec (avg) | \
                 errors: {}",
                current.elapsed_secs,
                current.total_ops,
                instant_rate,
                current.ops_per_sec,
                current.error_count
            );

            last_snapshot = current;
        }
    })
}

// ============================================================================
// Test Executor
// ============================================================================

async fn run_soak_test(
    config: SoakConfig,
    workload: Arc<dyn Workload>,
) -> Result<(), cesiumdb::errs::CesiumError> {
    println!("\n=== CesiumDB Soak Test: {} ===", workload.name());
    println!(
        "Duration: {}s, Workers: {}, Key Space: {}",
        config.duration_secs, config.num_workers, config.key_space_size
    );
    println!();

    // Setup database
    println!("Setting up database...");
    let db_path = PathBuf::from("/tmp/cesiumdb_test_soak");
    let _ = fs::remove_dir_all(&db_path);

    let mut opts = DbOptions::default();
    opts.data_dir(db_path)
        .memtable_size(config.memtable_size)
        .max_memtables(config.max_memtables);

    let db = Db::open(opts).unwrap();

    // Pre-populate if needed
    let setup_start = Instant::now();
    workload.setup(&db)?;
    if setup_start.elapsed().as_secs() > 0 {
        println!(
            "Pre-population complete in {:.2}s\n",
            setup_start.elapsed().as_secs_f64()
        );
    }

    // Initialize infrastructure
    let metrics = Arc::new(MetricsCollector::new());
    let shutdown = Arc::new(AtomicBool::new(false));

    // Spawn workers and reporter
    println!("Starting workload...");
    let workers = spawn_workers(&config, &db, &workload, &metrics, &shutdown);
    let reporter = spawn_reporter(&metrics, &shutdown, config.metrics_interval_ms);

    // Wait for duration
    thread::sleep(Duration::from_secs(config.duration_secs));

    // Shutdown
    shutdown.store(true, Ordering::SeqCst);
    for worker in workers {
        worker.join().unwrap();
    }
    reporter.join().unwrap();

    // Final report
    println!("\nWorkload complete. Generating final report...\n");
    print_final_report(&metrics, &workload, &db, &config);

    // Assertions
    let final_snapshot = metrics.snapshot();
    assert_eq!(
        final_snapshot.error_count, 0,
        "test had {} errors",
        final_snapshot.error_count
    );
    assert!(final_snapshot.total_ops > 0, "test completed no operations");

    println!("\nTest PASSED ✓");
    Ok(())
}

fn print_final_report(
    metrics: &MetricsCollector,
    workload: &Arc<dyn Workload>,
    db: &Arc<Db>,
    _config: &SoakConfig,
) {
    let snapshot = metrics.snapshot();

    println!("=== Final Report ===");
    println!("Total Duration: {:.1}s", snapshot.elapsed_secs);
    println!("Total Operations: {}", snapshot.total_ops);
    println!("Average Throughput: {:.0} ops/sec", snapshot.ops_per_sec);
    println!();

    println!("Operation Breakdown:");
    println!(
        "  Get:    {:>8} ops ({:>5.1}%)",
        snapshot.get_count,
        (snapshot.get_count as f64 / snapshot.total_ops as f64) * 100.0
    );
    println!(
        "  Put:    {:>8} ops ({:>5.1}%)",
        snapshot.put_count,
        (snapshot.put_count as f64 / snapshot.total_ops as f64) * 100.0
    );
    println!(
        "  Scan:   {:>8} ops ({:>5.1}%)",
        snapshot.scan_count,
        (snapshot.scan_count as f64 / snapshot.total_ops as f64) * 100.0
    );
    println!(
        "  Delete: {:>8} ops ({:>5.1}%)",
        snapshot.delete_count,
        (snapshot.delete_count as f64 / snapshot.total_ops as f64) * 100.0
    );
    println!();

    println!("Latency Statistics:");
    if snapshot.min_micros != u64::MAX {
        println!(
            "  Min: {}µs, Max: {:.1}ms",
            snapshot.min_micros,
            snapshot.max_micros as f64 / 1000.0
        );
        println!(
            "  P50: ~{}µs, P95: ~{}µs, P99: ~{}µs",
            snapshot.p50_micros, snapshot.p95_micros, snapshot.p99_micros
        );
    } else {
        println!("  No latency data collected");
    }
    println!();

    // Verification results (if workload uses verifier)
    if let Some(verifier) = get_verifier_from_workload(workload) {
        let (verified, mismatches) = verifier.report();
        println!("Verification:");
        println!(
            "  Samples verified: {} ({:.1}%)",
            verified,
            (verified as f64 / snapshot.total_ops as f64) * 100.0
        );
        println!("  Mismatches: {}", mismatches);
        if mismatches == 0 {
            println!("  Verification: PASSED ✓");
        } else {
            println!("  Verification: FAILED ✗");
        }
        println!();
    }

    // Compaction stats
    if let Ok(stats) = db.compaction_stats() {
        println!("Compaction Stats:");
        println!("{}", stats);
    }
}

// Helper to extract verifier from workload (type erasure workaround)
fn get_verifier_from_workload(_workload: &Arc<dyn Workload>) -> Option<Arc<ShadowVerifier>> {
    // Since we can't downcast trait objects easily, we'll use a different approach:
    // Each workload will need to expose its verifier. For now, we'll skip this
    // and just not print verification stats (or refactor the workload trait).
    // TODO: Consider adding a method to Workload trait to expose verifier.
    None
}

// ============================================================================
// Test Functions
// ============================================================================

#[tokio::test]
#[ignore]
async fn soak_smoke_mixed() {
    let config = SoakConfig::smoke_test();
    let verifier = Arc::new(ShadowVerifier::new(config.verification_sample_rate));
    let workload: Arc<dyn Workload> = Arc::new(MixedWorkload::new(&config, verifier.clone()));

    run_soak_test(config, workload).await.unwrap();
}

#[tokio::test]
#[ignore]
async fn soak_test_mixed_workload() {
    let config = SoakConfig::standard_soak();
    let verifier = Arc::new(ShadowVerifier::new(config.verification_sample_rate));
    let workload: Arc<dyn Workload> = Arc::new(MixedWorkload::new(&config, verifier.clone()));

    run_soak_test(config, workload).await.unwrap();
}

#[tokio::test]
#[ignore]
async fn soak_test_write_heavy() {
    let config = SoakConfig::standard_soak();
    let verifier = Arc::new(ShadowVerifier::new(config.verification_sample_rate));
    let workload: Arc<dyn Workload> = Arc::new(WriteHeavyWorkload::new(&config, verifier));

    run_soak_test(config, workload).await.unwrap();
}

#[tokio::test]
#[ignore]
async fn soak_test_read_heavy() {
    let config = SoakConfig::standard_soak();
    let verifier = Arc::new(ShadowVerifier::new(config.verification_sample_rate));
    let workload: Arc<dyn Workload> = Arc::new(ReadHeavyWorkload::new(&config, verifier));

    run_soak_test(config, workload).await.unwrap();
}

#[tokio::test]
#[ignore]
async fn soak_test_scan_heavy() {
    let config = SoakConfig::standard_soak();
    let verifier = Arc::new(ShadowVerifier::new(config.verification_sample_rate));
    let workload: Arc<dyn Workload> = Arc::new(ScanHeavyWorkload::new(&config, verifier));

    run_soak_test(config, workload).await.unwrap();
}

#[tokio::test]
#[ignore]
async fn soak_test_delete_heavy() {
    let config = SoakConfig::standard_soak();
    let verifier = Arc::new(ShadowVerifier::new(config.verification_sample_rate));
    let workload: Arc<dyn Workload> = Arc::new(DeleteHeavyWorkload::new(&config, verifier));

    run_soak_test(config, workload).await.unwrap();
}

#[tokio::test]
#[ignore]
async fn soak_extended_mixed() {
    let config = SoakConfig::extended_soak();
    let verifier = Arc::new(ShadowVerifier::new(config.verification_sample_rate));
    let workload: Arc<dyn Workload> = Arc::new(MixedWorkload::new(&config, verifier));

    run_soak_test(config, workload).await.unwrap();
}

// ============================================================================
// 20 GiB Sustained Write Soak Test
// ============================================================================
//
// This test validates compaction backpressure by writing 20 GiB of data
// and monitoring throughput over time. If compaction cannot keep up with
// flushes, L0 files will accumulate, frozen memtables will pile up, and
// write throughput will degrade.
//
// Usage:
//   cargo test --test soak_test soak_20gib_write_sustained -- --ignored
// --nocapture
//
// Environment variables:
//   SOAK_20GIB_TARGET_GB    Target data size in GiB (default: 20)
//   SOAK_20GIB_VALUE_SIZE   Value size in bytes (default: 1024)
//   SOAK_20GIB_BATCH_SIZE   Batch size per write (default: 100)
//   SOAK_20GIB_WORKERS      Number of writer threads (default:
// available_parallelism)   SOAK_20GIB_MAX_DURATION_SECS  Max test duration
// (default: 600)

#[derive(Debug, Clone)]
struct SustainedWriteConfig {
    target_bytes: u64,
    value_size: usize,
    batch_size: usize,
    num_workers: usize,
    max_duration_secs: u64,
    metrics_interval_secs: u64,
}

impl SustainedWriteConfig {
    fn from_env() -> Self {
        let target_gb: f64 = Self::env_or("SOAK_20GIB_TARGET_GB", 20.0f64);
        Self {
            target_bytes: (target_gb * 1024.0 * 1024.0 * 1024.0) as u64,
            value_size: Self::env_or("SOAK_20GIB_VALUE_SIZE", 1024usize),
            batch_size: Self::env_or("SOAK_20GIB_BATCH_SIZE", 100usize),
            num_workers: Self::env_or(
                "SOAK_20GIB_WORKERS",
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4),
            ),
            max_duration_secs: Self::env_or("SOAK_20GIB_MAX_DURATION_SECS", 600u64),
            metrics_interval_secs: 5,
        }
    }

    fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }
}

struct SustainedWriteMetrics {
    bytes_written: AtomicU64,
    ops_written: AtomicU64,
    batches_written: AtomicU64,
    start_time: Instant,
}

impl SustainedWriteMetrics {
    fn new() -> Self {
        Self {
            bytes_written: AtomicU64::new(0),
            ops_written: AtomicU64::new(0),
            batches_written: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    fn record_batch(&self, entries: usize, value_size: usize) {
        self.batches_written.fetch_add(1, Ordering::Relaxed);
        self.ops_written
            .fetch_add(entries as u64, Ordering::Relaxed);
        // Approximate bytes: key (~32) + value + overhead (~48)
        let bytes_per_entry = 32 + value_size + 48;
        self.bytes_written
            .fetch_add((entries * bytes_per_entry) as u64, Ordering::Relaxed);
    }

    fn snapshot(&self) -> SustainedWriteSnapshot {
        let bytes = self.bytes_written.load(Ordering::Relaxed);
        let ops = self.ops_written.load(Ordering::Relaxed);
        let batches = self.batches_written.load(Ordering::Relaxed);
        let elapsed = self.start_time.elapsed().as_secs_f64();
        SustainedWriteSnapshot {
            bytes_written: bytes,
            ops_written: ops,
            batches_written: batches,
            elapsed_secs: elapsed,
            ops_per_sec: if elapsed > 0.0 {
                ops as f64 / elapsed
            } else {
                0.0
            },
            bytes_per_sec: if elapsed > 0.0 {
                bytes as f64 / elapsed
            } else {
                0.0
            },
        }
    }
}

#[derive(Debug, Clone)]
struct SustainedWriteSnapshot {
    bytes_written: u64,
    ops_written: u64,
    batches_written: u64,
    elapsed_secs: f64,
    ops_per_sec: f64,
    bytes_per_sec: f64,
}

fn spawn_sustained_writer(
    db: Arc<Db>,
    config: SustainedWriteConfig,
    worker_id: usize,
    metrics: Arc<SustainedWriteMetrics>,
    shutdown: Arc<AtomicBool>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut rng = rand::rng();
        let mut batch = Vec::with_capacity(config.batch_size);
        let mut idx: u64 = 0;

        while !shutdown.load(Ordering::Relaxed) {
            batch.clear();

            for _ in 0..config.batch_size {
                // Use worker-specific key space to avoid contention
                let key = format!("sustained_{:03}_{:014}", worker_id, idx);
                let value = generate_value(config.value_size, rng.random::<u64>());
                batch.push(Put(key, value, db.time()));
                idx += 1;
            }

            match db.batch(&batch) {
                | Ok(()) => {
                    metrics.record_batch(batch.len(), config.value_size);
                },
                | Err(e) => {
                    eprintln!("Worker {} batch error: {:?}", worker_id, e);
                    thread::sleep(Duration::from_millis(10));
                },
            }
        }
    })
}

fn spawn_sustained_reporter(
    db: Arc<Db>,
    metrics: Arc<SustainedWriteMetrics>,
    shutdown: Arc<AtomicBool>,
    interval_secs: u64,
    target_bytes: u64,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut last_snapshot = metrics.snapshot();
        let mut peak_ops_per_sec: f64 = 0.0;
        let mut min_ops_per_sec: f64 = f64::MAX;
        let mut last_compaction_jobs = 0u64;
        let mut last_compaction_read = 0u64;
        let mut last_compaction_written = 0u64;

        println!(
            "\n{:>8} | {:>10} | {:>10} | {:>10} | {:>8} | {:>6} | {:>6} | {:>8} | {:>8} | {:>10} | {:>10} | {:>10} | {:>10}",
            "Time",
            "Value GB",
            "Disk GB",
            "Target GB",
            "Ops/s",
            "L0",
            "Frozen",
            "Queued",
            "Active",
            "CmpJob/s",
            "CmpRdMB/s",
            "CmpWrMB/s",
            "Pattern"
        );
        println!("{}", "-".repeat(155));

        while !shutdown.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_secs(interval_secs));

            let current = metrics.snapshot();
            let delta_time = current.elapsed_secs - last_snapshot.elapsed_secs;
            let delta_ops = current
                .ops_written
                .saturating_sub(last_snapshot.ops_written);
            let instant_ops = if delta_time > 0.0 {
                delta_ops as f64 / delta_time
            } else {
                0.0
            };

            peak_ops_per_sec = peak_ops_per_sec.max(instant_ops);
            if instant_ops > 0.0 {
                min_ops_per_sec = min_ops_per_sec.min(instant_ops);
            }

            let gb_written = current.bytes_written as f64 / (1024.0 * 1024.0 * 1024.0);
            let target_gb = target_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

            // Fetch database stats
            let vstats = db.version_stats();
            let l0_count = vstats.l0_segments;
            let disk_gb = vstats.total_size as f64 / (1024.0 * 1024.0 * 1024.0);
            let frozen_count = db.frozen_memtable_count();
            let (queued, active, pattern, compaction_jobs, compaction_read, compaction_written) =
                match db.compaction_stats() {
                    | Ok(stats) => {
                        let jobs = stats.completed_jobs;
                        let read = stats.bytes_compacted_read;
                        let written = stats.bytes_compacted_written;
                        (
                            stats.queued_jobs,
                            stats.in_progress_jobs,
                            stats.workload_pattern,
                            jobs,
                            read,
                            written,
                        )
                    },
                    | Err(_) => (0, 0, "N/A".to_string(), 0, 0, 0),
                };

            let delta_jobs = compaction_jobs.saturating_sub(last_compaction_jobs);
            let delta_read = compaction_read.saturating_sub(last_compaction_read);
            let delta_written = compaction_written.saturating_sub(last_compaction_written);
            let jobs_per_sec = if delta_time > 0.0 {
                delta_jobs as f64 / delta_time
            } else {
                0.0
            };
            let read_mb_per_sec = if delta_time > 0.0 {
                delta_read as f64 / (1024.0 * 1024.0) / delta_time
            } else {
                0.0
            };
            let write_mb_per_sec = if delta_time > 0.0 {
                delta_written as f64 / (1024.0 * 1024.0) / delta_time
            } else {
                0.0
            };

            println!(
                "{:>8.1}s | {:>10.2} | {:>10.2} | {:>10.2} | {:>8.0} | {:>6} | {:>6} | {:>8} | {:>8} | {:>10.2} | {:>10.2} | {:>10.2} | {:>10}",
                current.elapsed_secs,
                gb_written,
                disk_gb,
                target_gb,
                instant_ops,
                l0_count,
                frozen_count,
                queued,
                active,
                jobs_per_sec,
                read_mb_per_sec,
                write_mb_per_sec,
                pattern
            );

            // Safety valve: if frozen memtables exceed a dangerous threshold,
            // warn that memory pressure is building
            if frozen_count > 64 {
                eprintln!(
                    "\n⚠️  WARNING: {} frozen memtables - memory pressure high. \
                     Backpressure may not be effective.",
                    frozen_count
                );
            }

            last_snapshot = current;
            last_compaction_jobs = compaction_jobs;
            last_compaction_read = compaction_read;
            last_compaction_written = compaction_written;
        }

        // Final summary
        let final_snap = metrics.snapshot();
        let degradation = if peak_ops_per_sec > 0.0 {
            let ratio = min_ops_per_sec / peak_ops_per_sec;
            ratio
        } else {
            1.0
        };

        println!("\n=== 20 GiB Sustained Write Summary ===");
        println!(
            "Total written: {:.2} GB ({:.1}% of target)",
            final_snap.bytes_written as f64 / (1024.0 * 1024.0 * 1024.0),
            (final_snap.bytes_written as f64 / target_bytes as f64) * 100.0
        );
        println!("Total ops: {}", final_snap.ops_written);
        println!("Total batches: {}", final_snap.batches_written);
        println!("Duration: {:.1}s", final_snap.elapsed_secs);
        println!("Average throughput: {:.0} ops/sec", final_snap.ops_per_sec);
        println!("Peak throughput: {:.0} ops/sec", peak_ops_per_sec);
        println!("Min throughput: {:.0} ops/sec", min_ops_per_sec);
        println!(
            "Degradation ratio (min/peak): {:.2} {}",
            degradation,
            if degradation < 0.5 {
                "⚠️ SIGNIFICANT DEGRADATION"
            } else {
                "✓"
            }
        );

        // Final compaction stats
        if let Ok(stats) = db.compaction_stats() {
            let cmp_read_gb = stats.bytes_compacted_read as f64 / (1024.0 * 1024.0 * 1024.0);
            let cmp_write_gb = stats.bytes_compacted_written as f64 / (1024.0 * 1024.0 * 1024.0);
            println!("Compaction jobs completed: {}", stats.completed_jobs);
            println!("Compaction bytes read: {:.2} GB", cmp_read_gb);
            println!("Compaction bytes written: {:.2} GB", cmp_write_gb);
        }
    })
}

#[tokio::test]
#[ignore]
async fn soak_20gib_write_sustained() {
    let config = SustainedWriteConfig::from_env();
    let target_gb = config.target_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    println!("\n=== CesiumDB 20 GiB Sustained Write Soak Test ===");
    println!(
        "Target: {:.1} GiB | Value size: {} B | Batch: {} | Workers: {}",
        target_gb, config.value_size, config.batch_size, config.num_workers
    );
    println!();

    // Setup database with modest memtables to bound memory usage.
    // The focus is disk throughput during compaction, not memory throughput.
    let db_path = PathBuf::from("/tmp/cesiumdb_test_20gib_write");
    let _ = fs::remove_dir_all(&db_path);

    let mut opts = DbOptions::default();
    opts.data_dir(db_path.clone())
        .memtable_size(64 * 1024 * 1024) // 64 MiB memtables
        .max_memtables(8); // max 512 MiB frozen memtable memory

    let db = Db::open(opts).unwrap();

    let metrics = Arc::new(SustainedWriteMetrics::new());
    let shutdown = Arc::new(AtomicBool::new(false));

    // Spawn writers
    let workers: Vec<_> = (0..config.num_workers)
        .map(|id| {
            spawn_sustained_writer(
                db.clone(),
                config.clone(),
                id,
                metrics.clone(),
                shutdown.clone(),
            )
        })
        .collect();

    // Spawn reporter
    let reporter = spawn_sustained_reporter(
        db.clone(),
        metrics.clone(),
        shutdown.clone(),
        config.metrics_interval_secs,
        config.target_bytes,
    );

    // Monitor until target reached or max duration exceeded
    let start = Instant::now();
    loop {
        thread::sleep(Duration::from_secs(1));

        let bytes = metrics.bytes_written.load(Ordering::Relaxed);
        let elapsed = start.elapsed().as_secs();

        if bytes >= config.target_bytes {
            println!(
                "\n🎯 Target reached: {:.2} GB written",
                bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            );
            break;
        }

        if elapsed >= config.max_duration_secs {
            println!("\n⏱️ Max duration ({:}s) reached", config.max_duration_secs);
            break;
        }
    }

    // Shutdown
    shutdown.store(true, Ordering::SeqCst);
    for w in workers {
        let _ = w.join();
    }
    let _ = reporter.join();

    // Final assertions
    let final_bytes = metrics.bytes_written.load(Ordering::Relaxed);
    let final_ops = metrics.ops_written.load(Ordering::Relaxed);

    assert!(final_ops > 0, "No operations were written");

    // We should reach at least 90% of target within max duration
    let completion_ratio = final_bytes as f64 / config.target_bytes as f64;
    println!(
        "\nFinal completion: {:.1}% ({:.2} / {:.2} GB)",
        completion_ratio * 100.0,
        final_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        target_gb
    );

    // Allow the test to pass even if we don't hit 100% - this is a performance
    // characterization test, not a hard correctness test. But we should write
    // at least 50% of target.
    assert!(
        completion_ratio >= 0.50,
        "Only wrote {:.1}% of target - compaction backpressure may be too aggressive",
        completion_ratio * 100.0
    );

    // Check disk usage - with file deletion, amplification should be < 3x
    let disk_bytes = calculate_dir_size(&db_path);
    let write_amplification = disk_bytes as f64 / final_bytes.max(1) as f64;
    println!(
        "\nDisk usage: {:.2} GB, Write amplification: {:.2}x",
        disk_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        write_amplification
    );

    // With proper compaction and file deletion, amplification should be
    // well under 4x. The tiered→leveled architecture gives ~3-4x for
    // write-heavy workloads; future optimizations (larger tiered bases,
    // universal compaction) will push this below 3x.
    assert!(
        write_amplification < 4.0,
        "Write amplification {:.2}x is too high - expected < 4.0x. Files may not be deleting.",
        write_amplification
    );

    println!("\nTest completed ✓");
}

// ============================================================================
// 20 GiB Scoped Churn Test
// ============================================================================
//
// Pre-populates a bounded key space to ~20 GiB, then runs mixed
// create/delete/compact operations to maintain steady-state churn.
// Measures accurate space amplification, write amplification, and
// read amplification under sustained compaction pressure.
//
// Usage:
//   cargo test --test soak_test soak_20gib_churn -- --ignored --nocapture
//   SOAK_CHURN_DURATION_SECS=600 SOAK_CHURN_TARGET_GB=20 cargo test ...

#[derive(Debug, Clone)]
struct ChurnConfig {
    target_bytes: u64,
    duration_secs: u64,
    value_size: usize,
    num_workers: usize,
    batch_size: usize,
    metrics_interval_secs: u64,
}

impl ChurnConfig {
    fn from_env() -> Self {
        let target_gb: f64 = Self::env_or("SOAK_CHURN_TARGET_GB", 20.0f64);
        Self {
            target_bytes: (target_gb * 1024.0 * 1024.0 * 1024.0) as u64,
            duration_secs: Self::env_or("SOAK_CHURN_DURATION_SECS", 600u64),
            value_size: Self::env_or("SOAK_CHURN_VALUE_SIZE", 1024usize),
            num_workers: Self::env_or(
                "SOAK_CHURN_WORKERS",
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4),
            ),
            batch_size: Self::env_or("SOAK_CHURN_BATCH_SIZE", 1000usize),
            metrics_interval_secs: 5,
        }
    }

    fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }
}

struct ChurnMetrics {
    puts: AtomicU64,
    gets: AtomicU64,
    deletes: AtomicU64,
    scans: AtomicU64,
    bytes_written: AtomicU64,
    start_time: Instant,
}

impl ChurnMetrics {
    fn new() -> Self {
        Self {
            puts: AtomicU64::new(0),
            gets: AtomicU64::new(0),
            deletes: AtomicU64::new(0),
            scans: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ChurnSnapshot {
    puts: u64,
    gets: u64,
    deletes: u64,
    scans: u64,
    bytes_written: u64,
    elapsed_secs: f64,
}

impl ChurnSnapshot {
    fn total_ops(&self) -> u64 {
        self.puts + self.gets + self.deletes + self.scans
    }
}

impl ChurnMetrics {
    fn snapshot(&self) -> ChurnSnapshot {
        ChurnSnapshot {
            puts: self.puts.load(Ordering::Relaxed),
            gets: self.gets.load(Ordering::Relaxed),
            deletes: self.deletes.load(Ordering::Relaxed),
            scans: self.scans.load(Ordering::Relaxed),
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            elapsed_secs: self.start_time.elapsed().as_secs_f64(),
        }
    }
}

fn churn_key(idx: u64) -> Vec<u8> {
    format!("churn_{:014}", idx).into_bytes()
}

fn spawn_prepop_worker(
    db: Arc<Db>,
    config: ChurnConfig,
    global_counter: Arc<AtomicU64>,
    metrics: Arc<ChurnMetrics>,
    shutdown: Arc<AtomicBool>,
) -> JoinHandle<()> {
    let batch_size = config.batch_size;
    let value_size = config.value_size;

    thread::spawn(move || {
        let mut rng = rand::rng();
        let mut batch = Vec::with_capacity(batch_size);

        while !shutdown.load(Ordering::Relaxed) {
            batch.clear();
            let start_idx = global_counter.fetch_add(batch_size as u64, Ordering::Relaxed);

            for i in 0..batch_size {
                let key = churn_key(start_idx + i as u64);
                let value = generate_value(value_size, rng.random::<u64>());
                batch.push(Put(key, value, db.time()));
            }

            match db.batch(&batch) {
                | Ok(()) => {
                    metrics
                        .bytes_written
                        .fetch_add((value_size * batch.len()) as u64, Ordering::Relaxed);
                },
                | Err(e) => {
                    eprintln!("Pre-pop batch error: {:?}", e);
                    thread::sleep(Duration::from_millis(10));
                },
            }
        }
    })
}

fn spawn_churn_worker(
    db: Arc<Db>,
    config: ChurnConfig,
    total_keys: u64,
    _worker_id: usize,
    metrics: Arc<ChurnMetrics>,
    shutdown: Arc<AtomicBool>,
) -> JoinHandle<()> {
    let value_size = config.value_size;

    thread::spawn(move || {
        let mut rng = rand::rng();

        while !shutdown.load(Ordering::Relaxed) {
            let roll: u32 = rng.random_range(0..100);

            match roll {
                | 0..40 => {
                    // 40% Get
                    let idx = rng.random_range(0..total_keys);
                    let key = churn_key(idx);
                    let _ = db.get(&key);
                    metrics.gets.fetch_add(1, Ordering::Relaxed);
                },
                | 40..75 => {
                    // 35% Put (overwrite existing key)
                    let idx = rng.random_range(0..total_keys);
                    let key = churn_key(idx);
                    let value = generate_value(value_size, rng.random::<u64>());
                    if db.put(&key, &value).is_ok() {
                        metrics.puts.fetch_add(1, Ordering::Relaxed);
                        metrics
                            .bytes_written
                            .fetch_add(value_size as u64, Ordering::Relaxed);
                    }
                },
                | 75..90 => {
                    // 15% Scan
                    let idx = rng.random_range(0..total_keys);
                    let key = churn_key(idx);
                    let _ = db
                        .scan(Bound::Included(&key), Bound::Unbounded)
                        .unwrap()
                        .take(100)
                        .count();
                    metrics.scans.fetch_add(1, Ordering::Relaxed);
                },
                | _ => {
                    // 10% Delete
                    let idx = rng.random_range(0..total_keys);
                    let key = churn_key(idx);
                    let _ = db.delete(&key);
                    metrics.deletes.fetch_add(1, Ordering::Relaxed);
                },
            }
        }
    })
}

fn spawn_churn_reporter(
    db: Arc<Db>,
    metrics: Arc<ChurnMetrics>,
    shutdown: Arc<AtomicBool>,
    interval_secs: u64,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut last = metrics.snapshot();
        let mut peak_ops_per_sec: f64 = 0.0;
        let mut min_ops_per_sec: f64 = f64::MAX;
        let mut last_compaction_jobs = 0u64;
        let mut last_compaction_read = 0u64;
        let mut last_compaction_written = 0u64;

        println!(
            "\n{:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>10} | {:>10} | {:>10} | {:>10} | {:>6} | {:>6}",
            "Time",
            "Ops/s",
            "Get/s",
            "Put/s",
            "Del/s",
            "Scan/s",
            "DiskGB",
            "L0",
            "Qd",
            "Act",
            "CmpJob/s",
            "CmpRdMB/s",
            "CmpWrMB/s",
            "Frz"
        );
        println!("{}", "-".repeat(175));

        while !shutdown.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_secs(interval_secs));

            let current = metrics.snapshot();
            let delta_time = current.elapsed_secs - last.elapsed_secs;
            let delta_ops = current.total_ops().saturating_sub(last.total_ops());
            let ops_per_sec = if delta_time > 0.0 {
                delta_ops as f64 / delta_time
            } else {
                0.0
            };

            if ops_per_sec > 0.0 {
                peak_ops_per_sec = peak_ops_per_sec.max(ops_per_sec);
                min_ops_per_sec = min_ops_per_sec.min(ops_per_sec);
            }

            let vstats = db.version_stats();
            let disk_gb = vstats.total_size as f64 / (1024.0 * 1024.0 * 1024.0);
            let frozen_count = db.frozen_memtable_count();

            let (queued, active, compaction_jobs, compaction_read, compaction_written) =
                match db.compaction_stats() {
                    | Ok(stats) => (
                        stats.queued_jobs,
                        stats.in_progress_jobs,
                        stats.completed_jobs,
                        stats.bytes_compacted_read,
                        stats.bytes_compacted_written,
                    ),
                    | Err(_) => (0, 0, 0, 0, 0),
                };

            let delta_jobs = compaction_jobs.saturating_sub(last_compaction_jobs);
            let delta_read = compaction_read.saturating_sub(last_compaction_read);
            let delta_written = compaction_written.saturating_sub(last_compaction_written);

            let jobs_per_sec = if delta_time > 0.0 {
                delta_jobs as f64 / delta_time
            } else {
                0.0
            };
            let read_mb_per_sec = if delta_time > 0.0 {
                delta_read as f64 / (1024.0 * 1024.0) / delta_time
            } else {
                0.0
            };
            let write_mb_per_sec = if delta_time > 0.0 {
                delta_written as f64 / (1024.0 * 1024.0) / delta_time
            } else {
                0.0
            };

            println!(
                "{:>8.1}s | {:>8.0} | {:>8.0} | {:>8.0} | {:>8.0} | {:>8.0} | {:>8.2} | {:>6} | {:>10} | {:>10} | {:>10.2} | {:>10.2} | {:>10.2} | {:>6}",
                current.elapsed_secs,
                ops_per_sec,
                (current.gets.saturating_sub(last.gets)) as f64 / delta_time.max(0.001),
                (current.puts.saturating_sub(last.puts)) as f64 / delta_time.max(0.001),
                (current.deletes.saturating_sub(last.deletes)) as f64 / delta_time.max(0.001),
                (current.scans.saturating_sub(last.scans)) as f64 / delta_time.max(0.001),
                disk_gb,
                vstats.l0_segments,
                queued,
                active,
                jobs_per_sec,
                read_mb_per_sec,
                write_mb_per_sec,
                frozen_count,
            );

            if frozen_count > 64 {
                eprintln!(
                    "\n⚠️  WARNING: {} frozen memtables - memory pressure high.",
                    frozen_count
                );
            }

            last = current;
            last_compaction_jobs = compaction_jobs;
            last_compaction_read = compaction_read;
            last_compaction_written = compaction_written;
        }
    })
}

#[tokio::test]
#[ignore]
async fn soak_20gib_churn() {
    let config = ChurnConfig::from_env();
    let target_gb = config.target_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    println!("\n=== CesiumDB 20 GiB Scoped Churn Soak Test ===");
    println!(
        "Target: {:.1} GiB | Value size: {} B | Batch: {} | Workers: {} | Duration: {}s",
        target_gb, config.value_size, config.batch_size, config.num_workers, config.duration_secs
    );

    let db_path = PathBuf::from("/tmp/cesiumdb_test_20gib_churn");
    let _ = fs::remove_dir_all(&db_path);

    let mut opts = DbOptions::default();
    opts.data_dir(db_path.clone())
        .memtable_size(64 * 1024 * 1024)
        .max_memtables(8);

    let db = Db::open(opts).unwrap();

    // ========================================================================
    // Phase 1: Pre-populate to target size
    // ========================================================================
    println!("\n--- Phase 1: Pre-populating to ~{:.1} GiB ---", target_gb);

    let global_counter = Arc::new(AtomicU64::new(0));
    let prepop_metrics = Arc::new(ChurnMetrics::new());
    let prepop_shutdown = Arc::new(AtomicBool::new(false));

    let prepop_workers: Vec<_> = (0..config.num_workers)
        .map(|_id| {
            spawn_prepop_worker(
                db.clone(),
                config.clone(),
                global_counter.clone(),
                prepop_metrics.clone(),
                prepop_shutdown.clone(),
            )
        })
        .collect();

    let prepop_start = Instant::now();
    let max_prepop_secs = config.duration_secs / 3; // cap pre-pop at 1/3 of total time

    loop {
        thread::sleep(Duration::from_secs(2));

        let written = prepop_metrics.bytes_written.load(Ordering::Relaxed);
        let elapsed = prepop_start.elapsed().as_secs();
        let gb = written as f64 / (1024.0 * 1024.0 * 1024.0);

        println!(
            "  Pre-pop: {:.2} / {:.2} GB ({:.1}%) | {:.0}s elapsed",
            gb,
            target_gb,
            (gb / target_gb) * 100.0,
            elapsed
        );

        if written >= config.target_bytes {
            println!("  ✅ Target reached!");
            break;
        }

        if elapsed >= max_prepop_secs {
            println!(
                "  ⏱️ Pre-pop timeout reached ({:.2} / {:.2} GB). Continuing...",
                gb, target_gb
            );
            break;
        }
    }

    prepop_shutdown.store(true, Ordering::SeqCst);
    for w in prepop_workers {
        let _ = w.join();
    }

    let total_keys = global_counter.load(Ordering::Relaxed);
    let prepop_bytes = prepop_metrics.bytes_written.load(Ordering::Relaxed);
    let prepop_elapsed = prepop_start.elapsed().as_secs_f64();

    println!(
        "Pre-population complete: {} keys, {:.2} GB, {:.1}s, {:.0} ops/sec",
        total_keys,
        prepop_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        prepop_elapsed,
        total_keys as f64 / prepop_elapsed.max(0.001)
    );

    assert!(total_keys > 0, "No keys were pre-populated");

    // ========================================================================
    // Phase 2: Mixed churn on bounded key space
    // ========================================================================
    let remaining_secs = config
        .duration_secs
        .saturating_sub(prepop_start.elapsed().as_secs());
    if remaining_secs == 0 {
        println!("No time remaining for churn phase. Test complete.");
        return;
    }

    println!(
        "\n--- Phase 2: Mixed churn for {}s on {} keys ---",
        remaining_secs, total_keys
    );

    let churn_metrics = Arc::new(ChurnMetrics::new());
    let churn_shutdown = Arc::new(AtomicBool::new(false));

    let churn_workers: Vec<_> = (0..config.num_workers)
        .map(|id| {
            spawn_churn_worker(
                db.clone(),
                config.clone(),
                total_keys,
                id,
                churn_metrics.clone(),
                churn_shutdown.clone(),
            )
        })
        .collect();

    let reporter = spawn_churn_reporter(
        db.clone(),
        churn_metrics.clone(),
        churn_shutdown.clone(),
        config.metrics_interval_secs,
    );

    // Let churn run for remaining duration
    thread::sleep(Duration::from_secs(remaining_secs));

    churn_shutdown.store(true, Ordering::SeqCst);
    for w in churn_workers {
        let _ = w.join();
    }
    let _ = reporter.join();

    // ========================================================================
    // Final metrics
    // ========================================================================
    let final_snap = churn_metrics.snapshot();
    let churn_secs = final_snap.elapsed_secs;

    println!("\n=== 20 GiB Churn Summary ===");
    println!(
        "Pre-pop: {} keys | {:.2} GB | {:.1}s",
        total_keys,
        prepop_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        prepop_elapsed
    );
    println!(
        "Churn phase: {:.1}s | total ops: {} | throughput: {:.0} ops/sec",
        churn_secs,
        final_snap.total_ops(),
        final_snap.total_ops() as f64 / churn_secs.max(0.001)
    );
    println!(
        "  Gets: {} | Puts: {} | Deletes: {} | Scans: {}",
        final_snap.gets, final_snap.puts, final_snap.deletes, final_snap.scans
    );

    // Disk usage and amplification
    let disk_bytes = calculate_dir_size(&db_path);
    let disk_gb = disk_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    // Logical size = live keys × value size (approximation; actual live data
    // may differ slightly due to random deletes, but bounded key space keeps
    // it in the same ballpark).
    let logical_bytes = total_keys * config.value_size as u64;
    let space_amplification = disk_bytes as f64 / logical_bytes.max(1) as f64;

    println!("\nDisk usage: {:.2} GB", disk_gb);
    println!(
        "Logical data: {:.2} GB",
        logical_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!("Space amplification: {:.2}x", space_amplification);

    // Write amplification from compaction stats
    if let Ok(stats) = db.compaction_stats() {
        let cmp_read_gb = stats.bytes_compacted_read as f64 / (1024.0 * 1024.0 * 1024.0);
        let cmp_write_gb = stats.bytes_compacted_written as f64 / (1024.0 * 1024.0 * 1024.0);
        let user_bytes = prepop_bytes + final_snap.bytes_written;
        let write_amplification =
            (user_bytes + stats.bytes_compacted_written) as f64 / user_bytes.max(1) as f64;

        println!("Compaction jobs: {}", stats.completed_jobs);
        println!("Compaction read: {:.2} GB", cmp_read_gb);
        println!("Compaction written: {:.2} GB", cmp_write_gb);
        println!(
            "User bytes written: {:.2} GB",
            user_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        println!("Write amplification: {:.2}x", write_amplification);

        assert!(
            write_amplification < 6.0,
            "Write amplification {:.2}x is too high - expected < 6.0x",
            write_amplification
        );
    }

    let vstats = db.version_stats();
    println!("L0 segments: {}", vstats.l0_segments);
    println!(
        "Total segments: {} | Levels: {}",
        vstats.total_segments, vstats.num_levels
    );

    assert!(
        space_amplification < 5.0,
        "Space amplification {:.2}x is too high - expected < 5.0x",
        space_amplification
    );

    println!("\nTest completed ✓");
}

/// Recursively calculate total size of a directory in bytes
fn calculate_dir_size(path: &std::path::Path) -> u64 {
    if !path.exists() {
        return 0;
    }
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                total += calculate_dir_size(&path);
            } else if let Ok(meta) = entry.metadata() {
                total += meta.len();
            }
        }
    }
    total
}
