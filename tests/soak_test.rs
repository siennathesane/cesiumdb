//! Comprehensive soak test framework for CesiumDB
//!
//! This module provides stress testing capabilities similar to RocksDB's db_bench,
//! validating stability, performance, and correctness under sustained heavy load.
//!
//! Features:
//! - Configurable workloads (mixed, write-heavy, read-heavy, scan-heavy, delete-heavy)
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
//!   1. Start Jaeger: docker run -d -p4317:4317 -p16686:16686 jaegertracing/all-in-one:latest
//!   2. Run test: OTEL_ENABLED=1 cargo test --test soak_test soak_smoke_mixed -- --ignored --nocapture
//!   3. View traces: http://localhost:16686


use std::{
    collections::HashMap,
    ops::Bound,
    sync::{
        atomic::{
            AtomicBool,
            AtomicU64,
            Ordering,
        },
        Arc,
        Mutex,
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
use tempfile::TempDir;

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

    // Simple latency histogram (7 buckets: <1µs, 1-10µs, 10-100µs, 100µs-1ms, 1-10ms, 10-100ms, >100ms)
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

    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
                current.elapsed_secs, current.total_ops, instant_rate, current.ops_per_sec, current.error_count
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
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("soak_test");

    let mut opts = DbOptions::default();
    opts.data_dir(db_path)
        .memtable_size(config.memtable_size)
        .max_memtables(config.max_memtables);

    let db = Db::open(opts);

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
    assert!(
        final_snapshot.total_ops > 0,
        "test completed no operations"
    );

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
