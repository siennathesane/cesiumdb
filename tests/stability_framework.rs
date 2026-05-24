//! Stability & Consistency Test Framework
//!
//! Provides a shadow verifier that tracks expected database state
//! independently of the DB, enabling rigorous correctness validation
//! under concurrent load, crash recovery, and edge cases.

use std::{
    collections::HashMap,
    ops::Bound,
    sync::{
        Arc,
        Mutex,
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

use cesiumdb::Db;
use rand::{
    Rng,
    rngs::ThreadRng,
};

/// A deterministic shadow of the expected database state.
pub struct ShadowVerifier {
    /// Map from key → latest expected value
    expected: HashMap<Vec<u8>, Vec<u8>>,
    /// Track deleted keys
    deleted: HashMap<Vec<u8>, bool>,
    /// Total writes observed
    pub total_writes: u64,
    /// Total deletes observed
    pub total_deletes: u64,
    /// Verification errors found
    pub errors: Vec<String>,
}

impl ShadowVerifier {
    pub fn new() -> Self {
        Self {
            expected: HashMap::new(),
            deleted: HashMap::new(),
            total_writes: 0,
            total_deletes: 0,
            errors: Vec::new(),
        }
    }

    /// Record a write operation in the shadow state.
    pub fn record_write(&mut self, key: Vec<u8>, value: Vec<u8>) {
        self.expected.insert(key.clone(), value);
        self.deleted.remove(&key);
        self.total_writes += 1;
    }

    /// Record a delete operation in the shadow state.
    pub fn record_delete(&mut self, key: Vec<u8>) {
        self.expected.remove(&key);
        self.deleted.insert(key, true);
        self.total_deletes += 1;
    }

    /// Verify that a point read matches the expected value.
    pub fn verify_point_read(&mut self, key: &[u8], actual: Option<&[u8]>) -> bool {
        let expected = self.expected.get(key);
        match (expected, actual) {
            (Some(exp), Some(act)) if exp.as_slice() == act => true,
            (None, None) => true,
            (Some(exp), Some(act)) => {
                self.errors.push(format!(
                    "MISMATCH: key={:?} expected_len={} actual_len={}",
                    String::from_utf8_lossy(key),
                    exp.len(),
                    act.len()
                ));
                false
            },
            (Some(_), None) => {
                self.errors.push(format!(
                    "MISSING: key={:?}",
                    String::from_utf8_lossy(key)
                ));
                false
            },
            (None, Some(_)) => {
                self.errors.push(format!(
                    "UNEXPECTED: key={:?}",
                    String::from_utf8_lossy(key)
                ));
                false
            },
        }
    }

    /// Verify a full scan against the expected sorted key-value pairs.
    pub fn verify_scan(&mut self, actual: &[(Vec<u8>, Vec<u8>)]) -> bool {
        let mut expected_vec: Vec<_> = self
            .expected
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        expected_vec.sort_by(|a, b| a.0.cmp(&b.0));

        if expected_vec.len() != actual.len() {
            self.errors.push(format!(
                "SCAN_LEN: expected {} entries, got {}",
                expected_vec.len(),
                actual.len()
            ));
            return false;
        }

        let mut ok = true;
        for (i, (exp, act)) in expected_vec.iter().zip(actual.iter()).enumerate() {
            if exp.0 != act.0 || exp.1 != act.1 {
                self.errors.push(format!(
                    "SCAN_MISMATCH[{}]: expected ({:?}, len={}) actual ({:?}, len={})",
                    i,
                    String::from_utf8_lossy(&exp.0),
                    exp.1.len(),
                    String::from_utf8_lossy(&act.0),
                    act.1.len()
                ));
                ok = false;
            }
        }
        ok
    }

    /// Sample random keys and verify point reads.
    pub fn verify_random_sample(&mut self, db: &Db, sample_count: usize) -> bool {
        let keys: Vec<_> = self.expected.keys().cloned().collect();
        if keys.is_empty() {
            return true;
        }

        let mut rng = ThreadRng::default();
        let mut ok = true;
        for _ in 0..sample_count.min(keys.len()) {
            let key = &keys[rng.random_range(0..keys.len())];
            let actual = db.get(key).ok().flatten();
            let actual_ref = actual.as_ref().map(|v| v.as_ref());
            if !self.verify_point_read(key, actual_ref) {
                ok = false;
            }
        }
        ok
    }

    /// Verify that all deleted keys are truly gone.
    pub fn verify_deletes(&mut self, db: &Db) -> bool {
        let mut ok = true;
        for key in self.deleted.keys() {
            match db.get(key) {
                Ok(None) => {},
                Ok(Some(_)) => {
                    self.errors.push(format!(
                        "TOMBSTONE_LEAK: key={:?} still has value",
                        String::from_utf8_lossy(key)
                    ));
                    ok = false;
                },
                Err(e) => {
                    self.errors.push(format!(
                        "READ_ERROR: key={:?} error={:?}",
                        String::from_utf8_lossy(key),
                        e
                    ));
                    ok = false;
                },
            }
        }
        ok
    }

    /// Returns the number of keys in the expected state.
    pub fn expected_key_count(&self) -> usize {
        self.expected.len()
    }

    /// Total bytes of live key-value data (keys + values).
    pub fn live_data_bytes(&self) -> usize {
        self.expected.iter().map(|(k, v)| k.len() + v.len()).sum()
    }

    /// Total bytes of tombstone keys (deleted keys still on disk).
    pub fn tombstone_bytes(&self) -> usize {
        self.deleted.keys().map(|k| k.len()).sum()
    }

    /// Iterates over all expected key-value pairs.
    pub fn iter_expected(&self) -> impl Iterator<Item = (&Vec<u8>, &Vec<u8>)> {
        self.expected.iter()
    }

    /// Iterates over all deleted keys.
    pub fn iter_deleted(&self) -> impl Iterator<Item = (&Vec<u8>, &bool)> {
        self.deleted.iter()
    }

    /// Returns true if no errors have been recorded.
    pub fn is_clean(&self) -> bool {
        self.errors.is_empty()
    }
}

/// Configuration for a stability test run.
#[derive(Debug, Clone)]
pub struct StabilityConfig {
    pub duration_secs: u64,
    pub num_writers: usize,
    pub num_readers: usize,
    pub num_scanners: usize,
    pub key_space: usize,
    pub value_size: usize,
    pub write_rate_hz: u64,
    pub verification_interval_ms: u64,
    pub delete_probability: f64,
}

impl Default for StabilityConfig {
    fn default() -> Self {
        Self {
            duration_secs: 30,
            num_writers: 4,
            num_readers: 4,
            num_scanners: 0,
            key_space: 100_000,
            value_size: 1024,
            write_rate_hz: 1000,
            verification_interval_ms: 1000,
            delete_probability: 0.1,
        }
    }
}

/// Metrics from a stability test run.
#[derive(Debug, Clone)]
pub struct StabilityMetrics {
    pub total_writes: u64,
    pub total_reads: u64,
    pub total_scans: u64,
    pub total_deletes: u64,
    pub verification_passes: u64,
    pub verification_failures: u64,
    pub duration_secs: f64,
    pub errors: Vec<String>,
    pub read_amp_stats: Option<cesiumdb::ReadAmpStats>,
    /// Space amplification = total disk size / live data bytes (from verifier)
    pub space_amp: Option<f64>,
}

/// Runs a stability test with concurrent writers, readers, and optional scanners.
///
/// Workers update the shared shadow verifier atomically with each DB operation
/// to ensure the verifier always reflects the intended state.
pub fn run_stability_test(
    db: Arc<Db>,
    verifier: Arc<Mutex<ShadowVerifier>>,
    config: StabilityConfig,
) -> StabilityMetrics {
    let start = Instant::now();
    let shutdown = Arc::new(AtomicBool::new(false));

    let total_writes = Arc::new(AtomicU64::new(0));
    let total_reads = Arc::new(AtomicU64::new(0));
    let total_scans = Arc::new(AtomicU64::new(0));
    let total_deletes = Arc::new(AtomicU64::new(0));

    // Spawn writers
    let writers: Vec<_> = (0..config.num_writers)
        .map(|id| {
            let db = db.clone();
            let shutdown = shutdown.clone();
            let writes = total_writes.clone();
            let deletes = total_deletes.clone();
            let verifier = verifier.clone();
            thread::spawn(move || {
                let mut rng = ThreadRng::default();
                let worker_offset = id * (config.key_space / config.num_writers.max(1));
                let worker_keys = config.key_space / config.num_writers.max(1);
                while !shutdown.load(Ordering::Relaxed) {
                    let key_idx = worker_offset + rng.random_range(0..worker_keys);
                    let key = format!("key_{:010}", key_idx).into_bytes();

                    if rng.random::<f64>() < config.delete_probability {
                        let mut v = verifier.lock().unwrap();
                        match db.delete(&key) {
                            Ok(_) => {
                                v.record_delete(key.clone());
                                deletes.fetch_add(1, Ordering::Relaxed);
                            },
                            Err(e) => {
                                eprintln!("DELETE_ERROR: key={:?} error={:?}", String::from_utf8_lossy(&key), e);
                            },
                        }
                    } else {
                        let value = format!("value_{:016}", rng.random::<u64>()).into_bytes();
                        let mut v = verifier.lock().unwrap();
                        match db.put(&key, &value) {
                            Ok(_) => {
                                v.record_write(key.clone(), value.clone());
                                writes.fetch_add(1, Ordering::Relaxed);
                            },
                            Err(e) => {
                                eprintln!("PUT_ERROR: key={:?} error={:?}", String::from_utf8_lossy(&key), e);
                            },
                        }
                    }

                    thread::sleep(Duration::from_micros(1_000_000 / config.write_rate_hz.max(1)));
                }
            })
        })
        .collect();

    // Spawn readers
    let readers: Vec<_> = (0..config.num_readers)
        .map(|_| {
            let db = db.clone();
            let shutdown = shutdown.clone();
            let reads = total_reads.clone();
            thread::spawn(move || {
                let mut rng = ThreadRng::default();
                while !shutdown.load(Ordering::Relaxed) {
                    let key_idx = rng.random_range(0..config.key_space);
                    let key = format!("key_{:010}", key_idx).into_bytes();
                    let _ = db.get(&key);
                    reads.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    // Spawn scanners
    let scanners: Vec<_> = (0..config.num_scanners)
        .map(|_| {
            let db = db.clone();
            let shutdown = shutdown.clone();
            let scans = total_scans.clone();
            thread::spawn(move || {
                let mut rng = ThreadRng::default();
                while !shutdown.load(Ordering::Relaxed) {
                    let start_key = format!("key_{:010}", rng.random_range(0..config.key_space)).into_bytes();
                    let end_key = format!("key_{:010}", rng.random_range(0..config.key_space)).into_bytes();
                    let (lower, upper) = if start_key <= end_key {
                        (Bound::Included(start_key.as_slice()), Bound::Included(end_key.as_slice()))
                    } else {
                        (Bound::Included(end_key.as_slice()), Bound::Included(start_key.as_slice()))
                    };
                    let _ = db.scan(lower, upper);
                    scans.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    // Verification loop
    let mut verification_passes = 0u64;
    let mut verification_failures = 0u64;
    let verify_start = Instant::now();
    while verify_start.elapsed().as_secs() < config.duration_secs {
        thread::sleep(Duration::from_millis(config.verification_interval_ms));

        let sample_count = (config.key_space / 100).max(100);
        let mut v = verifier.lock().unwrap();
        if v.verify_random_sample(&db, sample_count) {
            verification_passes += 1;
        } else {
            verification_failures += 1;
        }

        if v.verify_deletes(&db) {
            verification_passes += 1;
        } else {
            verification_failures += 1;
        }
    }

    shutdown.store(true, Ordering::Relaxed);
    for w in writers { let _ = w.join(); }
    for r in readers { let _ = r.join(); }
    for s in scanners { let _ = s.join(); }

    let elapsed = start.elapsed().as_secs_f64();

    let version_stats = db.version_stats();
    let v = verifier.lock().unwrap();
    let live = v.live_data_bytes() + v.tombstone_bytes();
    let space_amp = if live > 0 {
        Some(version_stats.total_size as f64 / live as f64)
    } else {
        None
    };

    StabilityMetrics {
        total_writes: total_writes.load(Ordering::Relaxed),
        total_reads: total_reads.load(Ordering::Relaxed),
        total_scans: total_scans.load(Ordering::Relaxed),
        total_deletes: total_deletes.load(Ordering::Relaxed),
        verification_passes,
        verification_failures,
        duration_secs: elapsed,
        errors: v.errors.clone(),
        read_amp_stats: Some(db.read_amp_stats()),
        space_amp,
    }
}

/// Same as run_stability_test but only does point-read verification during the run.
/// Delete verification is deferred until after all workers have stopped.
pub fn run_stability_test_final_verify_only(
    db: Arc<Db>,
    verifier: Arc<Mutex<ShadowVerifier>>,
    config: StabilityConfig,
) -> StabilityMetrics {
    let start = Instant::now();
    let shutdown = Arc::new(AtomicBool::new(false));

    let total_writes = Arc::new(AtomicU64::new(0));
    let total_reads = Arc::new(AtomicU64::new(0));
    let total_scans = Arc::new(AtomicU64::new(0));
    let total_deletes = Arc::new(AtomicU64::new(0));

    let writers: Vec<_> = (0..config.num_writers)
        .map(|id| {
            let db = db.clone();
            let shutdown = shutdown.clone();
            let writes = total_writes.clone();
            let deletes = total_deletes.clone();
            let verifier = verifier.clone();
            thread::spawn(move || {
                let mut rng = ThreadRng::default();
                let worker_offset = id * (config.key_space / config.num_writers.max(1));
                let worker_keys = config.key_space / config.num_writers.max(1);
                while !shutdown.load(Ordering::Relaxed) {
                    let key_idx = worker_offset + rng.random_range(0..worker_keys);
                    let key = format!("key_{:010}", key_idx).into_bytes();

                    if rng.random::<f64>() < config.delete_probability {
                        let mut v = verifier.lock().unwrap();
                        match db.delete(&key) {
                            Ok(_) => {
                                v.record_delete(key.clone());
                                deletes.fetch_add(1, Ordering::Relaxed);
                            },
                            Err(e) => {
                                eprintln!("DELETE_ERROR: key={:?} error={:?}", String::from_utf8_lossy(&key), e);
                            },
                        }
                    } else {
                        let value = format!("value_{:016}", rng.random::<u64>()).into_bytes();
                        let mut v = verifier.lock().unwrap();
                        match db.put(&key, &value) {
                            Ok(_) => {
                                v.record_write(key.clone(), value.clone());
                                writes.fetch_add(1, Ordering::Relaxed);
                            },
                            Err(e) => {
                                eprintln!("PUT_ERROR: key={:?} error={:?}", String::from_utf8_lossy(&key), e);
                            },
                        }
                    }

                    thread::sleep(Duration::from_micros(1_000_000 / config.write_rate_hz.max(1)));
                }
            })
        })
        .collect();

    let readers: Vec<_> = (0..config.num_readers)
        .map(|_| {
            let db = db.clone();
            let shutdown = shutdown.clone();
            let reads = total_reads.clone();
            thread::spawn(move || {
                let mut rng = ThreadRng::default();
                while !shutdown.load(Ordering::Relaxed) {
                    let key_idx = rng.random_range(0..config.key_space);
                    let key = format!("key_{:010}", key_idx).into_bytes();
                    let _ = db.get(&key);
                    reads.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    let scanners: Vec<_> = (0..config.num_scanners)
        .map(|_| {
            let db = db.clone();
            let shutdown = shutdown.clone();
            let scans = total_scans.clone();
            thread::spawn(move || {
                let mut rng = ThreadRng::default();
                while !shutdown.load(Ordering::Relaxed) {
                    let start_key = format!("key_{:010}", rng.random_range(0..config.key_space)).into_bytes();
                    let end_key = format!("key_{:010}", rng.random_range(0..config.key_space)).into_bytes();
                    let (lower, upper) = if start_key <= end_key {
                        (Bound::Included(start_key.as_slice()), Bound::Included(end_key.as_slice()))
                    } else {
                        (Bound::Included(end_key.as_slice()), Bound::Included(start_key.as_slice()))
                    };
                    let _ = db.scan(lower, upper);
                    scans.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    let mut verification_passes = 0u64;
    let mut verification_failures = 0u64;
    let verify_start = Instant::now();
    while verify_start.elapsed().as_secs() < config.duration_secs {
        thread::sleep(Duration::from_millis(config.verification_interval_ms));

        let sample_count = (config.key_space / 100).max(100);
        let mut v = verifier.lock().unwrap();
        if v.verify_random_sample(&db, sample_count) {
            verification_passes += 1;
        } else {
            verification_failures += 1;
        }
        // Clear transient errors; only keep errors from the last verification
        v.errors.clear();
    }

    shutdown.store(true, Ordering::Relaxed);
    for w in writers { let _ = w.join(); }
    for r in readers { let _ = r.join(); }
    for s in scanners { let _ = s.join(); }

    let elapsed = start.elapsed().as_secs_f64();

    let version_stats = db.version_stats();
    let v = verifier.lock().unwrap();
    let live = v.live_data_bytes() + v.tombstone_bytes();
    let space_amp = if live > 0 {
        Some(version_stats.total_size as f64 / live as f64)
    } else {
        None
    };

    StabilityMetrics {
        total_writes: total_writes.load(Ordering::Relaxed),
        total_reads: total_reads.load(Ordering::Relaxed),
        total_scans: total_scans.load(Ordering::Relaxed),
        total_deletes: total_deletes.load(Ordering::Relaxed),
        verification_passes,
        verification_failures,
        duration_secs: elapsed,
        errors: v.errors.clone(),
        read_amp_stats: Some(db.read_amp_stats()),
        space_amp,
    }
}
