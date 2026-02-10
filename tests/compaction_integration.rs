//! Large-scale integration tests for the compaction system
//!
//! These tests validate the entire compaction pipeline with realistic data
//! volumes. Run with: cargo test --test compaction_integration -- --ignored
//! --test-threads=1

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

use cesiumdb::{
    Batch::*,
    Db,
    DbOptions,
};
use tempfile::TempDir;

/// Helper to generate deterministic test data
fn generate_key(ns: &str, i: u64) -> Vec<u8> {
    format!("{}-key-{:010}", ns, i).into_bytes()
}

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

#[test]
#[ignore] // Run with: cargo test -- --ignored
fn test_10gb_write_heavy_workload() {
    println!("\n=== Testing 10GB Write-Heavy Workload ===");

    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("10gb_write_heavy");

    let mut opts = DbOptions::default();
    opts.data_dir(db_path.clone())
        .memtable_size(64 * 1024 * 1024) // 64MB memtables
        .max_memtables(4);

    let db = Db::open(opts);

    // Target: 10GB of data
    // Strategy: Write 1KB values, need ~10 million entries
    const VALUE_SIZE: usize = 1024;
    const TARGET_ENTRIES: u64 = 10_000_000; // 10M * 1KB = ~10GB
    const BATCH_SIZE: usize = 1000; // Batch 1000 writes at a time
    const NAMESPACE: &str = "write-heavy";

    let start = Instant::now();
    let written = Arc::new(AtomicU64::new(0));
    let next_idx = Arc::new(AtomicU64::new(0));

    // Spawn multiple writer threads (one per CPU core)
    let num_writers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);

    println!(
        "Writing {} entries of {}KB each (batches of {}) using {} threads...",
        TARGET_ENTRIES,
        VALUE_SIZE / 1024,
        BATCH_SIZE,
        num_writers
    );

    let handles: Vec<_> = (0..num_writers)
        .map(|_thread_id| {
            let db = db.clone();
            let written = Arc::clone(&written);
            let next_idx = Arc::clone(&next_idx);

            thread::spawn(move || {
                let mut local_written = 0u64;

                loop {
                    // Claim next batch range
                    let start_idx = next_idx.fetch_add(BATCH_SIZE as u64, Ordering::Relaxed);
                    if start_idx >= TARGET_ENTRIES {
                        break;
                    }

                    let mut batch = Vec::with_capacity(BATCH_SIZE);

                    // Build batch
                    for j in 0..BATCH_SIZE {
                        let idx = start_idx + j as u64;
                        if idx >= TARGET_ENTRIES {
                            break;
                        }
                        let key = generate_key(NAMESPACE, idx);
                        let value = generate_value(VALUE_SIZE, idx);
                        batch.push(Put(key, value, db.time()));
                    }

                    // Write batch
                    db.batch(&batch).expect("failed to batch write");
                    local_written += batch.len() as u64;
                    written.fetch_add(batch.len() as u64, Ordering::Relaxed);
                }

                local_written
            })
        })
        .collect();

    // Progress reporter thread
    let written_clone = Arc::clone(&written);
    let reporter = thread::spawn(move || {
        let mut last_written = 0u64;
        let mut last_report = Instant::now();

        while last_written < TARGET_ENTRIES {
            thread::sleep(Duration::from_millis(500));
            let current = written_clone.load(Ordering::Relaxed);

            if current - last_written >= 100_000 {
                let elapsed = last_report.elapsed();
                let delta = current - last_written;
                let rate = delta as f64 / elapsed.as_secs_f64();

                println!(
                    "Progress: {}/{} ({:.1}%) - {:.0} writes/sec",
                    current,
                    TARGET_ENTRIES,
                    (current as f64 / TARGET_ENTRIES as f64) * 100.0,
                    rate
                );

                last_written = current;
                last_report = Instant::now();
            }
        }
    });

    // Wait for all writers to complete
    for handle in handles {
        handle.join().unwrap();
    }

    let total_written = written.load(Ordering::Relaxed);
    reporter.join().unwrap();

    let write_duration = start.elapsed();
    println!(
        "\nWrite phase complete: {} entries in {:.2}s ({:.0} writes/sec)",
        total_written,
        write_duration.as_secs_f64(),
        total_written as f64 / write_duration.as_secs_f64()
    );

    // Wait for background flusher to catch up
    println!("\nWaiting for background flush to complete...");
    thread::sleep(Duration::from_secs(5));

    // Trigger final compaction
    println!("\nRunning final compaction...");
    let compact_start = Instant::now();
    db.compact().expect("final compaction failed");
    println!(
        "Final compaction took {:.2}s",
        compact_start.elapsed().as_secs_f64()
    );

    // Wait a bit more for compaction to complete
    thread::sleep(Duration::from_secs(2));

    // Verification phase: random read sampling
    println!("\nVerifying data integrity (sampling 10k random entries)...");
    let verify_start = Instant::now();
    let sample_size = 10_000;

    for i in 0..sample_size {
        let idx = (i * (TARGET_ENTRIES / sample_size)) % TARGET_ENTRIES;
        let key = generate_key(NAMESPACE, idx);

        let result = db.get(&key).expect("failed to get");
        if result.is_none() {
            eprintln!(
                "Missing key at index {}, key: {:?}",
                idx,
                String::from_utf8_lossy(&key)
            );
            eprintln!("Sample index i={}, total checked so far: {}", i, i);
        }
        assert!(result.is_some(), "missing key at index {}", idx);

        let value = result.unwrap();
        assert!(
            verify_value(&value, VALUE_SIZE, idx),
            "value corruption at index {}",
            idx
        );
    }

    println!(
        "Verification complete: {} samples in {:.2}s",
        sample_size,
        verify_start.elapsed().as_secs_f64()
    );

    // Report final stats
    if let Ok(stats) = db.compaction_stats() {
        println!("\nFinal Compaction Stats:");
        println!("{}", stats);
    }

    let total_duration = start.elapsed();
    println!(
        "\n=== Test Complete ===\nTotal time: {:.2}s\nData written: ~{:.2}GB",
        total_duration.as_secs_f64(),
        (total_written * VALUE_SIZE as u64) as f64 / (1024.0 * 1024.0 * 1024.0)
    );
}

#[test]
#[ignore]
fn test_concurrent_reads_during_compaction() {
    println!("\n=== Testing Concurrent Reads During Compaction ===");

    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("concurrent_reads");

    let mut opts = DbOptions::default();
    opts.data_dir(db_path.clone())
        .memtable_size(32 * 1024 * 1024);

    let db = Db::open(opts);

    // Phase 1: Populate database with 1GB of data
    const VALUE_SIZE: usize = 1024;
    const NUM_ENTRIES: u64 = 1_000_000; // 1M * 1KB = ~1GB
    const NAMESPACE: &str = "concurrent";

    println!("Phase 1: Writing {} entries...", NUM_ENTRIES);
    let start = Instant::now();

    const BATCH_SIZE: usize = 1000;
    let mut i = 0u64;
    while i < NUM_ENTRIES {
        let mut batch = Vec::with_capacity(BATCH_SIZE);

        for j in 0..BATCH_SIZE {
            if i + j as u64 >= NUM_ENTRIES {
                break;
            }
            let idx = i + j as u64;
            let key = generate_key(NAMESPACE, idx);
            let value = generate_value(VALUE_SIZE, idx);
            batch.push(Put(key, value, db.time()));
        }

        db.batch(&batch).expect("failed to batch write");
        i += batch.len() as u64;

        if i % 100_000 == 0 && i > 0 {
            println!("  Written {}/{}", i, NUM_ENTRIES);
        }
    }

    println!("Write phase took {:.2}s", start.elapsed().as_secs_f64());

    // Phase 2: Concurrent reads while compacting
    println!("\nPhase 2: Starting concurrent reads and compaction...");

    let shutdown = Arc::new(AtomicBool::new(false));
    let read_count = Arc::new(AtomicU64::new(0));
    let error_count = Arc::new(AtomicU64::new(0));

    // Spawn reader threads
    let mut handles = vec![];
    for thread_id in 0..4 {
        let db = Arc::clone(&db);
        let shutdown = Arc::clone(&shutdown);
        let read_count = Arc::clone(&read_count);
        let error_count = Arc::clone(&error_count);

        let handle = thread::spawn(move || {
            let mut local_reads = 0u64;
            while !shutdown.load(Ordering::Relaxed) {
                // Random read
                let idx = (thread_id as u64 * 1000 + local_reads) % NUM_ENTRIES;
                let key = generate_key(NAMESPACE, idx);

                match db.get(&key) {
                    | Ok(Some(value)) => {
                        // Verify value integrity
                        if !verify_value(&value, VALUE_SIZE, idx) {
                            error_count.fetch_add(1, Ordering::Relaxed);
                        }
                        local_reads += 1;
                    },
                    | Ok(None) => {
                        error_count.fetch_add(1, Ordering::Relaxed);
                    },
                    | Err(_) => {
                        error_count.fetch_add(1, Ordering::Relaxed);
                    },
                }
            }
            read_count.fetch_add(local_reads, Ordering::Relaxed);
        });

        handles.push(handle);
    }

    // Run multiple compactions while readers are active
    println!("Running compactions while reading...");
    let compact_start = Instant::now();
    for i in 0..5 {
        println!("  Compaction round {}/5", i + 1);
        db.compact().expect("compaction failed");
        thread::sleep(Duration::from_millis(500)); // Brief pause between compactions
    }
    let compact_duration = compact_start.elapsed();

    // Stop readers
    shutdown.store(true, Ordering::Relaxed);
    for handle in handles {
        handle.join().unwrap();
    }

    let total_reads = read_count.load(Ordering::Relaxed);
    let total_errors = error_count.load(Ordering::Relaxed);

    println!("\n=== Results ===");
    println!("Compaction time: {:.2}s", compact_duration.as_secs_f64());
    println!("Total reads during compaction: {}", total_reads);
    println!("Read errors: {}", total_errors);
    println!(
        "Read throughput: {:.0} reads/sec",
        total_reads as f64 / compact_duration.as_secs_f64()
    );

    assert_eq!(
        total_errors, 0,
        "should have no read errors during compaction"
    );
    assert!(total_reads > 0, "should have completed some reads");
}

#[test]
#[ignore]
fn test_mixed_workload_with_updates() {
    println!("\n=== Testing Mixed Workload with Updates ===");

    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("mixed_workload");

    let mut opts = DbOptions::default();
    opts.data_dir(db_path.clone())
        .memtable_size(32 * 1024 * 1024);

    let db = Db::open(opts);

    const VALUE_SIZE: usize = 512;
    const NUM_KEYS: u64 = 1_000_000; // 1M keys
    const NAMESPACE: &str = "mixed";

    println!("Phase 1: Initial data load ({} keys)...", NUM_KEYS);
    let start = Instant::now();

    // Initial write
    const BATCH_SIZE: usize = 1000;
    let mut i = 0u64;
    while i < NUM_KEYS {
        let mut batch = Vec::with_capacity(BATCH_SIZE);

        for j in 0..BATCH_SIZE {
            if i + j as u64 >= NUM_KEYS {
                break;
            }
            let idx = i + j as u64;
            let key = generate_key(NAMESPACE, idx);
            let value = generate_value(VALUE_SIZE, idx);
            batch.push(Put(key, value, db.time()));
        }

        db.batch(&batch).expect("failed to batch write");
        i += batch.len() as u64;

        if i % 100_000 == 0 && i > 0 {
            println!("  Written {}/{}", i, NUM_KEYS);
        }
    }

    println!("Initial load took {:.2}s", start.elapsed().as_secs_f64());

    // Phase 2: Mixed updates (overwrites) and reads
    println!("\nPhase 2: Mixed updates and reads...");
    let mixed_start = Instant::now();

    let update_count = Arc::new(AtomicU64::new(0));
    let read_count = Arc::new(AtomicU64::new(0));
    let shutdown = Arc::new(AtomicBool::new(false));

    // Writer thread (updates existing keys)
    let db_writer = Arc::clone(&db);
    let update_count_clone = Arc::clone(&update_count);
    let writer_handle = thread::spawn(move || {
        const WRITE_BATCH_SIZE: usize = 500;
        let mut updates = 0u64;

        while updates < 500_000 {
            // 500k updates
            let mut batch = Vec::with_capacity(WRITE_BATCH_SIZE);

            for j in 0..WRITE_BATCH_SIZE {
                if updates + j as u64 >= 500_000 {
                    break;
                }
                let idx = (updates + j as u64) % NUM_KEYS;
                let key = generate_key(NAMESPACE, idx);
                // New value with different seed
                let value = generate_value(VALUE_SIZE, idx + 1_000_000);
                batch.push(Put(key, value, db_writer.time()));
            }

            db_writer.batch(&batch).expect("failed to batch write");
            updates += batch.len() as u64;

            if updates % 50_000 == 0 {
                println!("  Updates: {}/500k", updates);
            }

            // Compact periodically
            if updates % 100_000 == 0 {
                db_writer.compact().expect("compaction failed");
            }
        }
        update_count_clone.store(updates, Ordering::Relaxed);
    });

    // Reader threads
    let mut reader_handles = vec![];
    for _ in 0..2 {
        let db = Arc::clone(&db);
        let read_count = Arc::clone(&read_count);
        let shutdown = Arc::clone(&shutdown);

        let handle = thread::spawn(move || {
            let mut reads = 0u64;
            while !shutdown.load(Ordering::Relaxed) {
                let idx = reads % NUM_KEYS;
                let key = generate_key(NAMESPACE, idx);

                if let Ok(Some(_)) = db.get(&key) {
                    reads += 1;
                }

                // Small delay to avoid tight spinning
                if reads % 1000 == 0 {
                    thread::sleep(Duration::from_micros(100));
                }
            }
            read_count.fetch_add(reads, Ordering::Relaxed);
        });

        reader_handles.push(handle);
    }

    // Wait for writer
    writer_handle.join().unwrap();

    // Stop readers
    shutdown.store(true, Ordering::Relaxed);
    for handle in reader_handles {
        handle.join().unwrap();
    }

    let mixed_duration = mixed_start.elapsed();
    let total_updates = update_count.load(Ordering::Relaxed);
    let total_reads = read_count.load(Ordering::Relaxed);

    println!("\n=== Results ===");
    println!(
        "Mixed workload duration: {:.2}s",
        mixed_duration.as_secs_f64()
    );
    println!("Total updates: {}", total_updates);
    println!("Total reads: {}", total_reads);
    println!(
        "Update throughput: {:.0} updates/sec",
        total_updates as f64 / mixed_duration.as_secs_f64()
    );
    println!(
        "Read throughput: {:.0} reads/sec",
        total_reads as f64 / mixed_duration.as_secs_f64()
    );

    // Final compaction
    println!("\nRunning final compaction...");
    db.compact().expect("final compaction failed");

    // Verify some updated values
    println!("Verifying updates...");
    for i in (0..10_000).step_by(100) {
        let key = generate_key(NAMESPACE, i);
        let result = db.get(&key).expect("failed to get");
        assert!(result.is_some(), "key {} should exist", i);

        // Value should have the updated seed
        let value = result.unwrap();
        assert!(
            verify_value(&value, VALUE_SIZE, i + 1_000_000),
            "value at {} should be updated",
            i
        );
    }

    println!("Verification complete!");
}

#[test]
#[ignore]
fn test_deletion_with_compaction() {
    println!("\n=== Testing Deletion with Compaction ===");

    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("deletion");

    let mut opts = DbOptions::default();
    opts.data_dir(db_path.clone())
        .memtable_size(16 * 1024 * 1024);

    let db = Db::open(opts);

    const VALUE_SIZE: usize = 1024;
    const NUM_ENTRIES: u64 = 500_000; // 500k entries
    const NAMESPACE: &str = "deletion";

    println!("Phase 1: Writing {} entries...", NUM_ENTRIES);
    const BATCH_SIZE: usize = 1000;
    let mut i = 0u64;
    while i < NUM_ENTRIES {
        let mut batch = Vec::with_capacity(BATCH_SIZE);

        for j in 0..BATCH_SIZE {
            if i + j as u64 >= NUM_ENTRIES {
                break;
            }
            let idx = i + j as u64;
            let key = generate_key(NAMESPACE, idx);
            let value = generate_value(VALUE_SIZE, idx);
            batch.push(Put(key, value, db.time()));
        }

        db.batch(&batch).expect("failed to batch write");
        i += batch.len() as u64;

        if i % 100_000 == 0 && i > 0 {
            println!("  Written {}/{}", i, NUM_ENTRIES);
        }
    }

    println!("\nPhase 2: Deleting 50% of entries...");
    let mut deleted_keys = vec![];
    let delete_indices: Vec<u64> = (0..NUM_ENTRIES).step_by(2).collect();

    let mut i = 0;
    while i < delete_indices.len() {
        let mut batch: Vec<cesiumdb::Batch<Vec<u8>, Vec<u8>>> = Vec::with_capacity(BATCH_SIZE);

        for j in 0..BATCH_SIZE {
            if i + j >= delete_indices.len() {
                break;
            }
            let idx = delete_indices[i + j];
            let key = generate_key(NAMESPACE, idx);
            batch.push(Delete(key, db.time()));
        }

        db.batch(&batch).expect("failed to batch delete");
        i += batch.len();

        if i % 100_000 == 0 && i > 0 {
            println!("  Deleted {}/{}", i, delete_indices.len());
        }
    }

    deleted_keys = delete_indices;

    println!("\nPhase 3: Compacting to reclaim space...");
    let compact_start = Instant::now();
    db.compact().expect("compaction failed");
    println!(
        "Compaction took {:.2}s",
        compact_start.elapsed().as_secs_f64()
    );

    println!("\nPhase 4: Verifying deletions...");
    // Verify deleted keys are gone
    for &i in deleted_keys.iter().take(1000) {
        let key = generate_key(NAMESPACE, i);
        let result = db.get(&key).expect("failed to get");
        assert!(result.is_none(), "key {} should be deleted", i);
    }

    // Verify remaining keys still exist
    for i in (1..NUM_ENTRIES).step_by(2).take(1000) {
        let key = generate_key(NAMESPACE, i);
        let result = db.get(&key).expect("failed to get");
        assert!(result.is_some(), "key {} should exist", i);

        let value = result.unwrap();
        assert!(
            verify_value(&value, VALUE_SIZE, i),
            "value corruption at key {}",
            i
        );
    }

    println!("Verification complete!");
}

#[test]
#[ignore]
fn test_background_compaction_triggers() {
    println!("\n=== Testing Background Compaction Triggers ===");

    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("background");

    let mut opts = DbOptions::default();
    opts.data_dir(db_path.clone())
        .memtable_size(8 * 1024 * 1024) // Small memtables to trigger flushes
        .max_memtables(4); // Trigger compaction quickly

    let db = Db::open(opts);

    const VALUE_SIZE: usize = 1024;
    const NUM_ENTRIES: u64 = 200_000;
    const NAMESPACE: &str = "background";

    println!(
        "Writing {} entries to trigger background compaction...",
        NUM_ENTRIES
    );
    let start = Instant::now();

    const BATCH_SIZE: usize = 1000;
    let mut i = 0u64;
    while i < NUM_ENTRIES {
        let mut batch = Vec::with_capacity(BATCH_SIZE);

        for j in 0..BATCH_SIZE {
            if i + j as u64 >= NUM_ENTRIES {
                break;
            }
            let idx = i + j as u64;
            let key = generate_key(NAMESPACE, idx);
            let value = generate_value(VALUE_SIZE, idx);
            batch.push(Put(key, value, db.time()));
        }

        db.batch(&batch).expect("failed to batch write");
        i += batch.len() as u64;

        if i % 50_000 == 0 && i > 0 {
            println!("  Written {}/{}", i, NUM_ENTRIES);

            // Check compaction stats
            if let Ok(stats) = db.compaction_stats() {
                println!("  {}", stats);
            }
        }
    }

    println!("\nWrite phase took {:.2}s", start.elapsed().as_secs_f64());

    // Give background compaction time to catch up
    println!("Waiting for background compaction to settle...");
    thread::sleep(Duration::from_secs(5));

    if let Ok(stats) = db.compaction_stats() {
        println!("\nFinal stats:");
        println!("{}", stats);

        // We expect some compactions to have happened
        assert!(
            stats.completed_jobs > 0,
            "background compaction should have completed some jobs"
        );
    }

    // Verify data integrity
    println!("\nVerifying data integrity...");
    for i in (0..NUM_ENTRIES).step_by(1000) {
        let key = generate_key(NAMESPACE, i);
        let result = db.get(&key).expect("failed to get");
        assert!(result.is_some(), "key {} should exist", i);

        let value = result.unwrap();
        assert!(
            verify_value(&value, VALUE_SIZE, i),
            "value corruption at key {}",
            i
        );
    }

    println!("Verification complete!");
}

#[test]
#[ignore]
fn test_point_lookup_performance() {
    println!("\n=== Testing Point Lookup Performance ===");

    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("point_lookup");

    let mut opts = DbOptions::default();
    opts.data_dir(db_path.clone())
        .memtable_size(64 * 1024 * 1024);

    let db = Db::open(opts);

    const VALUE_SIZE: usize = 256;
    const NUM_ENTRIES: u64 = 1_000_000;
    const NAMESPACE: &str = "perf";

    println!("Phase 1: Loading {} entries...", NUM_ENTRIES);
    let start = Instant::now();

    const BATCH_SIZE: usize = 1000;
    let mut i = 0u64;
    while i < NUM_ENTRIES {
        let mut batch = Vec::with_capacity(BATCH_SIZE);

        for j in 0..BATCH_SIZE {
            if i + j as u64 >= NUM_ENTRIES {
                break;
            }
            let idx = i + j as u64;
            let key = generate_key(NAMESPACE, idx);
            let value = generate_value(VALUE_SIZE, idx);
            batch.push(Put(key, value, db.time()));
        }

        db.batch(&batch).expect("failed to batch write");
        i += batch.len() as u64;

        if i % 200_000 == 0 && i > 0 {
            println!("  Loaded {}/{}", i, NUM_ENTRIES);
        }
    }

    println!("Load took {:.2}s", start.elapsed().as_secs_f64());

    // Measure before compaction
    println!("\nPhase 2: Measuring read performance before compaction...");
    let before_start = Instant::now();
    const READ_SAMPLES: u64 = 10_000;

    for i in 0..READ_SAMPLES {
        let idx = (i * (NUM_ENTRIES / READ_SAMPLES)) % NUM_ENTRIES;
        let key = generate_key(NAMESPACE, idx);
        let _ = db.get(&key).expect("failed to get");
    }

    let before_duration = before_start.elapsed();
    let before_rate = READ_SAMPLES as f64 / before_duration.as_secs_f64();
    println!("Before compaction: {:.0} reads/sec", before_rate);

    // Compact
    println!("\nPhase 3: Compacting...");
    let compact_start = Instant::now();
    db.compact().expect("compaction failed");
    println!(
        "Compaction took {:.2}s",
        compact_start.elapsed().as_secs_f64()
    );

    // Measure after compaction
    println!("\nPhase 4: Measuring read performance after compaction...");
    let after_start = Instant::now();

    for i in 0..READ_SAMPLES {
        let idx = (i * (NUM_ENTRIES / READ_SAMPLES)) % NUM_ENTRIES;
        let key = generate_key(NAMESPACE, idx);
        let _ = db.get(&key).expect("failed to get");
    }

    let after_duration = after_start.elapsed();
    let after_rate = READ_SAMPLES as f64 / after_duration.as_secs_f64();
    println!("After compaction: {:.0} reads/sec", after_rate);

    println!("\n=== Performance Summary ===");
    println!("Before compaction: {:.0} reads/sec", before_rate);
    println!("After compaction:  {:.0} reads/sec", after_rate);
    println!(
        "Improvement:       {:.1}%",
        ((after_rate / before_rate) - 1.0) * 100.0
    );
}
