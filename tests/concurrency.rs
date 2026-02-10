//! Concurrency tests for CesiumDB
//!
//! Tests concurrent access patterns including:
//! - Concurrent writes and reads
//! - Reads during compaction
//! - Rapid memtable rotation

use std::{
    sync::Arc,
    thread,
};

use cesiumdb::{
    Batch::*,
    Db,
    DbOptions,
};
use tempfile::TempDir;

fn make_db(dir: &std::path::Path) -> Arc<Db> {
    let mut opts = DbOptions::default();
    opts.data_dir(dir.to_path_buf())
        .memtable_size(512 * 1024) // 512KB memtable for faster rotation
        .max_memtables(4);
    Db::open(opts)
}

#[test]
fn test_concurrent_writes_and_reads() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("concurrent_rw");
    let db = Arc::new(make_db(&db_path));

    const NUM_WRITERS: usize = 4;
    const KEYS_PER_WRITER: u64 = 500;

    // Spawn writer threads
    let mut handles = vec![];
    for w in 0..NUM_WRITERS {
        let db = Arc::clone(&db);
        handles.push(thread::spawn(move || {
            for i in 0..KEYS_PER_WRITER {
                let key = format!("w{}-key-{:06}", w, i);
                let val = format!("w{}-val-{:06}", w, i);
                db.batch(&[Put(key.into_bytes(), val.into_bytes(), db.time())])
                    .expect("write failed");
            }
        }));
    }

    // Spawn reader threads that read keys concurrently
    for w in 0..NUM_WRITERS {
        let db = Arc::clone(&db);
        handles.push(thread::spawn(move || {
            // Read with some delay to overlap with writers
            thread::sleep(std::time::Duration::from_millis(10));
            for i in 0..KEYS_PER_WRITER {
                let key = format!("w{}-key-{:06}", w, i);
                // Don't assert found — writer may not have written yet
                let _ = db.get(key.as_bytes());
            }
        }));
    }

    for h in handles {
        h.join().expect("thread panicked");
    }

    // After all writers complete, verify all keys exist
    for w in 0..NUM_WRITERS {
        for i in 0..KEYS_PER_WRITER {
            let key = format!("w{}-key-{:06}", w, i);
            let result = db.get(key.as_bytes()).expect("get failed");
            assert!(result.is_some(), "key w{}-key-{:06} missing", w, i);
        }
    }
}

#[test]
fn test_reads_during_compaction() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("read_during_compact");
    let db = Arc::new(make_db(&db_path));

    const NUM_KEYS: u64 = 5000;

    // Write enough keys to trigger flushing
    for i in 0..NUM_KEYS {
        let key = format!("compact-key-{:08}", i);
        let val = vec![i as u8; 100];
        db.batch(&[Put(key.into_bytes(), val, db.time())])
            .expect("write failed");
    }

    // Spawn readers that continue reading while we compact
    let db_reader = Arc::clone(&db);
    let reader_handle = thread::spawn(move || {
        let mut found = 0u64;
        for i in 0..NUM_KEYS {
            let key = format!("compact-key-{:08}", i);
            if let Ok(Some(_)) = db_reader.get(key.as_bytes()) {
                found += 1;
            }
        }
        found
    });

    // Trigger compaction while reader is active
    let _ = db.compact();

    let found = reader_handle.join().expect("reader panicked");
    // All keys should still be accessible (compaction doesn't lose data)
    assert_eq!(
        found,
        NUM_KEYS,
        "lost {} keys during compaction",
        NUM_KEYS - found
    );
}

#[test]
fn test_rapid_memtable_rotation() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("rapid_rotation");

    let mut opts = DbOptions::default();
    opts.data_dir(db_path)
        .memtable_size(16 * 1024) // 16KB — very small for rapid rotation
        .max_memtables(8);
    let db = Db::open(opts);

    const NUM_KEYS: u64 = 10_000;

    // Write lots of data to force many memtable rotations
    for i in 0..NUM_KEYS {
        let key = format!("rot-key-{:08}", i);
        let val = vec![(i % 256) as u8; 50];
        db.batch(&[Put(key.into_bytes(), val, db.time())])
            .expect("write failed");
    }

    // All data should still be readable
    let mut found = 0u64;
    for i in 0..NUM_KEYS {
        let key = format!("rot-key-{:08}", i);
        if let Ok(Some(_)) = db.get(key.as_bytes()) {
            found += 1;
        }
    }

    assert_eq!(
        found,
        NUM_KEYS,
        "lost {} keys during rapid rotation",
        NUM_KEYS - found
    );
}

#[test]
fn test_concurrent_get_and_delete() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("concurrent_del");
    let db = Arc::new(make_db(&db_path));

    const NUM_KEYS: u64 = 1000;

    // Pre-populate
    for i in 0..NUM_KEYS {
        let key = format!("del-key-{:06}", i);
        let val = format!("del-val-{:06}", i);
        db.batch(&[Put(key.into_bytes(), val.into_bytes(), db.time())])
            .expect("write failed");
    }

    // Delete even keys while reading odd keys concurrently
    let db_deleter = Arc::clone(&db);
    let deleter = thread::spawn(move || {
        for i in (0..NUM_KEYS).step_by(2) {
            let key = format!("del-key-{:06}", i);
            db_deleter
                .batch(&[Delete::<Vec<u8>, Vec<u8>>(
                    key.into_bytes(),
                    db_deleter.time(),
                )])
                .expect("delete failed");
        }
    });

    let db_reader = Arc::clone(&db);
    let reader = thread::spawn(move || {
        for i in (1..NUM_KEYS).step_by(2) {
            let key = format!("del-key-{:06}", i);
            // Odd keys should still exist
            let _ = db_reader.get(key.as_bytes());
        }
    });

    deleter.join().expect("deleter panicked");
    reader.join().expect("reader panicked");

    // Verify: odd keys present, even keys deleted
    for i in 0..NUM_KEYS {
        let key = format!("del-key-{:06}", i);
        let result = db.get(key.as_bytes()).expect("get failed");
        if i % 2 == 0 {
            assert!(result.is_none(), "even key {} should be deleted", i);
        } else {
            assert!(result.is_some(), "odd key {} should still exist", i);
        }
    }
}
