use std::{
    fs,
    path::Path,
    time::Duration,
};

use cesiumdb::{
    Db,
    DbOptions,
};
use tempfile::TempDir;

/// Count segment directories in a given level path
fn count_segment_dirs(level_path: &Path) -> usize {
    if !level_path.exists() {
        return 0;
    }
    fs::read_dir(level_path)
        .unwrap()
        .filter(|e| {
            if let Ok(e) = e {
                e.path().is_dir()
            } else {
                false
            }
        })
        .count()
}

#[test]
fn test_flush_creates_segment_files() {
    let temp_dir = TempDir::new().unwrap();
    let mut opts = DbOptions::new();
    opts.data_dir(temp_dir.path().to_path_buf())
        .memtable_size(64 * 1024 * 1024) // 64MB memtables
        .max_memtables(2);
    let db = Db::open(opts).unwrap();

    // Write enough data to fill at least one memtable and force a flush.
    // With 64MB default memtables, we need ~64MB of data.
    // Using 1KB values, ~64,000 keys = ~64MB.
    const NUM_KEYS: usize = 7_000;
    const VALUE_SIZE: usize = 1024;
    let value = vec![b'v'; VALUE_SIZE];

    for i in 0..NUM_KEYS {
        let key = format!("flush_key_{:06}", i);
        db.put(key.as_bytes(), &value).unwrap();
    }
    db.sync().unwrap();

    // Wait for flush to complete
    std::thread::sleep(Duration::from_millis(500));

    let l0_path = temp_dir.path().join("segments");
    let l0_count = count_segment_dirs(&l0_path);
    println!("L0 segments after flush: {}", l0_count);
    assert!(l0_count >= 1, "expected at least 1 L0 segment after flush");

    // Verify we can read the data back
    for i in 0..NUM_KEYS {
        let key = format!("flush_key_{:06}", i);
        let result = db.get(key.as_bytes()).unwrap();
        assert!(result.is_some(), "key {} should be readable", key);
        assert_eq!(result.unwrap().len(), VALUE_SIZE);
    }

    db.close().unwrap();
}

#[test]
fn test_compaction_data_integrity() {
    let temp_dir = TempDir::new().unwrap();
    let mut opts = DbOptions::new();
    opts.data_dir(temp_dir.path().to_path_buf())
        .max_memtables(2);
    let db = Db::open(opts).unwrap();

    // Write a moderate amount of data
    for batch in 0..3 {
        for i in 0..500 {
            let key = format!("key_{:02}_{:05}", batch, i);
            db.put(key.as_bytes(), b"value").unwrap();
        }
        db.sync().unwrap();
    }

    // Trigger compaction (may or may not schedule jobs depending on L0 count)
    db.compact().unwrap();

    // Wait for any background work
    for _ in 0..30 {
        let stats = db.compaction_stats().unwrap();
        if stats.queued_jobs == 0 && stats.in_progress_jobs == 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }

    // Verify all data is still accessible
    for batch in 0..3 {
        for i in 0..500 {
            let key = format!("key_{:02}_{:05}", batch, i);
            let result = db.get(key.as_bytes()).unwrap();
            assert!(result.is_some(), "Key {} missing after compaction", key);
        }
    }

    db.close().unwrap();
}
