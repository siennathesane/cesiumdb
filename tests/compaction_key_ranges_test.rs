use cesiumdb::{Db, DbOptions};
use tempfile::TempDir;

/// Tests that compaction jobs have valid key ranges (not empty placeholders)
/// by verifying that L0→L1 compaction actually works and doesn't fail.
#[test]
fn test_l0_to_l1_compaction_with_key_ranges() {
    let temp_dir = TempDir::new().unwrap();
    let mut opts = DbOptions::new();
    opts.data_dir(temp_dir.path().to_path_buf())
        .memtable_size(256 * 1024) // Small memtable for frequent flushes
        .max_memtables(2);
    let db = Db::open(opts);

    // Write data in multiple batches to create several L0 segments
    for batch in 0..3 {
        for i in 0..500 {
            let key = format!("key_{:02}_{:05}", batch, i);
            db.put(key.as_bytes(), b"value").unwrap();
        }
        db.sync().unwrap();
    }

    // Trigger compaction - if key ranges were empty, overlap detection would fail
    // and compaction wouldn't work correctly
    db.compact().unwrap();

    // Wait for compaction to complete
    std::thread::sleep(std::time::Duration::from_secs(3));

    // Verify all data is still accessible (compaction worked correctly)
    for batch in 0..3 {
        for i in 0..500 {
            let key = format!("key_{:02}_{:05}", batch, i);
            let result = db.get(key.as_bytes()).unwrap();
            assert!(
                result.is_some(),
                "Key {} missing after compaction - key range tracking failed",
                key
            );
            assert_eq!(
                &result.unwrap()[..],
                b"value",
                "Value corruption for key {}",
                key
            );
        }
    }

    db.close().unwrap();
}

/// Tests that compaction correctly finds overlapping L1 segments
/// when compacting L0 segments with specific key ranges.
#[test]
fn test_overlap_detection_with_key_ranges() {
    let temp_dir = TempDir::new().unwrap();
    let mut opts = DbOptions::new();
    opts.data_dir(temp_dir.path().to_path_buf())
        .memtable_size(128 * 1024)
        .max_memtables(2);
    let db = Db::open(opts);

    // Create L0 segments with distinct key ranges
    // Batch 1: keys starting with 'a'
    for i in 0..500 {
        let key = format!("a_key_{:05}", i);
        db.put(key.as_bytes(), b"value_a").unwrap();
    }
    db.sync().unwrap();

    // Batch 2: keys starting with 'z'
    for i in 0..500 {
        let key = format!("z_key_{:05}", i);
        db.put(key.as_bytes(), b"value_z").unwrap();
    }
    db.sync().unwrap();

    // Trigger compaction
    db.compact().unwrap();
    std::thread::sleep(std::time::Duration::from_secs(2));

    // Verify both ranges are accessible
    for i in 0..500 {
        let key_a = format!("a_key_{:05}", i);
        let result = db.get(key_a.as_bytes()).unwrap();
        assert!(result.is_some(), "Key {} missing", key_a);
        assert_eq!(&result.unwrap()[..], b"value_a");

        let key_z = format!("z_key_{:05}", i);
        let result = db.get(key_z.as_bytes()).unwrap();
        assert!(result.is_some(), "Key {} missing", key_z);
        assert_eq!(&result.unwrap()[..], b"value_z");
    }

    db.close().unwrap();
}

/// Tests that compaction works correctly across multiple levels
/// with proper key range tracking.
#[test]
fn test_multi_level_compaction_with_key_ranges() {
    let temp_dir = TempDir::new().unwrap();
    let mut opts = DbOptions::new();
    opts.data_dir(temp_dir.path().to_path_buf())
        .memtable_size(128 * 1024)
        .max_memtables(2);
    let db = Db::open(opts);

    // Create enough data to trigger multiple levels of compaction
    for batch in 0..3 {
        for i in 0..300 {
            let key = format!("multi_level_key_{:02}_{:05}", batch, i);
            db.put(key.as_bytes(), b"multi_level_value").unwrap();
        }
        db.sync().unwrap();
    }

    // Trigger compactions
    db.compact().unwrap();
    std::thread::sleep(std::time::Duration::from_secs(2));
    db.compact().unwrap();
    std::thread::sleep(std::time::Duration::from_secs(2));

    // Verify all data is accessible across all levels
    for batch in 0..3 {
        for i in 0..300 {
            let key = format!("multi_level_key_{:02}_{:05}", batch, i);
            let result = db.get(key.as_bytes()).unwrap();
            assert!(
                result.is_some(),
                "Key {} missing - multi-level compaction failed",
                key
            );
            assert_eq!(&result.unwrap()[..], b"multi_level_value");
        }
    }

    db.close().unwrap();
}
