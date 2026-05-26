use cesiumdb::{
    Db,
    DbOptions,
};
use tempfile::TempDir;

/// Tests that manual compaction doesn't create duplicate jobs
/// by verifying that calling compact() multiple times works correctly.
#[test]
fn test_manual_compact_no_duplicates() {
    let temp_dir = TempDir::new().unwrap();
    let mut opts = DbOptions::new();
    opts.data_dir(temp_dir.path().to_path_buf())
        .memtable_size(256 * 1024)
        .max_memtables(2);
    let db = Db::open(opts).unwrap();

    // Create L0 segments
    for i in 0..1000 {
        let key = format!("key{:05}", i);
        db.put(key.as_bytes(), b"value").unwrap();
    }
    db.sync().unwrap();

    // Call compact multiple times - should not create duplicate jobs
    db.compact().unwrap();
    db.compact().unwrap();
    db.compact().unwrap();

    // Wait for compactions to complete
    std::thread::sleep(std::time::Duration::from_secs(2));

    // Verify data is still accessible
    for i in 0..1000 {
        let key = format!("key{:05}", i);
        let result = db.get(key.as_bytes()).unwrap();
        assert!(result.is_some(), "Key {} missing", key);
        assert_eq!(&result.unwrap()[..], b"value");
    }

    db.close().unwrap();
}

/// Tests that in-flight tracking doesn't block valid subsequent compactions
/// after the first compaction completes.
#[test]
fn test_in_flight_cleared_after_completion() {
    let temp_dir = TempDir::new().unwrap();
    let mut opts = DbOptions::new();
    opts.data_dir(temp_dir.path().to_path_buf())
        .memtable_size(256 * 1024)
        .max_memtables(2);
    let db = Db::open(opts).unwrap();

    // First compaction
    for i in 0..800 {
        let key = format!("key1_{:05}", i);
        db.put(key.as_bytes(), b"value1").unwrap();
    }
    db.sync().unwrap();
    db.compact().unwrap();

    // Wait for completion
    std::thread::sleep(std::time::Duration::from_secs(2));

    // Second compaction with new data - should not be blocked
    for i in 0..800 {
        let key = format!("key2_{:05}", i);
        db.put(key.as_bytes(), b"value2").unwrap();
    }
    db.sync().unwrap();
    db.compact().unwrap();

    // Wait for completion
    std::thread::sleep(std::time::Duration::from_secs(2));

    // Verify all data is accessible
    for i in 0..800 {
        let key1 = format!("key1_{:05}", i);
        let result = db.get(key1.as_bytes()).unwrap();
        assert!(result.is_some(), "Key {} missing", key1);

        let key2 = format!("key2_{:05}", i);
        let result = db.get(key2.as_bytes()).unwrap();
        assert!(result.is_some(), "Key {} missing", key2);
    }

    db.close().unwrap();
}

/// Tests that concurrent manual compactions work correctly
/// without creating duplicate jobs.
#[test]
fn test_concurrent_manual_compactions() {
    let temp_dir = TempDir::new().unwrap();
    let mut opts = DbOptions::new();
    opts.data_dir(temp_dir.path().to_path_buf())
        .memtable_size(128 * 1024)
        .max_memtables(2);
    let db = std::sync::Arc::new(Db::open(opts).unwrap());

    // Create data
    for i in 0..600 {
        let key = format!("concurrent_key_{:05}", i);
        db.put(key.as_bytes(), b"concurrent_value").unwrap();
    }
    db.sync().unwrap();

    // Trigger multiple concurrent compactions
    let mut handles = vec![];
    for _ in 0..3 {
        let db_clone = std::sync::Arc::clone(&db);
        let handle = std::thread::spawn(move || {
            db_clone.compact().unwrap();
        });
        handles.push(handle);
    }

    // Wait for all compaction requests
    for handle in handles {
        handle.join().unwrap();
    }

    // Wait for compactions to complete
    std::thread::sleep(std::time::Duration::from_secs(3));

    // Verify data integrity
    for i in 0..600 {
        let key = format!("concurrent_key_{:05}", i);
        let result = db.get(key.as_bytes()).unwrap();
        assert!(result.is_some(), "Key {} missing", key);
        assert_eq!(&result.unwrap()[..], b"concurrent_value");
    }

    db.close().unwrap();
}
