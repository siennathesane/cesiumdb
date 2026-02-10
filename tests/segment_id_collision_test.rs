use cesiumdb::{Db, DbOptions};
use std::sync::Arc;
use tempfile::TempDir;

/// Tests that flush and compaction use the same global segment ID counter
/// and don't create colliding segment IDs.
///
/// This test verifies that:
/// 1. Multiple flushes create unique segment IDs
/// 2. Compactions triggered during/after flushes don't reuse IDs
/// 3. The database remains consistent without corruption
#[test]
fn test_no_segment_id_collision_between_flush_and_compaction() {
    let temp_dir = TempDir::new().unwrap();
    let mut opts = DbOptions::new();
    opts.data_dir(temp_dir.path().to_path_buf())
        .memtable_size(512 * 1024) // 512KB memtable for faster flushes
        .max_memtables(2);
    let db = Db::open(opts);

    // Trigger multiple flushes to create L0 segments
    for batch in 0..3 {
        for i in 0..1000 {
            let key = format!("key_batch{:02}_item{:05}", batch, i);
            db.put(key.as_bytes(), b"value_data_here").unwrap();
        }
        db.sync().unwrap();
    }

    // Trigger compaction which should create new segments with unique IDs
    db.compact().unwrap();

    // Wait briefly for compaction to complete
    std::thread::sleep(std::time::Duration::from_secs(3));

    // Verify data integrity - if there were ID collisions, we'd have corruption
    // Check all the keys we wrote
    for batch in 0..3 {
        for i in 0..1000 {
            let key = format!("key_batch{:02}_item{:05}", batch, i);
            let result = db.get(key.as_bytes()).unwrap();
            assert!(
                result.is_some(),
                "Key {} missing - possible segment ID collision corruption",
                key
            );
            assert_eq!(
                &result.unwrap()[..],
                b"value_data_here",
                "Value mismatch for key {} - possible corruption",
                key
            );
        }
    }

    db.close().unwrap();
}

/// Tests that segment IDs persist correctly across database restarts
/// and the next segment ID counter starts from the correct value.
///
/// This verifies that:
/// 1. Segment IDs from the first session are preserved
/// 2. New segments after restart have unique IDs (no overlap)
/// 3. No data corruption from ID reuse
#[test]
fn test_segment_id_persists_across_restart() {
    let temp_dir = TempDir::new().unwrap();

    // First session: write some data
    {
        let mut opts = DbOptions::new();
        opts.data_dir(temp_dir.path().to_path_buf());
        let db = Db::open(opts);

        db.put(b"key1", b"value1").unwrap();
        db.put(b"key2", b"value2").unwrap();
        db.put(b"key3", b"value3").unwrap();
        db.sync().unwrap();

        db.close().unwrap();
    }

    // Second session: write more data
    {
        let mut opts = DbOptions::new();
        opts.data_dir(temp_dir.path().to_path_buf());
        let db = Db::open(opts);

        db.put(b"key4", b"value4").unwrap();
        db.put(b"key5", b"value5").unwrap();
        db.put(b"key6", b"value6").unwrap();
        db.sync().unwrap();

        // Verify all keys from both sessions
        assert_eq!(&db.get(b"key1").unwrap().unwrap()[..], b"value1");
        assert_eq!(&db.get(b"key2").unwrap().unwrap()[..], b"value2");
        assert_eq!(&db.get(b"key3").unwrap().unwrap()[..], b"value3");
        assert_eq!(&db.get(b"key4").unwrap().unwrap()[..], b"value4");
        assert_eq!(&db.get(b"key5").unwrap().unwrap()[..], b"value5");
        assert_eq!(&db.get(b"key6").unwrap().unwrap()[..], b"value6");

        db.close().unwrap();
    }
}

/// Tests that concurrent flushes and compactions don't create duplicate IDs
/// even when running in parallel.
///
/// This stress test verifies that:
/// 1. Multiple threads triggering flushes simultaneously don't get duplicate IDs
/// 2. Manual compactions running concurrently with flushes don't reuse IDs
/// 3. All data remains accessible and consistent
#[test]
fn test_concurrent_flush_and_compaction_no_collisions() {
    let temp_dir = TempDir::new().unwrap();
    let mut opts = DbOptions::new();
    opts.data_dir(temp_dir.path().to_path_buf())
        .memtable_size(256 * 1024) // Small memtable for frequent flushes
        .max_memtables(3);
    let db = Db::open(opts);

    // Spawn multiple threads doing writes (triggering flushes)
    let mut handles = vec![];
    for thread_id in 0..3 {
        let db_clone = Arc::clone(&db);
        let handle = std::thread::spawn(move || {
            for i in 0..500 {
                let key = format!("thread{:02}_key{:05}", thread_id, i);
                db_clone.put(key.as_bytes(), b"concurrent_value").unwrap();
            }
            db_clone.sync().unwrap();
        });
        handles.push(handle);
    }

    // While writes are happening, trigger manual compactions
    for _ in 0..3 {
        std::thread::sleep(std::time::Duration::from_millis(300));
        let _ = db.compact();
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Wait for any pending compactions
    std::thread::sleep(std::time::Duration::from_secs(3));

    // Verify all keys are accessible and correct
    for thread_id in 0..3 {
        for i in 0..500 {
            let key = format!("thread{:02}_key{:05}", thread_id, i);
            let result = db.get(key.as_bytes()).unwrap();
            assert!(
                result.is_some(),
                "Key {} missing in concurrent scenario",
                key
            );
            assert_eq!(
                &result.unwrap()[..],
                b"concurrent_value",
                "Value corruption for key {}",
                key
            );
        }
    }

    db.close().unwrap();
}
