//! Lifecycle tests for Db::close(), Db::sync(), and Drop behavior

use std::sync::Arc;

use cesiumdb::{
    Batch::*,
    Db,
    DbOptions,
};
use tempfile::TempDir;

fn make_db(dir: &std::path::Path) -> Arc<Db> {
    let mut opts = DbOptions::default();
    opts.data_dir(dir.to_path_buf())
        .memtable_size(1024 * 1024) // 1MB memtable for faster flushes
        .max_memtables(2);
    Db::open(opts).unwrap()
}

#[test]
#[ignore] // Requires manifest-based recovery on reopen (post-1.0)
fn test_close_persists_data() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("close_test");

    // Write data, close, reopen, verify
    {
        let db = make_db(&db_path);
        for i in 0..100u64 {
            let key = format!("key-{:04}", i);
            let val = format!("val-{:04}", i);
            db.batch(&[Put(key.into_bytes(), val.into_bytes(), db.time())])
                .expect("write failed");
        }
        db.close().expect("close failed");
    }

    // Reopen and verify
    {
        let db = make_db(&db_path);
        for i in 0..100u64 {
            let key = format!("key-{:04}", i);
            let expected_val = format!("val-{:04}", i);
            let result = db.get(key.as_bytes()).expect("get failed");
            assert!(result.is_some(), "key {} missing after reopen", i);
            assert_eq!(&result.unwrap()[..], expected_val.as_bytes());
        }
    }
}

#[test]
fn test_close_during_writes() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("close_during_writes");

    let db = make_db(&db_path);

    // Write enough data to trigger memtable flushes
    for i in 0..5000u64 {
        let key = format!("key-{:08}", i);
        let val = vec![i as u8; 200];
        db.batch(&[Put(key.into_bytes(), val, db.time())])
            .expect("write failed");
    }

    // Close should succeed even with pending flushes
    db.close().expect("close should succeed");
}

#[test]
fn test_sync_persists_data() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("sync_test");

    let db = make_db(&db_path);
    for i in 0..50u64 {
        let key = format!("sync-key-{:04}", i);
        let val = format!("sync-val-{:04}", i);
        db.batch(&[Put(key.into_bytes(), val.into_bytes(), db.time())])
            .expect("write failed");
    }

    // Sync should not panic or error
    db.sync().expect("sync failed");

    // Data should still be readable after sync
    for i in 0..50u64 {
        let key = format!("sync-key-{:04}", i);
        let result = db.get(key.as_bytes()).expect("get failed");
        assert!(result.is_some(), "key {} missing after sync", i);
    }
}

#[test]
fn test_double_close() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("double_close_test");

    let db = make_db(&db_path);
    db.batch(&[Put(b"key".to_vec(), b"val".to_vec(), db.time())])
        .expect("write failed");

    // First close should succeed
    db.close().expect("first close failed");

    // Second close should not panic (it's a no-op if already closed)
    db.close().expect("second close should not fail");
}

#[test]
fn test_drop_joins_threads() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("drop_test");

    {
        let db = make_db(&db_path);
        for i in 0..1000u64 {
            let key = format!("drop-key-{:06}", i);
            let val = vec![i as u8; 100];
            db.batch(&[Put(key.into_bytes(), val, db.time())])
                .expect("write failed");
        }
        // db drops here — should not hang or panic
    }

    // If we reach here, drop completed successfully
}
