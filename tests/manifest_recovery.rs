// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

//! End-to-end manifest recovery tests

use std::sync::Arc;

use cesiumdb::{
    Db,
    DbOptions,
};
use tempfile::TempDir;

fn make_db_with_path(path: std::path::PathBuf) -> Arc<Db> {
    let mut opts = DbOptions::new();
    opts.data_dir(path);
    Db::open(opts).unwrap()
}

#[test]
fn test_basic_recovery_after_close() {
    let temp_dir = TempDir::new().unwrap();

    // Write data to DB
    {
        let db = make_db_with_path(temp_dir.path().to_path_buf());

        // Write multiple keys
        db.put(b"key1", b"value1").unwrap();
        db.put(b"key2", b"value2").unwrap();
        db.put(b"key3", b"value3").unwrap();

        // Force flush to L0
        db.sync().unwrap();

        // Close cleanly
        db.close().unwrap();
    }

    // Reopen DB and verify data
    {
        let db = make_db_with_path(temp_dir.path().to_path_buf());

        let val1 = db.get(b"key1").unwrap().unwrap();
        assert_eq!(&val1[..], b"value1");

        let val2 = db.get(b"key2").unwrap().unwrap();
        assert_eq!(&val2[..], b"value2");

        let val3 = db.get(b"key3").unwrap().unwrap();
        assert_eq!(&val3[..], b"value3");
    }
}

#[test]
fn test_recovery_with_multiple_flushes() {
    let temp_dir = TempDir::new().unwrap();

    // Write data in multiple flushes
    {
        let db = make_db_with_path(temp_dir.path().to_path_buf());

        // First batch
        db.put(b"a1", b"val_a1").unwrap();
        db.put(b"a2", b"val_a2").unwrap();
        db.sync().unwrap();

        // Second batch
        db.put(b"b1", b"val_b1").unwrap();
        db.put(b"b2", b"val_b2").unwrap();
        db.sync().unwrap();

        // Third batch
        db.put(b"c1", b"val_c1").unwrap();
        db.put(b"c2", b"val_c2").unwrap();
        db.sync().unwrap();

        db.close().unwrap();
    }

    // Reopen and verify all data
    {
        let db = make_db_with_path(temp_dir.path().to_path_buf());

        assert_eq!(&db.get(b"a1").unwrap().unwrap()[..], b"val_a1");
        assert_eq!(&db.get(b"a2").unwrap().unwrap()[..], b"val_a2");
        assert_eq!(&db.get(b"b1").unwrap().unwrap()[..], b"val_b1");
        assert_eq!(&db.get(b"b2").unwrap().unwrap()[..], b"val_b2");
        assert_eq!(&db.get(b"c1").unwrap().unwrap()[..], b"val_c1");
        assert_eq!(&db.get(b"c2").unwrap().unwrap()[..], b"val_c2");
    }
}

#[test]
fn test_recovery_with_tombstones() {
    let temp_dir = TempDir::new().unwrap();

    // Write and delete data
    {
        let db = make_db_with_path(temp_dir.path().to_path_buf());

        // Write keys
        db.put(b"key1", b"value1").unwrap();
        db.put(b"key2", b"value2").unwrap();
        db.put(b"key3", b"value3").unwrap();
        db.sync().unwrap();

        // Delete key2
        db.delete(b"key2").unwrap();
        db.sync().unwrap();

        db.close().unwrap();
    }

    // Reopen and verify
    {
        let db = make_db_with_path(temp_dir.path().to_path_buf());

        assert_eq!(&db.get(b"key1").unwrap().unwrap()[..], b"value1");
        assert!(db.get(b"key2").unwrap().is_none()); // Should be deleted
        assert_eq!(&db.get(b"key3").unwrap().unwrap()[..], b"value3");
    }
}

#[test]
fn test_recovery_with_updates() {
    let temp_dir = TempDir::new().unwrap();

    // Write, update, and verify
    {
        let db = make_db_with_path(temp_dir.path().to_path_buf());

        // Initial write
        db.put(b"key", b"value1").unwrap();
        db.sync().unwrap();

        // Update same key
        db.put(b"key", b"value2").unwrap();
        db.sync().unwrap();

        // Another update
        db.put(b"key", b"value3").unwrap();
        db.sync().unwrap();

        db.close().unwrap();
    }

    // Reopen and verify latest value
    {
        let db = make_db_with_path(temp_dir.path().to_path_buf());
        assert_eq!(&db.get(b"key").unwrap().unwrap()[..], b"value3");
    }
}

#[test]
fn test_recovery_with_large_dataset() {
    let temp_dir = TempDir::new().unwrap();

    // Write many keys
    {
        let db = make_db_with_path(temp_dir.path().to_path_buf());

        for i in 0..1000 {
            let key = format!("key_{:04}", i);
            let value = format!("value_{:04}", i);
            db.put(key.as_bytes(), value.as_bytes()).unwrap();
        }

        db.sync().unwrap();
        db.close().unwrap();
    }

    // Reopen and verify random keys
    {
        let db = make_db_with_path(temp_dir.path().to_path_buf());

        for i in [0, 100, 500, 999] {
            let key = format!("key_{:04}", i);
            let expected_value = format!("value_{:04}", i);
            let value = db.get(key.as_bytes()).unwrap().unwrap();
            assert_eq!(&value[..], expected_value.as_bytes());
        }
    }
}

#[test]
fn test_recovery_empty_db() {
    let temp_dir = TempDir::new().unwrap();

    // Create DB but don't write anything
    {
        let db = make_db_with_path(temp_dir.path().to_path_buf());
        db.close().unwrap();
    }

    // Reopen empty DB
    {
        let db = make_db_with_path(temp_dir.path().to_path_buf());
        assert!(db.get(b"any_key").unwrap().is_none());
    }
}

#[test]
fn test_multiple_reopen_cycles() {
    let temp_dir = TempDir::new().unwrap();

    // First cycle: write initial data
    {
        let db = make_db_with_path(temp_dir.path().to_path_buf());
        db.put(b"cycle1", b"data1").unwrap();
        db.sync().unwrap();
        db.close().unwrap();
    }

    // Second cycle: add more data
    {
        let db = make_db_with_path(temp_dir.path().to_path_buf());
        assert_eq!(&db.get(b"cycle1").unwrap().unwrap()[..], b"data1");

        db.put(b"cycle2", b"data2").unwrap();
        db.sync().unwrap();
        db.close().unwrap();
    }

    // Third cycle: add even more data
    {
        let db = make_db_with_path(temp_dir.path().to_path_buf());
        assert_eq!(&db.get(b"cycle1").unwrap().unwrap()[..], b"data1");
        assert_eq!(&db.get(b"cycle2").unwrap().unwrap()[..], b"data2");

        db.put(b"cycle3", b"data3").unwrap();
        db.sync().unwrap();
        db.close().unwrap();
    }

    // Final verification
    {
        let db = make_db_with_path(temp_dir.path().to_path_buf());
        assert_eq!(&db.get(b"cycle1").unwrap().unwrap()[..], b"data1");
        assert_eq!(&db.get(b"cycle2").unwrap().unwrap()[..], b"data2");
        assert_eq!(&db.get(b"cycle3").unwrap().unwrap()[..], b"data3");
    }
}

#[test]
fn test_recovery_with_namespaces() {
    let temp_dir = TempDir::new().unwrap();

    // Write to different namespaces
    {
        let db = make_db_with_path(temp_dir.path().to_path_buf());

        db.put_ns(1, b"key1", b"value1").unwrap();
        db.put_ns(2, b"key1", b"value2").unwrap();
        db.put(b"key1", b"default_value").unwrap();

        db.sync().unwrap();
        db.close().unwrap();
    }

    // Reopen and verify namespace isolation
    {
        let db = make_db_with_path(temp_dir.path().to_path_buf());

        // Default namespace
        assert_eq!(&db.get(b"key1").unwrap().unwrap()[..], b"default_value");

        // ns1
        assert_eq!(&db.get_ns(1, b"key1").unwrap().unwrap()[..], b"value1");

        // ns2
        assert_eq!(&db.get_ns(2, b"key1").unwrap().unwrap()[..], b"value2");
    }
}

#[test]
fn test_recovery_after_unclean_shutdown_simulation() {
    let temp_dir = TempDir::new().unwrap();

    // Simulate unclean shutdown (no close() call)
    {
        let db = make_db_with_path(temp_dir.path().to_path_buf());

        db.put(b"persistent1", b"value1").unwrap();
        db.put(b"persistent2", b"value2").unwrap();

        // Sync to ensure data is on disk
        db.sync().unwrap();

        // Drop without calling close() - simulates crash
        drop(db);
    }

    // Reopen and verify data survived
    {
        let db = make_db_with_path(temp_dir.path().to_path_buf());

        assert_eq!(&db.get(b"persistent1").unwrap().unwrap()[..], b"value1");
        assert_eq!(&db.get(b"persistent2").unwrap().unwrap()[..], b"value2");
    }
}
