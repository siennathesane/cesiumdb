//! Error handling tests for CesiumDB
//!
//! Tests graceful behavior under error conditions

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
        .memtable_size(512 * 1024)
        .max_memtables(2);
    Db::open(opts).unwrap()
}

#[test]
fn test_get_nonexistent_key() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("nonexistent_key");
    let db = make_db(&db_path);

    let result = db.get(b"does-not-exist").expect("get should not error");
    assert!(result.is_none(), "nonexistent key should return None");
}

#[test]
fn test_empty_key_and_value() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("empty_kv");
    let db = make_db(&db_path);

    // Empty value should work
    db.batch(&[Put(b"key-with-empty-val".to_vec(), vec![], db.time())])
        .expect("write with empty value failed");

    let result = db.get(b"key-with-empty-val").expect("get failed");
    assert!(result.is_some(), "key with empty value should exist");
    assert!(result.unwrap().is_empty(), "value should be empty");
}

#[test]
fn test_overwrite_key() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("overwrite");
    let db = make_db(&db_path);

    // Write initial value
    db.batch(&[Put(
        b"overwrite-key".to_vec(),
        b"value1".to_vec(),
        db.time(),
    )])
    .expect("write failed");

    // Overwrite with new value
    db.batch(&[Put(
        b"overwrite-key".to_vec(),
        b"value2".to_vec(),
        db.time(),
    )])
    .expect("overwrite failed");

    let result = db.get(b"overwrite-key").expect("get failed");
    assert_eq!(&result.unwrap()[..], b"value2", "should read latest value");
}

#[test]
fn test_delete_nonexistent_key() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("del_nonexistent");
    let db = make_db(&db_path);

    // Deleting a key that doesn't exist should not error
    db.batch(&[Delete::<Vec<u8>, Vec<u8>>(b"ghost-key".to_vec(), db.time())])
        .expect("delete of nonexistent key should not fail");

    let result = db.get(b"ghost-key").expect("get failed");
    assert!(
        result.is_none(),
        "deleted nonexistent key should return None"
    );
}

#[test]
fn test_write_after_delete() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("write_after_del");
    let db = make_db(&db_path);

    // Write, delete, write again
    db.batch(&[Put(b"resurrected".to_vec(), b"v1".to_vec(), db.time())])
        .expect("first write failed");

    db.batch(&[Delete::<Vec<u8>, Vec<u8>>(
        b"resurrected".to_vec(),
        db.time(),
    )])
    .expect("delete failed");

    let result = db.get(b"resurrected").expect("get failed");
    assert!(result.is_none(), "should be deleted");

    db.batch(&[Put(b"resurrected".to_vec(), b"v2".to_vec(), db.time())])
        .expect("re-write failed");

    let result = db.get(b"resurrected").expect("get failed");
    assert_eq!(&result.unwrap()[..], b"v2", "should have new value");
}
