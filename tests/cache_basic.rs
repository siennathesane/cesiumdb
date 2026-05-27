use std::sync::Arc;
use cesiumdb::{Db, DbOptions, Batch::Put};
use tempfile::TempDir;

#[test]
fn test_cache_basic() {
    let temp_dir = TempDir::new().unwrap();
    let mut opts = DbOptions::default();
    opts.data_dir(temp_dir.path().to_path_buf())
        .memtable_size(512 * 1024)
        .max_memtables(4);
    let db = Arc::new(Db::open(opts).unwrap());

    // Write some keys
    for i in 0..1000 {
        let key = format!("key-{:06}", i);
        let val = format!("val-{:06}", i);
        db.batch(&[Put(key.into_bytes(), val.into_bytes(), db.time())]).unwrap();
    }

    // Read them back
    let mut missing = Vec::new();
    for i in 0..1000 {
        let key = format!("key-{:06}", i);
        if db.get(key.as_bytes()).unwrap().is_none() {
            missing.push(i);
        }
    }
    println!("Missing keys ({}): {:?}", missing.len(), missing);
    assert!(missing.is_empty(), "Missing {} keys", missing.len());
}
