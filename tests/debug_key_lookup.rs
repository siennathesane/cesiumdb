// Debug test to reproduce the key lookup failure
use cesiumdb::{
    Batch::*,
    Db,
    DbOptions,
};
use std::ops::Bound;
use tempfile::TempDir;

#[test]
fn test_simple_write_and_read() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("debug_test");

    let mut opts = DbOptions::default();
    opts.data_dir(db_path.clone())
        .memtable_size(1024 * 1024) // 1MB memtable - will flush quickly
        .max_memtables(2);

    let db = Db::open(opts);

    // Write a single key
    let key = b"test-key";
    let value = vec![42u8; 100];

    println!("Writing key: {:?}", String::from_utf8_lossy(key));
    db.batch(&[Put(key.to_vec(), value.clone(), db.time())])
        .expect("failed to write");

    // Force flush by writing enough data to trigger memtable flush
    for i in 0..10_000 {
        let k = format!("filler-key-{:05}", i);
        let v = vec![i as u8; 100];
        db.batch(&[Put(k.into_bytes(), v, db.time())])
            .expect("failed to write filler");
    }

    // Wait a bit for background flush
    std::thread::sleep(std::time::Duration::from_secs(2));

    // Debug: check DB state
    println!("Version stats: {}", db.version_stats());
    println!("Frozen memtables: {}", db.frozen_memtable_count());

    // Try scanning a few keys
    let scan_results: Vec<_> = db.scan(Bound::Unbounded, Bound::Unbounded).collect();
    println!("Total keys in DB scan: {}", scan_results.len());
    
    // Try reading the first filler key
    let filler_key = format!("filler-key-{:05}", 0).into_bytes();
    let filler_result = db.get(&filler_key).expect("failed to get filler");
    println!("Filler key 0 found: {}", filler_result.is_some());

    // Try to read the original key
    println!("Reading key: {:?}", String::from_utf8_lossy(key));
    let result = db.get(key).expect("failed to get");

    match result {
        | Some(v) => {
            println!("Found value: {} bytes", v.len());
            assert_eq!(v, value, "value mismatch");
        },
        | None => {
            panic!("Key not found!");
        },
    }
}
