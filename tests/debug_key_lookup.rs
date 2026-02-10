// Debug test to reproduce the key lookup failure
use cesiumdb::{
    Batch::*,
    Db,
    DbOptions,
};
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
