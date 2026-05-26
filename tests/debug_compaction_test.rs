use cesiumdb::{
    Db,
    DbOptions,
};
use tempfile::TempDir;

#[test]
fn debug_compaction() {
    let temp_dir = TempDir::new().unwrap();
    let mut opts = DbOptions::new();
    opts.data_dir(temp_dir.path().to_path_buf())
        .memtable_size(256 * 1024)
        .max_memtables(2);
    let db = Db::open(opts).unwrap();

    for batch in 0..3 {
        for i in 0..500 {
            let key = format!("key_{:02}_{:05}", batch, i);
            db.put(key.as_bytes(), b"value").unwrap();
        }
        db.sync().unwrap();
    }

    let before = db.get(b"key_00_00000").unwrap();
    println!("Before compaction: {:?}", before.is_some());
    assert!(before.is_some(), "key should exist before compaction");

    db.compact().unwrap();

    for _i in 0..30 {
        let stats = db.compaction_stats().unwrap();
        if stats.queued_jobs == 0 && stats.in_progress_jobs == 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }

    let after = db.get(b"key_00_00000").unwrap();
    println!("After compaction: {:?}", after.is_some());

    // Check L1 segment file sizes
    let l1 = temp_dir.path().join("L1").join("segments");
    if l1.exists() {
        for entry in std::fs::read_dir(&l1).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                for f in std::fs::read_dir(&path).unwrap() {
                    let f = f.unwrap();
                    let meta = f.metadata().unwrap();
                    println!("L1 file {:?}: {} bytes", f.file_name(), meta.len());
                }
            }
        }
    }

    // Also check if we can read ALL keys
    let mut found = 0;
    let mut missing = 0;
    for batch in 0..3 {
        for i in 0..500 {
            let key = format!("key_{:02}_{:05}", batch, i);
            if db.get(key.as_bytes()).unwrap().is_some() {
                found += 1;
            } else {
                missing += 1;
                if missing <= 5 {
                    println!("Missing: {}", key);
                }
            }
        }
    }
    println!("Found: {}, Missing: {}", found, missing);

    assert!(after.is_some(), "key should exist after compaction");
    db.close().unwrap();
}
