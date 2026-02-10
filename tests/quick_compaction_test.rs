// Quick test to debug compaction issues
use cesiumdb::{
    Batch::*,
    Db,
    DbOptions,
};
use tempfile::TempDir;

#[test]
fn test_compaction_small_dataset() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("quick_test");

    let mut opts = DbOptions::default();
    opts.data_dir(db_path.clone())
        .memtable_size(32 * 1024 * 1024)
        .max_memtables(4);

    let db = Db::open(opts);

    // Write 100k keys
    const NUM_KEYS: u64 = 100_000;
    const BATCH_SIZE: usize = 1000;

    println!("Writing {} keys...", NUM_KEYS);
    for i in (0..NUM_KEYS).step_by(BATCH_SIZE) {
        let mut batch = Vec::with_capacity(BATCH_SIZE);
        for j in 0..BATCH_SIZE {
            if i + j as u64 >= NUM_KEYS {
                break;
            }
            let idx = i + j as u64;
            let key = format!("test-key-{:08}", idx);
            let value = vec![idx as u8; 100];
            batch.push(Put(key.into_bytes(), value, db.time()));
        }
        db.batch(&batch).expect("failed to write batch");
    }

    println!("Running compaction...");
    db.compact().expect("compaction failed");

    println!("Verifying all keys...");
    let mut missing_count = 0;
    let mut first_missing = None;

    for idx in 0..NUM_KEYS {
        let key = format!("test-key-{:08}", idx);
        match db.get(key.as_bytes()) {
            | Ok(Some(_)) => {}, // Found
            | Ok(None) => {
                missing_count += 1;
                if first_missing.is_none() {
                    first_missing = Some(idx);
                }
            },
            | Err(e) => {
                panic!("Error getting key {}: {:?}", idx, e);
            },
        }
    }

    if missing_count > 0 {
        println!("Missing {} keys out of {}", missing_count, NUM_KEYS);
        println!("First missing key: {:?}", first_missing);
        panic!("Keys are missing after compaction!");
    }

    println!("All {} keys found successfully!", NUM_KEYS);
}
