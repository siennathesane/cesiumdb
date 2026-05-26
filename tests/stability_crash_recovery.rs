//! Stability test: crash recovery.
//!
//! Writes data, drops the Db without shutdown (simulating a crash),
//! then reopens and verifies all committed data is present.


use cesiumdb::{
    Db,
    DbOptions,
};
use stability_framework::ShadowVerifier;
use tempfile::TempDir;

mod stability_framework;

#[test]
fn stability_crash_recovery() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().to_path_buf();

    // Phase 1: Write data and track it in shadow verifier
    let mut opts = DbOptions::default();
    opts.data_dir(path.clone());

    let db = Db::open(opts).unwrap();
    let mut verifier = ShadowVerifier::new();

    // Write a known set of keys
    for i in 0..50_000usize {
        let key = format!("key_{:010}", i).into_bytes();
        let value = format!("value_{:010}", i).into_bytes();
        db.put(&key, &value).unwrap();
        verifier.record_write(key, value);
    }

    // Force some compactions by writing more
    for _ in 0..5 {
        for i in 0..10_000usize {
            let key = format!("key_{:010}", i).into_bytes();
            let value = format!("updated_{:010}", i).into_bytes();
            db.put(&key, &value).unwrap();
            verifier.record_write(key.clone(), value);
        }
    }

    // Simulate crash: drop without orderly shutdown
    drop(db);

    // Phase 2: Reopen and verify
    let mut opts2 = DbOptions::default();
    opts2.data_dir(path);
    let db2 = Db::open(opts2).unwrap();

    // Full point-read verification
    let mut errors = 0;
    for (key, expected_value) in verifier.iter_expected() {
        match db2.get(key) {
            | Ok(Some(actual)) => {
                if actual.as_ref() != expected_value.as_slice() {
                    errors += 1;
                    if errors <= 5 {
                        println!(
                            "MISMATCH: key={:?} expected={:?} actual={:?}",
                            String::from_utf8_lossy(key),
                            hex::encode(expected_value),
                            hex::encode(&actual)
                        );
                    }
                }
            },
            | Ok(None) => {
                errors += 1;
                if errors <= 5 {
                    println!("MISSING: key={:?}", String::from_utf8_lossy(key));
                }
            },
            | Err(e) => {
                errors += 1;
                if errors <= 5 {
                    println!(
                        "READ_ERROR: key={:?} error={:?}",
                        String::from_utf8_lossy(key),
                        e
                    );
                }
            },
        }
    }

    println!("Crash Recovery Results:");
    println!("  Expected keys: {}", verifier.expected_key_count());
    println!("  Errors: {}", errors);

    assert_eq!(errors, 0, "Data loss detected after crash recovery");
}
