//! Stability test: delete-heavy workload.
//!
//! 50% of operations are deletes. Verifies tombstone propagation
//! and that deleted keys stay deleted.

use std::sync::{
    Arc,
    Mutex,
};

use cesiumdb::{
    Db,
    DbOptions,
};
use stability_framework::{
    ShadowVerifier,
    StabilityConfig,
    run_stability_test,
};
use tempfile::TempDir;

mod stability_framework;

#[test]
fn stability_delete_heavy() {
    let temp_dir = TempDir::new().unwrap();
    let mut opts = DbOptions::default();
    opts.data_dir(temp_dir.path().to_path_buf());

    let db = Db::open(opts);
    let verifier = Arc::new(Mutex::new(ShadowVerifier::new()));

    // Pre-fill some data
    {
        let mut v = verifier.lock().unwrap();
        for i in 0..10_000usize {
            let key = format!("key_{:010}", i).into_bytes();
            let value = format!("value_{:010}", i).into_bytes();
            db.put(&key, &value).unwrap();
            v.record_write(key, value);
        }
    }

    let config = StabilityConfig {
        duration_secs: 30,
        num_writers: 4,
        num_readers: 4,
        num_scanners: 0,
        key_space: 10_000,
        value_size: 1024,
        write_rate_hz: 1000,
        verification_interval_ms: 500,
        delete_probability: 0.5,
    };

    let metrics = run_stability_test(db.clone(), verifier.clone(), config);

    println!("Stability Delete-Heavy Results:");
    println!("  Writes: {}", metrics.total_writes);
    println!("  Deletes: {}", metrics.total_deletes);
    println!("  Reads: {}", metrics.total_reads);
    println!("  Verification passes: {}", metrics.verification_passes);
    println!("  Verification failures: {}", metrics.verification_failures);

    // Final full verification
    let mut v = verifier.lock().unwrap();
    assert!(
        v.verify_deletes(&db),
        "Deleted keys were still readable: {:?}",
        v.errors
    );
    assert!(
        metrics.verification_failures == 0,
        "Verification failures: {:?}",
        metrics.errors
    );
    assert!(v.is_clean(), "Shadow verifier has errors: {:?}", v.errors);
}
