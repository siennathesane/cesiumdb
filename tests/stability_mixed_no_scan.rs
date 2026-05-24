use std::sync::Arc;
use std::sync::Mutex;

use cesiumdb::{Db, DbOptions};
use stability_framework::{
    run_stability_test,
    ShadowVerifier,
    StabilityConfig,
};
use tempfile::TempDir;

mod stability_framework;

#[test]
fn stability_mixed_concurrent_no_scan() {
    let temp_dir = TempDir::new().unwrap();
    let mut opts = DbOptions::default();
    opts.data_dir(temp_dir.path().to_path_buf());

    let db = Db::open(opts);
    let verifier = Arc::new(Mutex::new(ShadowVerifier::new()));

    let config = StabilityConfig {
        duration_secs: 30,
        num_writers: 4,
        num_readers: 4,
        num_scanners: 0,
        key_space: 500_000,
        value_size: 1024,
        write_rate_hz: 1500,
        verification_interval_ms: 1000,
        delete_probability: 0.1,
    };

    let metrics = run_stability_test(db, verifier.clone(), config);

    println!("Results:");
    println!("  Writes: {}", metrics.total_writes);
    println!("  Reads: {}", metrics.total_reads);
    println!("  Deletes: {}", metrics.total_deletes);
    println!("  Verification passes: {}", metrics.verification_passes);
    println!("  Verification failures: {}", metrics.verification_failures);
    if let Some(space_amp) = metrics.space_amp {
        println!("  Space amp: {:.2}x", space_amp);
    }

    let v = verifier.lock().unwrap();
    assert!(
        metrics.verification_failures == 0,
        "Verification failures detected: {:?}",
        metrics.errors
    );
    assert!(v.is_clean(), "Shadow verifier has errors: {:?}", v.errors);
}
