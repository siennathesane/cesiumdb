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
    run_stability_test_final_verify_only,
};
use tempfile::TempDir;

mod stability_framework;

#[test]
fn stability_mixed_debug_final_verify() {
    let temp_dir = TempDir::new().unwrap();
    let mut opts = DbOptions::default();
    opts.data_dir(temp_dir.path().to_path_buf());

    let db = Db::open(opts).unwrap();
    let verifier = Arc::new(Mutex::new(ShadowVerifier::new()));

    let config = StabilityConfig {
        duration_secs: 60,
        num_writers: 4,
        num_readers: 4,
        num_scanners: 0,
        key_space: 500_000,
        value_size: 1024,
        write_rate_hz: 1500,
        verification_interval_ms: 1000,
        delete_probability: 0.1,
    };

    let metrics = run_stability_test_final_verify_only(db.clone(), verifier.clone(), config);

    println!("Results:");
    println!("  Writes: {}", metrics.total_writes);
    println!("  Reads: {}", metrics.total_reads);
    println!("  Deletes: {}", metrics.total_deletes);
    println!("  Verification passes: {}", metrics.verification_passes);
    println!(
        "  Verification failures during run: {}",
        metrics.verification_failures
    );
    if let Some(space_amp) = metrics.space_amp {
        println!("  Space amp: {:.2}x", space_amp);
    }

    let mut v = verifier.lock().unwrap();
    println!("Final expected count: {}", v.expected_key_count());

    // Final verification after all workers stopped
    let sample_count = (500_000 / 100).max(100);
    let sample_ok = v.verify_random_sample(&db, sample_count);
    let deletes_ok = v.verify_deletes(&db);

    println!("Final sample verification: {}", sample_ok);
    println!("Final delete verification: {}", deletes_ok);
    println!("Final errors: {:?}", v.errors);

    assert!(
        sample_ok,
        "Final sample verification failed: {:?}",
        v.errors
    );
    assert!(
        deletes_ok,
        "Final delete verification failed: {:?}",
        v.errors
    );
    assert!(v.is_clean(), "Shadow verifier has errors: {:?}", v.errors);
}
