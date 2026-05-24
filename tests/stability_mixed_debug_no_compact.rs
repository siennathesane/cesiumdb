use std::sync::Arc;
use std::sync::Mutex;

use cesiumdb::{Db, DbOptions};
use cesiumdb::compaction::SchedulerConfig;
use stability_framework::{
    run_stability_test,
    ShadowVerifier,
    StabilityConfig,
};
use tempfile::TempDir;

mod stability_framework;

#[test]
fn stability_mixed_debug_no_compact() {
    let temp_dir = TempDir::new().unwrap();
    let mut opts = DbOptions::default();
    opts.data_dir(temp_dir.path().to_path_buf());
    
    // Disable compactions by setting trigger very high
    let mut scheduler_config = SchedulerConfig::default();
    scheduler_config.l0_compaction_trigger = 10000;
    opts.scheduler_config(scheduler_config);

    let db = Db::open(opts);
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

    let metrics = run_stability_test(db.clone(), verifier.clone(), config);

    println!("Results:");
    println!("  Writes: {}", metrics.total_writes);
    println!("  Reads: {}", metrics.total_reads);
    println!("  Deletes: {}", metrics.total_deletes);
    println!("  Verification passes: {}", metrics.verification_passes);
    println!("  Verification failures: {}", metrics.verification_failures);
    if let Some(amp) = metrics.read_amp_stats {
        let avg_l0 = if amp.total_gets > 0 { amp.l0_segments_checked as f64 / amp.total_gets as f64 } else { 0.0 };
        let avg_ln = if amp.total_gets > 0 { amp.ln_segments_checked as f64 / amp.total_gets as f64 } else { 0.0 };
        println!("  Read amp: total_gets={}, avg L0 segments={:.2}, avg L1-L7 segments={:.2}",
            amp.total_gets, avg_l0, avg_ln);
    }
    if let Some(space_amp) = metrics.space_amp {
        println!("  Space amp: {:.2}x", space_amp);
    }
    {
        let v = verifier.lock().unwrap();
        println!("  Version stats: {}", db.version_stats());
        println!("  Live data bytes: {}", v.live_data_bytes());
        println!("  Tombstone bytes: {}", v.tombstone_bytes());
        println!("  Expected keys: {}", v.expected_key_count());
    }

    db.sync().unwrap();
    
    let mut v = verifier.lock().unwrap();
    v.errors.clear();
    let sample_ok = v.verify_random_sample(&db, 5000);
    let deletes_ok = v.verify_deletes(&db);
    
    println!("Final sample: {}", sample_ok);
    println!("Final deletes: {}", deletes_ok);
    println!("Final errors: {:?}", v.errors);
    
    assert!(sample_ok, "Sample verification failed: {:?}", v.errors);
    assert!(deletes_ok, "Delete verification failed: {:?}", v.errors);
    assert!(v.is_clean(), "Verifier has errors: {:?}", v.errors);
}
