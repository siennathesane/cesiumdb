//! Autoconfigurator — finds optimal CesiumDB settings for your hardware.
//!
//! Runs a series of benchmarks with different configurations and uses
//! coordinate descent to converge on a global performance optimum.

use std::{
    fs,
    path::PathBuf,
    sync::Arc,
    time::Instant,
};

use crate::{
    compaction::SchedulerConfig,
    state::DbStorageBuilder,
    Db,
    DbOptions,
};

mod scoring;
mod search;
mod workload;

pub use scoring::Score;
pub use search::{
    ConfigPoint,
    CoordinateDescent,
    ParameterSpace,
};

/// Result of a completed autoconfiguration run.
#[derive(Debug, Clone)]
pub struct AutoconfigResult {
    /// Optimal storage configuration.
    pub optimal_storage: DbStorageBuilder,
    /// Optimal scheduler configuration.
    pub optimal_scheduler: SchedulerConfig,
    /// Baseline composite score.
    pub baseline_score: Score,
    /// Optimal composite score.
    pub optimal_score: Score,
    /// Percentage improvement over baseline.
    pub improvement_pct: f64,
    /// Path to the written TOML config file.
    pub config_path: PathBuf,
}

/// Error type for autoconfiguration failures.
#[derive(Debug, thiserror::Error)]
pub enum AutoconfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("No valid configuration found")]
    NoValidConfig,
}

/// Autoconfigurator — bundled performance optimizer.
pub struct Autoconfigurator {
    output_dir: PathBuf,
    quick_mode: bool,
    write_workers: usize,
    write_duration_secs: u64,
    read_duration_secs: u64,
    mixed_duration_secs: u64,
}

impl Autoconfigurator {
    /// Creates a new autoconfigurator that writes results to `output_dir`.
    pub fn new(output_dir: PathBuf) -> Self {
        Self {
            output_dir,
            quick_mode: false,
            write_workers: 10,
            write_duration_secs: 15,
            read_duration_secs: 15,
            mixed_duration_secs: 15,
        }
    }

    /// Enables quick mode (shorter benchmarks, fewer candidates).
    pub fn quick_mode(mut self, yes: bool) -> Self {
        self.quick_mode = yes;
        self
    }

    /// Runs the full autoconfiguration process.
    pub fn run(&self) -> Result<AutoconfigResult, AutoconfigError> {
        fs::create_dir_all(&self.output_dir)?;

        let space = if self.quick_mode {
            ParameterSpace::quick()
        } else {
            ParameterSpace::default()
        };

        let descent = CoordinateDescent::new(space);

        // Baseline configuration
        let mut baseline_storage = DbStorageBuilder::new();
        let baseline_scheduler = SchedulerConfig::default();
        baseline_storage.target_segment_size = baseline_scheduler.target_segment_size;
        let baseline = ConfigPoint {
            storage: baseline_storage,
            scheduler: baseline_scheduler,
        };

        println!("🧪 CesiumDB Autoconfigurator");
        println!("   Mode: {}", if self.quick_mode { "quick" } else { "full" });
        println!("   Benchmarking baseline configuration...");

        let baseline_score = self.benchmark_config(&baseline)?;
        println!(
            "   Baseline score: {:.0} (write={:.0} ops/s, read={:.0} ops/s, mixed={:.0} ops/s)",
            baseline_score.raw,
            baseline_score.write_throughput,
            baseline_score.read_throughput,
            baseline_score.mixed_throughput,
        );

        let mut best = baseline.clone();
        let mut best_score = baseline_score;
        let mut improved = true;
        let mut round = 0;

        while improved && round < 3 {
            improved = false;
            round += 1;
            println!("\n📐 Optimization round {}", round);

            let rounds = descent.iterate(&best);
            for (category, candidates) in rounds {
                if candidates.is_empty() {
                    continue;
                }
                println!(
                    "   Testing {:?} ({} candidates)...",
                    category,
                    candidates.len()
                );

                for (idx, candidate) in candidates.iter().enumerate() {
                    let score = self.benchmark_config(candidate)?;
                    println!(
                        "     [{}/{}] score={:.0} (write={:.0}, read={:.0}, mixed={:.0})",
                        idx + 1,
                        candidates.len(),
                        score.raw,
                        score.write_throughput,
                        score.read_throughput,
                        score.mixed_throughput,
                    );

                    if score.is_better_than(best_score) {
                        best = candidate.clone();
                        best_score = score;
                        improved = true;
                        println!("     ✨ New best!");
                    }
                }
            }
        }

        // Final validation
        println!("\n✅ Final validation run...");
        let final_score = self.benchmark_config(&best)?;
        println!(
            "   Final score: {:.0} (write={:.0}, read={:.0}, mixed={:.0})",
            final_score.raw,
            final_score.write_throughput,
            final_score.read_throughput,
            final_score.mixed_throughput,
        );

        let improvement_pct = if baseline_score.raw > 0.0 {
            ((final_score.raw - baseline_score.raw) / baseline_score.raw) * 100.0
        } else {
            0.0
        };

        // Write TOML config
        let config_path = self.output_dir.join("cesiumdb_autoconfig.toml");
        let toml = format!(
            r#"# CesiumDB Autoconfiguration
# Generated by autoconfigurator
# Baseline score: {baseline:.0}
# Optimal score:  {optimal:.0}
# Improvement:    {improvement:.1}%

[storage]
memtable_size = {memtable_size}
max_memtables = {max_memtables}
target_segment_size = {storage_target_segment_size}
max_concurrent_jobs = {max_concurrent_jobs}

[scheduler]
l0_compaction_trigger = {l0_trigger}
l0_stop_writes_trigger = {l0_stop}
target_segment_size = {scheduler_target_segment_size}
target_file_size_multiplier = {target_file_size_multiplier}
score_threshold = {score_threshold}
"#,
            baseline = baseline_score.raw,
            optimal = final_score.raw,
            improvement = improvement_pct,
            memtable_size = best.storage.memtable_size,
            max_memtables = best.storage.num_memtable_limit,
            storage_target_segment_size = best.storage.target_segment_size,
            max_concurrent_jobs = best.scheduler.max_concurrent_jobs,
            l0_trigger = best.scheduler.l0_compaction_trigger,
            l0_stop = best.scheduler.l0_stop_writes_trigger,
            scheduler_target_segment_size = best.scheduler.target_segment_size,
            target_file_size_multiplier = best.scheduler.target_file_size_multiplier,
            score_threshold = best.scheduler.score_threshold,
        );
        fs::write(&config_path, toml)?;
        println!("\n📝 Config written to: {}", config_path.display());

        Ok(AutoconfigResult {
            optimal_storage: best.storage,
            optimal_scheduler: best.scheduler,
            baseline_score,
            optimal_score: final_score,
            improvement_pct,
            config_path,
        })
    }

    /// Benchmarks a single configuration point.
    fn benchmark_config(&self, point: &ConfigPoint) -> Result<Score, AutoconfigError> {
        let db_path = std::env::temp_dir().join(format!("cesiumdb_autoconfig_{}", std::process::id()));
        fs::create_dir_all(&db_path)?;

        let mut opts = DbOptions::default();
        opts.data_dir(db_path.clone())
            .memtable_size(point.storage.memtable_size)
            .max_memtables(point.storage.num_memtable_limit)
            .target_segment_size(point.storage.target_segment_size)
            .target_file_size_multiplier(point.scheduler.target_file_size_multiplier)
            .scheduler_config(point.scheduler.clone());

        let db = Db::open(opts);

        // Write benchmark
        let write_metrics = workload::run_write_benchmark(
            db.clone(),
            self.write_workers,
            100,
            1024,
            self.write_duration_secs,
        );

        // Pre-fill for read benchmark (1 GiB)
        let value = vec![0u8; 1024];
        for i in 0..1_048_576usize {
            let key = format!("key_{:010}", i).into_bytes();
            let _ = db.put(&key, &value);
        }

        // Read benchmark
        let read_metrics = workload::run_read_benchmark(
            db.clone(),
            self.write_workers,
            1_000_000,
            1024,
            9,
            self.read_duration_secs,
        );

        // Mixed benchmark
        let mixed_metrics = workload::run_mixed_benchmark(
            db.clone(),
            self.write_workers,
            1_000_000,
            1024,
            self.mixed_duration_secs,
        );

        Ok(scoring::compute_score(&write_metrics, &read_metrics, &mixed_metrics))
    }
}
