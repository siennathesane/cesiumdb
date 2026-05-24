//! Coordinate descent search for optimal configuration.

use crate::{
    compaction::SchedulerConfig,
    state::DbStorageBuilder,
};

/// A single point in the configuration space.
#[derive(Debug, Clone)]
pub struct ConfigPoint {
    pub storage: DbStorageBuilder,
    pub scheduler: SchedulerConfig,
}

/// Parameter category for coordinate descent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterCategory {
    Memory,
    Compaction,
    Threading,
}

/// Generates candidate values for each parameter category.
pub struct ParameterSpace {
    pub memtable_sizes: Vec<u64>,
    pub max_memtables: Vec<u64>,
    pub target_segment_sizes: Vec<u64>,
    pub target_file_size_multipliers: Vec<u64>,
    pub l0_triggers: Vec<usize>,
    pub l0_stop_triggers: Vec<usize>,
    pub max_concurrent_jobs: Vec<usize>,
    pub score_thresholds: Vec<f64>,
    pub min_workers: Vec<usize>,
    pub max_workers: Vec<usize>,
}

impl Default for ParameterSpace {
    fn default() -> Self {
        Self {
            memtable_sizes: vec![
                16 * 1024 * 1024,
                32 * 1024 * 1024,
                64 * 1024 * 1024,
                128 * 1024 * 1024,
            ],
            max_memtables: vec![2, 4, 8, 16],
            target_segment_sizes: vec![32 * 1024 * 1024, 64 * 1024 * 1024, 128 * 1024 * 1024],
            target_file_size_multipliers: vec![1, 2, 4],
            l0_triggers: vec![4, 8, 16, 32],
            l0_stop_triggers: vec![8, 16, 32, 64],
            max_concurrent_jobs: vec![2, 4, 8, 16],
            score_thresholds: vec![0.5, 1.0, 2.0],
            min_workers: vec![1, 2, 4],
            max_workers: vec![2, 4, 8],
        }
    }
}

impl ParameterSpace {
    /// Returns a reduced space for quick mode (fewer candidates).
    pub fn quick() -> Self {
        Self {
            memtable_sizes: vec![32 * 1024 * 1024, 64 * 1024 * 1024, 128 * 1024 * 1024],
            max_memtables: vec![4, 8],
            target_segment_sizes: vec![64 * 1024 * 1024, 128 * 1024 * 1024],
            target_file_size_multipliers: vec![1, 2],
            l0_triggers: vec![8, 16],
            l0_stop_triggers: vec![16, 32],
            max_concurrent_jobs: vec![4, 8],
            score_thresholds: vec![1.0],
            min_workers: vec![2, 4],
            max_workers: vec![4, 8],
        }
    }
}

/// Coordinate descent search state.
pub struct CoordinateDescent {
    space: ParameterSpace,
}

impl CoordinateDescent {
    pub fn new(space: ParameterSpace) -> Self {
        Self { space }
    }

    /// Generates all candidate configurations for a single coordinate descent
    /// iteration, given the current best configuration.
    ///
    /// Returns `(category, configs)` tuples.  Each category varies one
    /// parameter while keeping the rest fixed at `base`.
    pub fn iterate(&self, base: &ConfigPoint) -> Vec<(ParameterCategory, Vec<ConfigPoint>)> {
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let mut rounds = Vec::new();

        // Memory category
        let mut memory_cfgs = Vec::new();
        for &size in &self.space.memtable_sizes {
            if size == base.storage.memtable_size {
                continue;
            }
            let mut cfg = base.clone();
            cfg.storage.memtable_size = size;
            memory_cfgs.push(cfg);
        }
        for &limit in &self.space.max_memtables {
            if limit == base.storage.num_memtable_limit {
                continue;
            }
            let mut cfg = base.clone();
            cfg.storage.num_memtable_limit = limit;
            memory_cfgs.push(cfg);
        }
        rounds.push((ParameterCategory::Memory, memory_cfgs));

        // Compaction category
        let mut compaction_cfgs = Vec::new();
        for &size in &self.space.target_segment_sizes {
            if size == base.scheduler.target_segment_size {
                continue;
            }
            let mut cfg = base.clone();
            cfg.scheduler.target_segment_size = size;
            cfg.storage.target_segment_size = size;
            compaction_cfgs.push(cfg);
        }
        for &mult in &self.space.target_file_size_multipliers {
            if mult == base.scheduler.target_file_size_multiplier {
                continue;
            }
            let mut cfg = base.clone();
            cfg.scheduler.target_file_size_multiplier = mult;
            compaction_cfgs.push(cfg);
        }
        for &trigger in &self.space.l0_triggers {
            if trigger == base.scheduler.l0_compaction_trigger {
                continue;
            }
            let mut cfg = base.clone();
            cfg.scheduler.l0_compaction_trigger = trigger;
            compaction_cfgs.push(cfg);
        }
        for &stop in &self.space.l0_stop_triggers {
            if stop == base.scheduler.l0_stop_writes_trigger {
                continue;
            }
            let mut cfg = base.clone();
            cfg.scheduler.l0_stop_writes_trigger = stop;
            compaction_cfgs.push(cfg);
        }
        for &max_jobs in &self.space.max_concurrent_jobs {
            if max_jobs == base.scheduler.max_concurrent_jobs {
                continue;
            }
            let mut cfg = base.clone();
            cfg.scheduler.max_concurrent_jobs = max_jobs;
            compaction_cfgs.push(cfg);
        }
        for &threshold in &self.space.score_thresholds {
            if (threshold - base.scheduler.score_threshold).abs() < f64::EPSILON {
                continue;
            }
            let mut cfg = base.clone();
            cfg.scheduler.score_threshold = threshold;
            compaction_cfgs.push(cfg);
        }
        rounds.push((ParameterCategory::Compaction, compaction_cfgs));

        // Threading category
        let mut threading_cfgs = Vec::new();
        for &min_w in &self.space.min_workers {
            let mut cfg = base.clone();
            cfg.scheduler.max_concurrent_jobs = min_w.max(cfg.scheduler.max_concurrent_jobs);
            threading_cfgs.push(cfg);
        }
        for &max_w in &self.space.max_workers {
            let mut cfg = base.clone();
            if max_w >= num_cpus / 2 {
                cfg.scheduler.max_concurrent_jobs = max_w;
                threading_cfgs.push(cfg);
            }
        }
        rounds.push((ParameterCategory::Threading, threading_cfgs));

        rounds
    }
}
