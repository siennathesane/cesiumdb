//! Compaction scheduler
//!
//! The scheduler analyzes the current LSM-tree state and generates
//! compaction jobs based on:
//! - Level sizes and scores
//! - L0 file count
//! - Write amplification
//! - Available resources

use std::{
    collections::{
        HashMap,
        HashSet,
    },
    sync::{
        Arc,
        atomic::{
            AtomicU64,
            Ordering,
        },
    },
};

use parking_lot::RwLock;

use crate::{
    compaction::job::{
        CompactionInput,
        CompactionJob,
        CompactionJobType,
        CompactionOutput,
    },
    levels::{
        CompactionStrategy,
        KeyRange,
        VersionSet,
    },
    segment::Segment,
    version::VersionManager,
};

/// Configuration for the compaction scheduler
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Trigger L0 compaction when this many files accumulate
    pub l0_compaction_trigger: usize,

    /// Stop writes when L0 reaches this many files
    pub l0_stop_writes_trigger: usize,

    /// Target size for output segments (bytes)
    pub target_segment_size: u64,

    /// Multiplier for target segment size per level.
    ///
    /// Level N target size = `target_segment_size * multiplier^(N-1)`.
    /// Default is 1 (same size for all levels).
    pub target_file_size_multiplier: u64,

    /// Maximum number of concurrent compaction jobs
    pub max_concurrent_jobs: usize,

    /// Minimum score to trigger a compaction
    ///
    /// Compactions with score < threshold won't run automatically
    pub score_threshold: f64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            l0_compaction_trigger: 8, // Smaller batches for faster, more frequent compactions
            l0_stop_writes_trigger: 16, // Stall writes if L0 gets too far ahead
            target_segment_size: 64 * 1024 * 1024, // 64 MB
            target_file_size_multiplier: 1,
            max_concurrent_jobs: 8,
            score_threshold: 1.0,
        }
    }
}

impl SchedulerConfig {
    /// Computes the target segment size for a given output level.
    ///
    /// Level 1 uses `target_segment_size` directly.
    /// Each subsequent level multiplies by `target_file_size_multiplier`.
    pub fn target_segment_size_for_level(&self, level: u8) -> u64 {
        if level <= 1 || self.target_file_size_multiplier <= 1 {
            return self.target_segment_size;
        }
        let exponent = (level - 1) as u32;
        self.target_segment_size
            .saturating_mul(self.target_file_size_multiplier.saturating_pow(exponent))
    }
}

/// Compaction scheduler
///
/// Analyzes the version set and generates compaction jobs.
pub struct CompactionScheduler {
    /// Configuration
    config: SchedulerConfig,

    /// Monotonic job ID counter
    next_job_id: AtomicU64,

    /// Round-robin tracker: maps level_num → last compacted segment_id
    last_compacted: RwLock<HashMap<u8, u64>>,

    /// Version manager for allocating segment IDs
    version_manager: Arc<VersionManager>,
}

impl CompactionScheduler {
    /// Creates a new scheduler with default configuration
    pub fn new(version_manager: Arc<VersionManager>) -> Self {
        Self::with_config(SchedulerConfig::default(), version_manager)
    }

    /// Creates a new scheduler with custom configuration
    pub fn with_config(config: SchedulerConfig, version_manager: Arc<VersionManager>) -> Self {
        Self {
            config,
            next_job_id: AtomicU64::new(0),
            last_compacted: RwLock::new(HashMap::new()),
            version_manager,
        }
    }

    /// Picks the next compaction job to run
    ///
    /// Returns the highest-priority job, or None if no compaction is needed.
    pub fn pick_compaction(&self, version: &VersionSet) -> Option<CompactionJob> {
        self.pick_compactions(version, &HashSet::new(), 1)
            .into_iter()
            .next()
    }

    /// Picks up to `max_jobs` non-conflicting compaction jobs
    ///
    /// Filters out jobs whose input or next-level segments are in the
    /// `in_flight` set.  L0 compactions are limited to at most one.
    /// Level compactions on different levels (or non-overlapping
    /// segments within a leveled level) may be returned together.
    pub fn pick_compactions(
        &self,
        version: &VersionSet,
        in_flight: &HashSet<u64>,
        max_jobs: usize,
    ) -> Vec<CompactionJob> {
        let mut jobs = Vec::new();

        // 1. Collect all trivial moves that don't conflict
        for level in &version.levels {
            if level.level_num as usize >= version.num_levels() - 1 {
                continue;
            }
            if level.strategy.allows_overlaps() {
                continue;
            }
            let next_level_num = level.level_num + 1;
            let next_level_idx = next_level_num as usize - 1;
            if next_level_idx >= version.levels.len() {
                continue;
            }
            let next_level = &version.levels[next_level_idx];

            for segment in &level.segments {
                if in_flight.contains(&segment.id()) {
                    continue;
                }
                let segment_range = match level
                    .key_ranges
                    .iter()
                    .find(|r| r.segment_id == segment.id())
                {
                    | Some(r) => r,
                    | None => continue,
                };
                let has_overlap = next_level
                    .key_ranges
                    .iter()
                    .any(|r| r.overlaps(segment_range));
                if !has_overlap {
                    let input = CompactionInput::with_key_range(
                        level.level_num,
                        vec![segment.clone()],
                        &level.key_ranges,
                    );
                    let output = CompactionOutput::new(
                        next_level_num,
                        self.config.target_segment_size_for_level(next_level_num),
                    );
                    let job_id = self.next_job_id.fetch_add(1, Ordering::SeqCst);
                    jobs.push(CompactionJob::new(
                        job_id,
                        CompactionJobType::TrivialMove,
                        input,
                        None,
                        output,
                        vec![],
                    ));
                    if jobs.len() >= max_jobs {
                        return jobs;
                    }
                }
            }
        }

        // 2. At most one L0 compaction
        if version.l0.len() >= self.config.l0_compaction_trigger {
            let l0_conflicts = version.l0.iter().any(|s| in_flight.contains(&s.id()));
            if !l0_conflicts {
                if let Some(job) = self.create_l0_compaction(version) {
                    // Verify none of the chosen L0 segments are in-flight
                    let conflicts = job
                        .input
                        .segments
                        .iter()
                        .any(|s| in_flight.contains(&s.id()));
                    if !conflicts {
                        jobs.push(job);
                        if jobs.len() >= max_jobs {
                            return jobs;
                        }
                    }
                }
            }
        }

        // 3. Level compactions — try every level above threshold, sorted by score
        let mut level_scores: Vec<_> = version
            .levels
            .iter()
            .map(|l| (l.level_num, l.score()))
            .filter(|(_, s)| *s > self.config.score_threshold)
            .collect();
        level_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (level_num, _) in level_scores {
            if jobs.len() >= max_jobs {
                break;
            }
            if let Some(job) = self.create_level_compaction(version, level_num) {
                let conflicts = job
                    .input
                    .segments
                    .iter()
                    .any(|s| in_flight.contains(&s.id())) ||
                    job.next_level_input.as_ref().map_or(false, |next| {
                        next.segments.iter().any(|s| in_flight.contains(&s.id()))
                    });
                if !conflicts {
                    jobs.push(job);
                }
            }
        }

        jobs
    }

    /// Finds a trivial move opportunity
    ///
    /// A segment can be trivially moved if it doesn't overlap with
    /// any segments in the next level.
    fn find_trivial_move(&self, version: &VersionSet) -> Option<CompactionJob> {
        // Check each level for trivial move candidates
        for level in &version.levels {
            // Can't move from the last level
            if level.level_num as usize >= version.num_levels() - 1 {
                continue;
            }

            // Leveled compaction only (tiered allows overlaps)
            if level.strategy.allows_overlaps() {
                continue;
            }

            // Look for segments that don't overlap with next level
            let next_level_num = level.level_num + 1;
            let next_level = &version.levels[next_level_num as usize - 1];

            for segment in &level.segments {
                let segment_range = level
                    .key_ranges
                    .iter()
                    .find(|r| r.segment_id == segment.id())
                    .expect("segment should have key range");

                // Check if this segment overlaps with any segment in next level
                let has_overlap = next_level
                    .key_ranges
                    .iter()
                    .any(|r| r.overlaps(segment_range));

                if !has_overlap {
                    // Found a trivial move!
                    let input = CompactionInput::with_key_range(
                        level.level_num,
                        vec![segment.clone()],
                        &level.key_ranges,
                    );

                    let output = CompactionOutput::new(
                        next_level_num,
                        self.config.target_segment_size_for_level(next_level_num),
                    );

                    let job_id = self.next_job_id.fetch_add(1, Ordering::SeqCst);

                    // Trivial moves don't create new segments, so allocate 0 IDs
                    let allocated_ids: Vec<u64> = vec![];

                    return Some(CompactionJob::new(
                        job_id,
                        CompactionJobType::TrivialMove,
                        input,
                        None,
                        output,
                        allocated_ids,
                    ));
                }
            }
        }

        None
    }

    /// Creates an L0 compaction job
    ///
    /// Selects L0 segments oldest-first (by ID, since IDs are monotonically
    /// increasing). Takes at most `l0_compaction_trigger` segments per
    /// batch to avoid oversized compactions.
    fn create_l0_compaction(&self, version: &VersionSet) -> Option<CompactionJob> {
        if version.l0.is_empty() {
            return None;
        }

        // Sort L0 segments by ID (oldest first, since IDs increase monotonically)
        let mut l0_segments = version.l0.clone();
        l0_segments.sort_by_key(|s| s.id());

        // Take at most l0_compaction_trigger segments per batch
        let batch_size = self.config.l0_compaction_trigger.min(l0_segments.len());
        l0_segments.truncate(batch_size);

        // Get corresponding key ranges for the L0 segments
        let l0_key_ranges: Vec<KeyRange> = l0_segments
            .iter()
            .filter_map(|seg| {
                version
                    .l0_key_ranges
                    .iter()
                    .find(|r| r.segment_id == seg.id())
                    .cloned()
            })
            .collect();

        let input = CompactionInput::with_key_range(0, l0_segments, &l0_key_ranges);

        // Find overlapping L1 segments only when L1 uses leveled compaction.
        // Tiered levels allow overlapping files, so L0 compaction should just
        // merge L0 segments and place the output in L1 without rewriting L1.
        let next_level_input = if !version.levels.is_empty() {
            let l1 = &version.levels[0];

            match l1.strategy {
                | CompactionStrategy::Leveled { .. } => {
                    // Leveled: include overlapping L1 segments
                    let overlapping = l1
                        .segments
                        .iter()
                        .filter(|seg| {
                            if let Some(range) =
                                l1.key_ranges.iter().find(|r| r.segment_id == seg.id())
                            {
                                input.key_range.overlaps(range)
                            } else {
                                false
                            }
                        })
                        .cloned()
                        .collect::<Vec<_>>();

                    if overlapping.is_empty() {
                        None
                    } else {
                        Some(CompactionInput::with_key_range(
                            1,
                            overlapping,
                            &l1.key_ranges,
                        ))
                    }
                },
                | _ => {
                    // Tiered / Universal: do not rewrite L1
                    None
                },
            }
        } else {
            None
        };

        let output = CompactionOutput::new(1, self.config.target_segment_size_for_level(1));

        let job_id = self.next_job_id.fetch_add(1, Ordering::SeqCst);

        // Pre-allocate segment IDs for output segments (estimate 1 for now)
        let num_output_segments = 1;
        let allocated_ids: Vec<u64> = (0..num_output_segments)
            .map(|_| self.version_manager.next_segment_id())
            .collect();

        Some(CompactionJob::new(
            job_id,
            CompactionJobType::L0Compaction,
            input,
            next_level_input,
            output,
            allocated_ids,
        ))
    }

    /// Finds the level with the highest compaction score
    fn find_level_compaction(&self, version: &VersionSet) -> Option<CompactionJob> {
        // Find level with highest score above threshold
        let mut best_level: Option<(u8, f64)> = None;

        for level in &version.levels {
            let score = level.score();

            if score > self.config.score_threshold {
                if let Some((_, best_score)) = best_level {
                    if score > best_score {
                        best_level = Some((level.level_num, score));
                    }
                } else {
                    best_level = Some((level.level_num, score));
                }
            }
        }

        if let Some((level_num, _)) = best_level {
            self.create_level_compaction(version, level_num)
        } else {
            None
        }
    }

    /// Creates a level compaction job
    ///
    /// For leveled levels: uses round-robin selection with overlap checking.
    /// For tiered levels: compacts ALL segments in the level at once.
    fn create_level_compaction(
        &self,
        version: &VersionSet,
        level_num: u8,
    ) -> Option<CompactionJob> {
        let level_idx = level_num as usize - 1;
        if level_idx >= version.levels.len() {
            return None;
        }

        let level = &version.levels[level_idx];

        if level.segments.is_empty() {
            return None;
        }

        match level.strategy {
            | CompactionStrategy::Tiered { .. } | CompactionStrategy::Universal { .. } => {
                // Tiered/Universal: compact ALL segments in the level
                let input = CompactionInput::with_key_range(
                    level_num,
                    level.segments.clone(),
                    &level.key_ranges,
                );

                let next_level_num = level_num + 1;
                let next_level_idx = next_level_num as usize - 1;

                // For tiered output levels, we don't need to include next-level
                // segments since overlaps are allowed.
                let next_level_input = if next_level_idx < version.levels.len() {
                    let next_level = &version.levels[next_level_idx];
                    match next_level.strategy {
                        | CompactionStrategy::Leveled { .. } => {
                            // If next level is leveled, find overlapping segments
                            let overlapping = next_level
                                .segments
                                .iter()
                                .filter(|seg| {
                                    if let Some(range) = next_level
                                        .key_ranges
                                        .iter()
                                        .find(|r| r.segment_id == seg.id())
                                    {
                                        input.key_range.overlaps(range)
                                    } else {
                                        false
                                    }
                                })
                                .cloned()
                                .collect::<Vec<_>>();

                            if overlapping.is_empty() {
                                None
                            } else {
                                Some(CompactionInput::with_key_range(
                                    next_level_num,
                                    overlapping,
                                    &next_level.key_ranges,
                                ))
                            }
                        },
                        | _ => None,
                    }
                } else {
                    None
                };

                let output = CompactionOutput::new(
                    next_level_num,
                    self.config.target_segment_size_for_level(next_level_num),
                );
                let job_id = self.next_job_id.fetch_add(1, Ordering::SeqCst);
                let num_output_segments = 1;
                let allocated_ids: Vec<u64> = (0..num_output_segments)
                    .map(|_| self.version_manager.next_segment_id())
                    .collect();

                Some(CompactionJob::new(
                    job_id,
                    CompactionJobType::LevelCompaction,
                    input,
                    next_level_input,
                    output,
                    allocated_ids,
                ))
            },
            | CompactionStrategy::Leveled { .. } => {
                // Leveled: round-robin + overlap checking
                let segment = {
                    let last_compacted = self.last_compacted.read();
                    let last_id = last_compacted.get(&level_num).copied();

                    match last_id {
                        | Some(id) => level
                            .segments
                            .iter()
                            .find(|s| s.id() > id)
                            .unwrap_or(&level.segments[0])
                            .clone(),
                        | None => level.segments[0].clone(),
                    }
                };

                {
                    let mut last_compacted = self.last_compacted.write();
                    last_compacted.insert(level_num, segment.id());
                }
                let segment_range = match level
                    .key_ranges
                    .iter()
                    .find(|r| r.segment_id == segment.id())
                {
                    | Some(r) => r,
                    | None => return None,
                };

                let input =
                    CompactionInput::with_key_range(level_num, vec![segment], &level.key_ranges);

                let next_level_num = level_num + 1;
                let next_level_idx = next_level_num as usize - 1;

                let next_level_input = if next_level_idx < version.levels.len() {
                    let next_level = &version.levels[next_level_idx];

                    let overlapping = next_level
                        .segments
                        .iter()
                        .filter(|seg| {
                            if let Some(range) = next_level
                                .key_ranges
                                .iter()
                                .find(|r| r.segment_id == seg.id())
                            {
                                segment_range.overlaps(range)
                            } else {
                                false
                            }
                        })
                        .cloned()
                        .collect::<Vec<_>>();

                    if overlapping.is_empty() {
                        None
                    } else {
                        Some(CompactionInput::with_key_range(
                            next_level_num,
                            overlapping,
                            &next_level.key_ranges,
                        ))
                    }
                } else {
                    None
                };

                let output = CompactionOutput::new(
                    next_level_num,
                    self.config.target_segment_size_for_level(next_level_num),
                );
                let job_id = self.next_job_id.fetch_add(1, Ordering::SeqCst);
                let num_output_segments = 1;
                let allocated_ids: Vec<u64> = (0..num_output_segments)
                    .map(|_| self.version_manager.next_segment_id())
                    .collect();

                Some(CompactionJob::new(
                    job_id,
                    CompactionJobType::LevelCompaction,
                    input,
                    next_level_input,
                    output,
                    allocated_ids,
                ))
            },
        }
    }

    /// Checks if writes should be stopped due to L0 file count
    pub fn should_stop_writes(&self, version: &VersionSet) -> bool {
        version.l0.len() >= self.config.l0_stop_writes_trigger
    }

    /// Returns the current configuration
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::levels::VersionSet;

    #[test]
    fn test_scheduler_creation() {
        let version_manager = Arc::new(VersionManager::new(7));
        let scheduler = CompactionScheduler::new(version_manager);
        assert_eq!(scheduler.config.l0_compaction_trigger, 8);
        assert_eq!(scheduler.config.max_concurrent_jobs, 8);
    }

    #[test]
    fn test_scheduler_custom_config() {
        let config = SchedulerConfig {
            l0_compaction_trigger: 8,
            l0_stop_writes_trigger: 16,
            target_segment_size: 128 * 1024 * 1024,
            target_file_size_multiplier: 1,
            max_concurrent_jobs: 8,
            score_threshold: 2.0,
        };

        let version_manager = Arc::new(VersionManager::new(7));
        let scheduler = CompactionScheduler::with_config(config, version_manager);
        assert_eq!(scheduler.config.l0_compaction_trigger, 8);
        assert_eq!(scheduler.config.target_segment_size, 128 * 1024 * 1024);
    }

    #[test]
    fn test_no_compaction_needed_empty_version() {
        let version_manager = Arc::new(VersionManager::new(7));
        let scheduler = CompactionScheduler::new(version_manager);
        let version = VersionSet::new(0, 7);

        let job = scheduler.pick_compaction(&version);
        assert!(job.is_none(), "empty version should not need compaction");
    }

    #[test]
    fn test_should_stop_writes() {
        let version_manager = Arc::new(VersionManager::new(7));
        let scheduler = CompactionScheduler::new(version_manager);
        let version = VersionSet::new(0, 7);

        // Empty L0 should not stop writes
        assert!(!scheduler.should_stop_writes(&version));

        // We can't easily add segments without full infrastructure,
        // but we can test the threshold logic
        assert_eq!(scheduler.config.l0_stop_writes_trigger, 16);
    }

    #[test]
    fn test_job_id_increments() {
        let version_manager = Arc::new(VersionManager::new(7));
        let scheduler = CompactionScheduler::new(version_manager);

        let id1 = scheduler.next_job_id.load(Ordering::SeqCst);
        scheduler.next_job_id.fetch_add(1, Ordering::SeqCst);
        let id2 = scheduler.next_job_id.load(Ordering::SeqCst);

        assert_eq!(id2, id1 + 1);
    }

    #[test]
    fn test_target_segment_size_for_level() {
        let config = SchedulerConfig {
            target_segment_size: 64 * 1024 * 1024,
            target_file_size_multiplier: 2,
            ..Default::default()
        };

        assert_eq!(config.target_segment_size_for_level(1), 64 * 1024 * 1024);
        assert_eq!(config.target_segment_size_for_level(2), 128 * 1024 * 1024);
        assert_eq!(config.target_segment_size_for_level(3), 256 * 1024 * 1024);
        assert_eq!(config.target_segment_size_for_level(4), 512 * 1024 * 1024);
    }

    #[test]
    fn test_target_segment_size_multiplier_one() {
        let config = SchedulerConfig {
            target_segment_size: 64 * 1024 * 1024,
            target_file_size_multiplier: 1,
            ..Default::default()
        };

        assert_eq!(config.target_segment_size_for_level(1), 64 * 1024 * 1024);
        assert_eq!(config.target_segment_size_for_level(5), 64 * 1024 * 1024);
    }
}
