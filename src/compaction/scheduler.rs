//! Compaction scheduler
//!
//! The scheduler analyzes the current LSM-tree state and generates
//! compaction jobs based on:
//! - Level sizes and scores
//! - L0 file count
//! - Write amplification
//! - Available resources

use crate::compaction::job::{CompactionInput, CompactionJob, CompactionJobType, CompactionOutput};
use crate::levels::{CompactionStrategy, VersionSet};
use crate::segment::Segment;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Configuration for the compaction scheduler
#[derive(Clone)]
pub struct SchedulerConfig {
    /// Trigger L0 compaction when this many files accumulate
    pub l0_compaction_trigger: usize,

    /// Stop writes when L0 reaches this many files
    pub l0_stop_writes_trigger: usize,

    /// Target size for output segments (bytes)
    pub target_segment_size: u64,

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
            l0_compaction_trigger: 4,
            l0_stop_writes_trigger: 8,
            target_segment_size: 64 * 1024 * 1024, // 64 MB
            max_concurrent_jobs: 4,
            score_threshold: 1.0,
        }
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
}

impl CompactionScheduler {
    /// Creates a new scheduler with default configuration
    pub fn new() -> Self {
        Self::with_config(SchedulerConfig::default())
    }

    /// Creates a new scheduler with custom configuration
    pub fn with_config(config: SchedulerConfig) -> Self {
        Self {
            config,
            next_job_id: AtomicU64::new(0),
        }
    }

    /// Picks the next compaction job to run
    ///
    /// Returns the highest-priority job, or None if no compaction is needed.
    pub fn pick_compaction(&self, version: &VersionSet) -> Option<CompactionJob> {
        // Priority order:
        // 1. Trivial moves (zero cost)
        // 2. L0 compaction if over trigger
        // 3. Highest-scoring level compaction

        // Check for trivial moves first
        if let Some(job) = self.find_trivial_move(version) {
            return Some(job);
        }

        // Check L0 compaction
        if version.l0.len() >= self.config.l0_compaction_trigger {
            if let Some(job) = self.create_l0_compaction(version) {
                return Some(job);
            }
        }

        // Check level compactions
        self.find_level_compaction(version)
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
                let has_overlap = next_level.key_ranges.iter().any(|r| r.overlaps(segment_range));

                if !has_overlap {
                    // Found a trivial move!
                    let input = CompactionInput::new(level.level_num, vec![segment.clone()]);

                    let output = CompactionOutput::new(
                        next_level_num,
                        self.config.target_segment_size,
                    );

                    let job_id = self.next_job_id.fetch_add(1, Ordering::SeqCst);

                    return Some(CompactionJob::new(
                        job_id,
                        CompactionJobType::TrivialMove,
                        input,
                        None,
                        output,
                    ));
                }
            }
        }

        None
    }

    /// Creates an L0 compaction job
    ///
    /// Selects multiple L0 segments and their overlapping L1 segments.
    fn create_l0_compaction(&self, version: &VersionSet) -> Option<CompactionJob> {
        if version.l0.is_empty() {
            return None;
        }

        // For now, compact all L0 files at once
        // TODO: More sophisticated selection (e.g., size-based, oldest-first)
        let l0_segments = version.l0.clone();

        let input = CompactionInput::new(0, l0_segments);

        // Find overlapping L1 segments
        let next_level_input = if !version.levels.is_empty() {
            let l1 = &version.levels[0];

            // Find all L1 segments that overlap with the L0 range
            let overlapping = l1
                .segments
                .iter()
                .filter(|seg| {
                    // Check if segment overlaps with L0 range
                    if let Some(range) = l1.key_ranges.iter().find(|r| r.segment_id == seg.id()) {
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
                Some(CompactionInput::new(1, overlapping))
            }
        } else {
            None
        };

        let output = CompactionOutput::new(1, self.config.target_segment_size);

        let job_id = self.next_job_id.fetch_add(1, Ordering::SeqCst);

        Some(CompactionJob::new(
            job_id,
            CompactionJobType::L0Compaction,
            input,
            next_level_input,
            output,
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
    fn create_level_compaction(&self, version: &VersionSet, level_num: u8) -> Option<CompactionJob> {
        let level_idx = level_num as usize - 1;
        if level_idx >= version.levels.len() {
            return None;
        }

        let level = &version.levels[level_idx];

        // For leveled compaction: pick first segment
        // TODO: More sophisticated selection (round-robin, largest file, etc.)
        if level.segments.is_empty() {
            return None;
        }

        let segment = level.segments[0].clone();
        let segment_range = level
            .key_ranges
            .iter()
            .find(|r| r.segment_id == segment.id())?;

        let input = CompactionInput::new(level_num, vec![segment]);

        // Find overlapping segments in next level
        let next_level_num = level_num + 1;
        let next_level_idx = next_level_num as usize - 1;

        let next_level_input = if next_level_idx < version.levels.len() {
            let next_level = &version.levels[next_level_idx];

            let overlapping = next_level
                .segments
                .iter()
                .filter(|seg| {
                    if let Some(range) = next_level.key_ranges.iter().find(|r| r.segment_id == seg.id()) {
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
                Some(CompactionInput::new(next_level_num, overlapping))
            }
        } else {
            None
        };

        let output = CompactionOutput::new(next_level_num, self.config.target_segment_size);

        let job_id = self.next_job_id.fetch_add(1, Ordering::SeqCst);

        Some(CompactionJob::new(
            job_id,
            CompactionJobType::LevelCompaction,
            input,
            next_level_input,
            output,
        ))
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

impl Default for CompactionScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::levels::VersionSet;

    #[test]
    fn test_scheduler_creation() {
        let scheduler = CompactionScheduler::new();
        assert_eq!(scheduler.config.l0_compaction_trigger, 4);
        assert_eq!(scheduler.config.max_concurrent_jobs, 4);
    }

    #[test]
    fn test_scheduler_custom_config() {
        let config = SchedulerConfig {
            l0_compaction_trigger: 8,
            l0_stop_writes_trigger: 16,
            target_segment_size: 128 * 1024 * 1024,
            max_concurrent_jobs: 8,
            score_threshold: 2.0,
        };

        let scheduler = CompactionScheduler::with_config(config);
        assert_eq!(scheduler.config.l0_compaction_trigger, 8);
        assert_eq!(scheduler.config.target_segment_size, 128 * 1024 * 1024);
    }

    #[test]
    fn test_no_compaction_needed_empty_version() {
        let scheduler = CompactionScheduler::new();
        let version = VersionSet::new(0, 7);

        let job = scheduler.pick_compaction(&version);
        assert!(job.is_none(), "empty version should not need compaction");
    }

    #[test]
    fn test_should_stop_writes() {
        let scheduler = CompactionScheduler::new();
        let version = VersionSet::new(0, 7);

        // Empty L0 should not stop writes
        assert!(!scheduler.should_stop_writes(&version));

        // We can't easily add segments without full infrastructure,
        // but we can test the threshold logic
        assert_eq!(scheduler.config.l0_stop_writes_trigger, 8);
    }

    #[test]
    fn test_job_id_increments() {
        let scheduler = CompactionScheduler::new();

        let id1 = scheduler.next_job_id.load(Ordering::SeqCst);
        scheduler.next_job_id.fetch_add(1, Ordering::SeqCst);
        let id2 = scheduler.next_job_id.load(Ordering::SeqCst);

        assert_eq!(id2, id1 + 1);
    }
}
