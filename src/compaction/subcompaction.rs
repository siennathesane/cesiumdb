//! Subcompaction - range splitting for parallel execution
#![allow(unused)]
//! Splits large compaction jobs into smaller range-based sub-jobs that can
//! be executed in parallel, dramatically improving throughput for large
//! compactions.

use std::cmp::min;

use crate::{
    compaction::job::{
        CompactionInput,
        CompactionJob,
        CompactionJobType,
        CompactionOutput,
    },
    levels::KeyRange,
};

/// Configuration for subcompaction
#[derive(Debug, Clone, Copy)]
pub struct SubcompactionConfig {
    /// Minimum size (in bytes) before splitting
    pub min_size_for_split: u64,

    /// Target number of subcompactions
    pub target_subcompactions: usize,

    /// Minimum size per subcompaction
    pub min_subcompaction_size: u64,

    /// Maximum number of subcompactions
    pub max_subcompactions: usize,
}

impl Default for SubcompactionConfig {
    fn default() -> Self {
        Self {
            min_size_for_split: 64 * 1024 * 1024, // 64MB
            target_subcompactions: 4,
            min_subcompaction_size: 16 * 1024 * 1024, // 16MB
            max_subcompactions: 8,
        }
    }
}

/// A sub-job created from splitting a compaction
#[derive(Clone)]
pub struct SubcompactionJob {
    /// Parent job ID
    pub parent_id: u64,

    /// Index of this subjob (0-based)
    pub index: usize,

    /// Key range for this subjob
    pub key_range: KeyRange,

    /// Input for this subjob
    pub input: CompactionInput,

    /// Next level input (if any)
    pub next_level_input: Option<CompactionInput>,

    /// Output configuration
    pub output: CompactionOutput,
}

/// Subcompaction planner
#[derive(Clone)]
pub struct SubcompactionPlanner {
    config: SubcompactionConfig,
}

impl SubcompactionPlanner {
    /// Creates a new planner with default configuration
    pub fn new() -> Self {
        Self::with_config(SubcompactionConfig::default())
    }

    /// Creates a new planner with custom configuration
    pub fn with_config(config: SubcompactionConfig) -> Self {
        Self { config }
    }

    /// Determines if a job should be split into subcompactions
    pub fn should_split(&self, job: &CompactionJob) -> bool {
        // Don't split flush operations
        if matches!(job.job_type, CompactionJobType::Flush) {
            return false;
        }

        // Don't split trivial moves
        if matches!(job.job_type, CompactionJobType::TrivialMove) {
            return false;
        }

        // Check total size
        let total_size = job.input.total_size +
            job.next_level_input
                .as_ref()
                .map(|i| i.total_size)
                .unwrap_or(0);

        total_size >= self.config.min_size_for_split
    }

    /// Splits a compaction job into sub-jobs based on key ranges
    ///
    /// Returns a vector of subjobs if splitting is beneficial, otherwise
    /// returns None.
    pub fn split(&self, job: &CompactionJob) -> Option<Vec<SubcompactionJob>> {
        if !self.should_split(job) {
            return None;
        }

        // Calculate total size
        let total_size = job.input.total_size +
            job.next_level_input
                .as_ref()
                .map(|i| i.total_size)
                .unwrap_or(0);

        // Determine number of subcompactions
        let num_subcompactions = self.calculate_num_subcompactions(total_size);

        if num_subcompactions <= 1 {
            return None;
        }

        // Split the key range
        let subjob_ranges = self.split_key_range(&job.input.key_range, num_subcompactions);

        // Create subjobs — each subjob gets ALL input segments.
        // The segment iterators use key-range bounds so each subcompaction
        // only reads and writes keys within its assigned range.
        let subjobs: Vec<_> = subjob_ranges
            .into_iter()
            .enumerate()
            .map(|(index, range)| SubcompactionJob {
                parent_id: job.id,
                index,
                key_range: range.clone(),
                input: CompactionInput {
                    level: job.input.level,
                    segments: job.input.segments.clone(),
                    key_range: range.clone(),
                    total_size: total_size / num_subcompactions as u64,
                },
                next_level_input: job.next_level_input.as_ref().map(|input| CompactionInput {
                    level: input.level,
                    segments: input.segments.clone(),
                    key_range: range.clone(),
                    total_size: input.total_size / num_subcompactions as u64,
                }),
                output: CompactionOutput::new(
                    job.output.level,
                    job.output.target_segment_size / num_subcompactions as u64,
                ),
            })
            .collect();

        Some(subjobs)
    }

    /// Calculates the optimal number of subcompactions
    fn calculate_num_subcompactions(&self, total_size: u64) -> usize {
        // Start with target number
        let mut num = self.config.target_subcompactions;

        // Ensure each subjob is at least min size
        let max_from_size = (total_size / self.config.min_subcompaction_size) as usize;
        num = min(num, max_from_size);

        // Cap at maximum
        num = min(num, self.config.max_subcompactions);

        // Must be at least 1
        num.max(1)
    }

    /// Splits a key range into N roughly equal sub-ranges
    fn split_key_range(&self, range: &KeyRange, n: usize) -> Vec<KeyRange> {
        if n <= 1 {
            return vec![range.clone()];
        }

        // For simplicity, we do byte-level splitting
        // In production, you'd want smarter splitting based on actual key distribution
        let start = &range.start;
        let end = &range.end;

        if start.is_empty() || end.is_empty() {
            // Can't split empty ranges meaningfully
            return vec![range.clone()];
        }

        let mut ranges = Vec::with_capacity(n);

        // Simple approach: divide the byte range
        // This works well for uniformly distributed keys
        for i in 0..n {
            let sub_start = if i == 0 {
                start.clone()
            } else {
                self.interpolate_key(start, end, i as f64 / n as f64)
            };

            let sub_end = if i == n - 1 {
                end.clone()
            } else {
                self.interpolate_key(start, end, (i + 1) as f64 / n as f64)
            };

            ranges.push(KeyRange::new(sub_start, sub_end, range.segment_id));
        }

        ranges
    }

    /// Interpolates a key between start and end at the given ratio
    fn interpolate_key(&self, start: &[u8], end: &[u8], ratio: f64) -> Vec<u8> {
        let len = start.len().max(end.len());
        let mut result = Vec::with_capacity(len);

        for i in 0..len {
            let start_byte = if i < start.len() {
                start[i] as f64
            } else {
                0.0
            };
            let end_byte = if i < end.len() { end[i] as f64 } else { 255.0 };

            let interpolated = start_byte + (end_byte - start_byte) * ratio;
            result.push(interpolated as u8);
        }

        result
    }
}

impl Default for SubcompactionPlanner {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about subcompaction execution
#[derive(Debug, Clone, Copy)]
pub struct SubcompactionStats {
    /// Number of jobs split
    pub jobs_split: u64,

    /// Total subcompactions created
    pub subcompactions_created: u64,

    /// Average subcompactions per job
    pub avg_subcompactions: f64,
}

impl std::fmt::Display for SubcompactionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Subcompaction: {} jobs split, {} subjobs created (avg {:.1}/job)",
            self.jobs_split, self.subcompactions_created, self.avg_subcompactions
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_job(id: u64, size: u64, start: Vec<u8>, end: Vec<u8>) -> CompactionJob {
        let input = CompactionInput {
            level: 1,
            segments: vec![],
            key_range: KeyRange::new(start, end, id),
            total_size: size,
        };

        let output = CompactionOutput::new(2, 1024 * 1024);

        CompactionJob {
            id,
            job_type: CompactionJobType::LevelCompaction,
            input,
            next_level_input: None,
            output,
            score: 1.0,
            can_parallelize: true,
            allocated_segment_ids: vec![id],
        }
    }

    #[test]
    fn test_planner_creation() {
        let planner = SubcompactionPlanner::new();
        assert_eq!(planner.config.target_subcompactions, 4);
    }

    #[test]
    fn test_should_not_split_small_job() {
        let planner = SubcompactionPlanner::new();
        let job = create_test_job(1, 10 * 1024 * 1024, b"a".to_vec(), b"z".to_vec()); // 10MB

        assert!(!planner.should_split(&job));
    }

    #[test]
    fn test_should_split_large_job() {
        let planner = SubcompactionPlanner::new();
        let job = create_test_job(1, 100 * 1024 * 1024, b"a".to_vec(), b"z".to_vec()); // 100MB

        assert!(planner.should_split(&job));
    }

    #[test]
    fn test_split_large_job() {
        let planner = SubcompactionPlanner::new();
        let job = create_test_job(1, 100 * 1024 * 1024, b"a".to_vec(), b"z".to_vec());

        let subjobs = planner.split(&job);
        assert!(subjobs.is_some());

        let subjobs = subjobs.unwrap();
        assert!(subjobs.len() > 1);
        assert!(subjobs.len() <= planner.config.max_subcompactions);

        // Verify all subjobs have the same parent ID
        for subjob in &subjobs {
            assert_eq!(subjob.parent_id, job.id);
        }

        // Verify ranges are contiguous
        for i in 1..subjobs.len() {
            assert_eq!(subjobs[i - 1].key_range.end, subjobs[i].key_range.start);
        }

        // Verify first and last boundaries match
        assert_eq!(subjobs[0].key_range.start, job.input.key_range.start);
        assert_eq!(
            subjobs[subjobs.len() - 1].key_range.end,
            job.input.key_range.end
        );
    }

    #[test]
    fn test_calculate_num_subcompactions() {
        let planner = SubcompactionPlanner::new();

        // Small size
        let num = planner.calculate_num_subcompactions(20 * 1024 * 1024);
        assert!(num >= 1);

        // Large size
        let num = planner.calculate_num_subcompactions(200 * 1024 * 1024);
        assert!(num > 1);
        assert!(num <= planner.config.max_subcompactions);
    }

    #[test]
    fn test_split_key_range() {
        let planner = SubcompactionPlanner::new();
        let range = KeyRange::new(b"a".to_vec(), b"z".to_vec(), 1);

        let ranges = planner.split_key_range(&range, 4);

        assert_eq!(ranges.len(), 4);
        assert_eq!(ranges[0].start, b"a".to_vec());
        assert_eq!(ranges[3].end, b"z".to_vec());

        // Verify ranges are ordered
        for i in 1..ranges.len() {
            assert!(ranges[i - 1].end <= ranges[i].start);
        }
    }

    #[test]
    fn test_split_key_range_single() {
        let planner = SubcompactionPlanner::new();
        let range = KeyRange::new(b"a".to_vec(), b"z".to_vec(), 1);

        let ranges = planner.split_key_range(&range, 1);

        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], range);
    }

    #[test]
    fn test_interpolate_key() {
        let planner = SubcompactionPlanner::new();

        let start = b"a";
        let end = b"z";

        // Midpoint
        let mid = planner.interpolate_key(start, end, 0.5);
        assert!(mid.as_slice() >= start.as_slice());
        assert!(mid.as_slice() <= end.as_slice());

        // Quarter point
        let quarter = planner.interpolate_key(start, end, 0.25);
        assert!(quarter.as_slice() >= start.as_slice());
        assert!(quarter < mid);

        // Three quarter point
        let three_quarter = planner.interpolate_key(start, end, 0.75);
        assert!(three_quarter > mid);
        assert!(three_quarter.as_slice() <= end.as_slice());
    }

    #[test]
    fn test_flush_not_split() {
        let planner = SubcompactionPlanner::new();

        let mut job = create_test_job(1, 100 * 1024 * 1024, b"a".to_vec(), b"z".to_vec());
        job.job_type = CompactionJobType::Flush;

        assert!(!planner.should_split(&job));
    }

    #[test]
    fn test_trivial_move_not_split() {
        let planner = SubcompactionPlanner::new();

        let mut job = create_test_job(1, 100 * 1024 * 1024, b"a".to_vec(), b"z".to_vec());
        job.job_type = CompactionJobType::TrivialMove;

        assert!(!planner.should_split(&job));
    }

    #[test]
    fn test_custom_config() {
        let config = SubcompactionConfig {
            min_size_for_split: 32 * 1024 * 1024,
            target_subcompactions: 8,
            min_subcompaction_size: 8 * 1024 * 1024,
            max_subcompactions: 16,
        };

        let planner = SubcompactionPlanner::with_config(config);
        assert_eq!(planner.config.target_subcompactions, 8);
        assert_eq!(planner.config.max_subcompactions, 16);
    }
}
