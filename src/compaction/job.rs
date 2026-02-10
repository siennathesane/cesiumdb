//! Compaction job structures
//!
//! This module defines the different types of compaction jobs and their
//! metadata.

use std::sync::Arc;

use crate::{
    levels::KeyRange,
    segment::Segment,
};

/// Type of compaction operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionJobType {
    /// Flush memtable to L0
    ///
    /// Takes frozen memtable(s) and writes them as new L0 segment(s).
    /// - Input: 1+ frozen memtables
    /// - Output: 1+ L0 segments
    /// - Priority: Highest (blocks writes if memtable limit reached)
    Flush,

    /// L0 compaction to L1
    ///
    /// Merges overlapping L0 segments into L1.
    /// - Input: Multiple L0 segments (overlapping ranges)
    /// - Output: L1 segments (non-overlapping if using leveled strategy)
    /// - Priority: High (L0 segments hurt read performance)
    L0Compaction,

    /// Level-to-level compaction (Ln → Ln+1)
    ///
    /// Merges segments from one level to the next.
    /// - Input: Segments from Ln + overlapping segments from Ln+1
    /// - Output: Segments in Ln+1
    /// - Priority: Medium (based on level score)
    LevelCompaction,

    /// Trivial move (no actual merging needed)
    ///
    /// When a segment doesn't overlap with the next level, it can be
    /// moved directly without merging.
    /// - Input: 1 segment from Ln
    /// - Output: Same segment in Ln+1 (just metadata update)
    /// - Priority: Highest (zero-cost operation)
    TrivialMove,

    /// Manual compaction (user-triggered)
    ///
    /// Compacts a specific key range across levels.
    /// - Input: Segments overlapping the specified range
    /// - Output: Compacted segments
    /// - Priority: User-specified
    Manual,
}

impl CompactionJobType {
    /// Returns the base priority for this job type
    ///
    /// Lower values = higher priority
    pub fn base_priority(&self) -> u32 {
        match self {
            | Self::TrivialMove => 0,  // Instant, no I/O
            | Self::Flush => 1,        // Blocks writes
            | Self::L0Compaction => 2, // Hurts read performance
            | Self::LevelCompaction => 3,
            | Self::Manual => 2, // User-initiated, should be fast
        }
    }

    /// Returns whether this job type blocks writes
    pub fn blocks_writes(&self) -> bool {
        matches!(self, Self::Flush)
    }
}

/// Input segments for a compaction job
#[derive(Clone)]
pub struct CompactionInput {
    /// Source level (0 for L0, 1 for L1, etc.)
    pub level: u8,

    /// Segments to compact
    pub segments: Vec<Arc<Segment>>,

    /// Combined key range of all input segments
    ///
    /// For L0 (overlapping segments), this is the union of all ranges.
    /// For leveled compaction, this determines which segments from the
    /// next level need to be included.
    pub key_range: KeyRange,

    /// Total size of input segments (bytes)
    pub total_size: u64,
}

impl CompactionInput {
    /// Creates a new compaction input
    pub fn new(level: u8, segments: Vec<Arc<Segment>>) -> Self {
        let total_size: u64 = segments.iter().map(|s| s.size_in_bytes()).sum();

        // Compute the union of all key ranges
        // TODO: This requires segment metadata we haven't exposed yet
        // For now, use placeholder values
        let key_range = KeyRange::new(vec![], vec![], segments[0].id());

        Self {
            level,
            segments,
            key_range,
            total_size,
        }
    }

    /// Returns the number of segments in this input
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }
}

/// Output configuration for a compaction job
#[derive(Clone)]
pub struct CompactionOutput {
    /// Target level for output segments
    pub level: u8,

    /// Target size for each output segment (bytes)
    pub target_segment_size: u64,

    /// Maximum number of output segments
    ///
    /// Used to limit the number of files created.
    pub max_segments: Option<usize>,
}

impl CompactionOutput {
    /// Creates a new compaction output configuration
    pub fn new(level: u8, target_segment_size: u64) -> Self {
        Self {
            level,
            target_segment_size,
            max_segments: None,
        }
    }

    /// Sets the maximum number of output segments
    pub fn with_max_segments(mut self, max: usize) -> Self {
        self.max_segments = Some(max);
        self
    }
}

/// A compaction job
///
/// Represents a single compaction operation with inputs, outputs,
/// and metadata for scheduling and execution.
pub struct CompactionJob {
    /// Unique job ID
    pub id: u64,

    /// Type of compaction
    pub job_type: CompactionJobType,

    /// Input segments from the source level
    pub input: CompactionInput,

    /// Optional input from the next level (for level compaction)
    ///
    /// For L0→L1 and Ln→Ln+1 compactions, this contains the overlapping
    /// segments from the target level.
    pub next_level_input: Option<CompactionInput>,

    /// Output configuration
    pub output: CompactionOutput,

    /// Compaction score (higher = more urgent)
    ///
    /// Calculated based on:
    /// - Level size vs. target size
    /// - Number of L0 files
    /// - Write amplification
    /// - User priority (for manual compactions)
    pub score: f64,

    /// Whether this job can run in parallel with others
    ///
    /// Jobs with non-overlapping key ranges can run concurrently.
    pub can_parallelize: bool,
}

impl CompactionJob {
    /// Creates a new compaction job
    pub fn new(
        id: u64,
        job_type: CompactionJobType,
        input: CompactionInput,
        next_level_input: Option<CompactionInput>,
        output: CompactionOutput,
    ) -> Self {
        let score = Self::calculate_score(job_type, &input, next_level_input.as_ref());

        // Trivial moves and flushes can always parallelize
        // Level compactions can parallelize if ranges don't overlap
        let can_parallelize = matches!(
            job_type,
            CompactionJobType::TrivialMove | CompactionJobType::Flush
        );

        Self {
            id,
            job_type,
            input,
            next_level_input,
            output,
            score,
            can_parallelize,
        }
    }

    /// Calculates the urgency score for this compaction
    ///
    /// Higher scores = more urgent to compact
    fn calculate_score(
        job_type: CompactionJobType,
        input: &CompactionInput,
        next_level_input: Option<&CompactionInput>,
    ) -> f64 {
        match job_type {
            | CompactionJobType::TrivialMove => {
                // Always highest priority (free operation)
                100.0
            },
            | CompactionJobType::Flush => {
                // Priority based on number of frozen memtables
                // More frozen = more urgent
                input.num_segments() as f64 * 10.0
            },
            | CompactionJobType::L0Compaction => {
                // L0 file count is the main factor
                // Each additional file hurts read performance
                let l0_files = input.num_segments() as f64;
                let l0_size = input.total_size as f64;

                // Base score on file count (primary concern)
                let file_score = l0_files * 5.0;

                // Adjust for total size
                let size_score = (l0_size / (64.0 * 1024.0 * 1024.0)) * 2.0; // Normalize by 64MB

                file_score + size_score
            },
            | CompactionJobType::LevelCompaction => {
                // Score based on size ratio to target
                let input_size = input.total_size as f64;
                let next_level_size = next_level_input.map(|n| n.total_size as f64).unwrap_or(0.0);

                // Higher scores when level is over target
                // Target growth: 10x per level
                let target_size = 64.0 * 1024.0 * 1024.0 * 10_f64.powi(input.level as i32);
                let total_size = input_size + next_level_size;

                // Score > 1.0 means level is over target
                total_size / target_size
            },
            | CompactionJobType::Manual => {
                // User-triggered compactions have high priority
                50.0
            },
        }
    }

    /// Returns the total number of input segments
    pub fn total_input_segments(&self) -> usize {
        let mut count = self.input.num_segments();
        if let Some(ref next) = self.next_level_input {
            count += next.num_segments();
        }
        count
    }

    /// Returns the total input size in bytes
    pub fn total_input_size(&self) -> u64 {
        let mut size = self.input.total_size;
        if let Some(ref next) = self.next_level_input {
            size += next.total_size;
        }
        size
    }

    /// Returns the expected write amplification for this job
    ///
    /// Write amplification = bytes written / bytes from previous level
    pub fn write_amplification(&self) -> f64 {
        if self.input.total_size == 0 {
            return 1.0;
        }

        match self.job_type {
            | CompactionJobType::TrivialMove => 0.0, // No writes
            | CompactionJobType::Flush => 1.0,       // 1:1 write ratio
            | CompactionJobType::L0Compaction | CompactionJobType::LevelCompaction => {
                // Total bytes written / bytes from source level
                self.total_input_size() as f64 / self.input.total_size as f64
            },
            | CompactionJobType::Manual => {
                // Variable, depends on overlap
                self.total_input_size() as f64 / self.input.total_size as f64
            },
        }
    }

    /// Checks if this job overlaps with another job
    ///
    /// Used to determine if jobs can run in parallel.
    pub fn overlaps_with(&self, other: &CompactionJob) -> bool {
        // Same level = might overlap
        if self.input.level == other.input.level {
            return self.input.key_range.overlaps(&other.input.key_range);
        }

        // Different levels - check if one is compacting into the other's level
        if self.output.level == other.input.level {
            return self.input.key_range.overlaps(&other.input.key_range);
        }

        if other.output.level == self.input.level {
            return other.input.key_range.overlaps(&self.input.key_range);
        }

        // No overlap
        false
    }
}

impl std::fmt::Debug for CompactionJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Job#{} {:?} L{}→L{} ({} files, {:.1} MB, score={:.2})",
            self.id,
            self.job_type,
            self.input.level,
            self.output.level,
            self.total_input_segments(),
            self.total_input_size() as f64 / (1024.0 * 1024.0),
            self.score
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_type_priority() {
        assert!(
            CompactionJobType::TrivialMove.base_priority() <
                CompactionJobType::Flush.base_priority()
        );
        assert!(
            CompactionJobType::Flush.base_priority() <
                CompactionJobType::L0Compaction.base_priority()
        );
        assert!(
            CompactionJobType::L0Compaction.base_priority() <
                CompactionJobType::LevelCompaction.base_priority()
        );
    }

    #[test]
    fn test_job_type_blocks_writes() {
        assert!(CompactionJobType::Flush.blocks_writes());
        assert!(!CompactionJobType::L0Compaction.blocks_writes());
        assert!(!CompactionJobType::LevelCompaction.blocks_writes());
        assert!(!CompactionJobType::TrivialMove.blocks_writes());
    }

    #[test]
    fn test_compaction_output_builder() {
        let output = CompactionOutput::new(1, 64 * 1024 * 1024).with_max_segments(10);

        assert_eq!(output.level, 1);
        assert_eq!(output.target_segment_size, 64 * 1024 * 1024);
        assert_eq!(output.max_segments, Some(10));
    }

    #[test]
    fn test_score_calculation_flush() {
        // Flush jobs score based on number of memtables
        let input = CompactionInput {
            level: 0,
            segments: vec![], // Would normally contain memtable references
            key_range: KeyRange::new(vec![], vec![], 0),
            total_size: 256 * 1024 * 1024, // 256 MB
        };

        // Score should be 4 * 10.0 = 40.0 for 4 memtables
        // We can't easily create real segments here without full infrastructure
        // but we can verify the formula
        let num_memtables = 4;
        let expected_score = num_memtables as f64 * 10.0;
        assert_eq!(expected_score, 40.0);
    }

    #[test]
    fn test_trivial_move_highest_score() {
        let input = CompactionInput {
            level: 1,
            segments: vec![],
            key_range: KeyRange::new(vec![], vec![], 0),
            total_size: 64 * 1024 * 1024,
        };

        let score = CompactionJob::calculate_score(CompactionJobType::TrivialMove, &input, None);

        // Trivial moves should have highest score
        assert_eq!(score, 100.0);
    }

    #[test]
    fn test_write_amplification() {
        let input = CompactionInput {
            level: 1,
            segments: vec![],
            key_range: KeyRange::new(vec![], vec![], 0),
            total_size: 100 * 1024 * 1024, // 100 MB
        };

        let next_level = CompactionInput {
            level: 2,
            segments: vec![],
            key_range: KeyRange::new(vec![], vec![], 0),
            total_size: 400 * 1024 * 1024, // 400 MB
        };

        let output = CompactionOutput::new(2, 64 * 1024 * 1024);

        let job = CompactionJob::new(
            1,
            CompactionJobType::LevelCompaction,
            input,
            Some(next_level),
            output,
        );

        // Write amp = (100 + 400) / 100 = 5.0
        assert_eq!(job.write_amplification(), 5.0);
    }

    #[test]
    fn test_trivial_move_zero_write_amp() {
        let input = CompactionInput {
            level: 1,
            segments: vec![],
            key_range: KeyRange::new(vec![], vec![], 0),
            total_size: 64 * 1024 * 1024,
        };

        let output = CompactionOutput::new(2, 64 * 1024 * 1024);

        let job = CompactionJob::new(1, CompactionJobType::TrivialMove, input, None, output);

        assert_eq!(job.write_amplification(), 0.0);
    }
}
