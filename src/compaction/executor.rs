//! Compaction executor
//!
//! Executes compaction jobs by:
//! - Merging input segments
//! - Writing output segments
//! - Updating the version set

use crate::compaction::job::{CompactionJob, CompactionJobType};
use crate::compact::compact;
use crate::errs::SegmentError;
use crate::levels::{KeyRange, VersionSet};
use crate::memtable::Memtable;
use crate::segment::Segment;
use crate::segment_reader::SegmentReader;
use crate::version::VersionManager;
use std::ops::Bound;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;

/// Compaction executor errors
#[derive(Error, Debug)]
pub enum ExecutorError {
    #[error("Segment error: {0}")]
    SegmentError(#[from] SegmentError),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Job aborted: version changed")]
    VersionChanged,

    #[error("No input segments")]
    NoInputSegments,

    #[error("Invalid job type: {0:?}")]
    InvalidJobType(CompactionJobType),
}

/// Result of a compaction job execution
pub struct CompactionResult {
    /// Output segments created
    pub output_segments: Vec<Arc<Segment>>,

    /// Key ranges of output segments
    pub output_ranges: Vec<KeyRange>,

    /// Segments that were compacted (to be deleted)
    pub inputs_to_delete: Vec<u64>,

    /// Number of entries processed
    pub entries_processed: u64,

    /// Bytes read
    pub bytes_read: u64,

    /// Bytes written
    pub bytes_written: u64,
}

/// Compaction executor
///
/// Executes compaction jobs and updates the version set.
pub struct CompactionExecutor {
    /// Version manager for atomic updates
    version_manager: Arc<VersionManager>,

    /// Base directory for segment files
    base_path: PathBuf,
}

impl CompactionExecutor {
    /// Creates a new compaction executor
    pub fn new(version_manager: Arc<VersionManager>, base_path: PathBuf) -> Self {
        Self {
            version_manager,
            base_path,
        }
    }

    /// Executes a compaction job
    ///
    /// This is the main entry point for running a compaction.
    /// It:
    /// 1. Validates the job is still valid
    /// 2. Executes the appropriate compaction type
    /// 3. Updates the version set atomically
    pub fn execute(&self, job: &CompactionJob) -> Result<CompactionResult, ExecutorError> {
        // Capture current version sequence for validation
        let initial_seq = self.version_manager.sequence();

        // Execute based on job type
        let result = match job.job_type {
            CompactionJobType::TrivialMove => self.execute_trivial_move(job)?,
            CompactionJobType::Flush => {
                // TODO: Flush requires memtable access
                return Err(ExecutorError::InvalidJobType(job.job_type));
            }
            CompactionJobType::L0Compaction | CompactionJobType::LevelCompaction => {
                self.execute_merge_compaction(job)?
            }
            CompactionJobType::Manual => self.execute_merge_compaction(job)?,
        };

        // Verify version hasn't changed
        if !self.version_manager.is_current(initial_seq) {
            return Err(ExecutorError::VersionChanged);
        }

        // Update version set
        self.install_compaction_result(job, &result)?;

        Ok(result)
    }

    /// Executes a trivial move (just metadata update, no I/O)
    fn execute_trivial_move(&self, job: &CompactionJob) -> Result<CompactionResult, ExecutorError> {
        if job.input.segments.is_empty() {
            return Err(ExecutorError::NoInputSegments);
        }

        // For trivial move, output segments are the same as input segments
        let output_segments = job.input.segments.clone();

        let output_ranges = vec![job.input.key_range.clone()];

        let inputs_to_delete = job.input.segments.iter().map(|s| s.id()).collect();

        Ok(CompactionResult {
            output_segments,
            output_ranges,
            inputs_to_delete,
            entries_processed: 0,
            bytes_read: 0,
            bytes_written: 0,
        })
    }

    /// Executes a merge compaction (L0→L1 or Ln→Ln+1)
    fn execute_merge_compaction(&self, job: &CompactionJob) -> Result<CompactionResult, ExecutorError> {
        if job.input.segments.is_empty() {
            return Err(ExecutorError::NoInputSegments);
        }

        // Collect all input segments
        let mut all_inputs = job.input.segments.clone();
        if let Some(ref next_level) = job.next_level_input {
            all_inputs.extend(next_level.segments.clone());
        }

        // Create readers and iterators for all input segments
        // We need to keep readers alive for the duration of iteration
        let readers: Vec<_> = all_inputs
            .iter()
            .map(|seg| seg.reader())
            .collect::<Result<Vec<_>, _>>()?;

        let iterators: Vec<_> = readers
            .iter()
            .map(|reader| {
                let iter = reader.scan(Bound::Unbounded, Bound::Unbounded);
                // Filter and unwrap Results - skip errors
                iter.filter_map(|r| r.ok())
            })
            .collect();

        // Calculate output path
        let output_dir = self.base_path.join(format!("L{}", job.output.level));
        let segment_id = job.id; // Use job ID as segment ID for now

        // Run the compaction
        let output_segment = compact(iterators, output_dir, segment_id)?;

        // TODO: Track statistics
        let bytes_read = job.total_input_size();
        let bytes_written = output_segment.size_in_bytes();

        // TODO: Compute key range from output segment
        // For now, use a placeholder
        let output_range = KeyRange::new(vec![], vec![], segment_id);

        let inputs_to_delete: Vec<u64> = all_inputs.iter().map(|s| s.id()).collect();

        Ok(CompactionResult {
            output_segments: vec![output_segment],
            output_ranges: vec![output_range],
            inputs_to_delete,
            entries_processed: 0, // TODO: Track this
            bytes_read,
            bytes_written,
        })
    }

    /// Installs compaction result into the version set
    fn install_compaction_result(
        &self,
        job: &CompactionJob,
        result: &CompactionResult,
    ) -> Result<(), ExecutorError> {
        self.version_manager.update(|version| {
            // Remove input segments
            match job.job_type {
                CompactionJobType::L0Compaction => {
                    // Remove from L0
                    version.l0.retain(|s| !result.inputs_to_delete.contains(&s.id()));
                }
                CompactionJobType::LevelCompaction | CompactionJobType::TrivialMove => {
                    // Remove from source level
                    let level_idx = job.input.level as usize - 1;
                    if level_idx < version.levels.len() {
                        for segment_id in &result.inputs_to_delete {
                            version.levels[level_idx].remove_segment(*segment_id);
                        }
                    }

                    // Also remove from next level if present
                    if let Some(ref next_input) = job.next_level_input {
                        let next_level_idx = next_input.level as usize - 1;
                        if next_level_idx < version.levels.len() {
                            for segment_id in &result.inputs_to_delete {
                                version.levels[next_level_idx].remove_segment(*segment_id);
                            }
                        }
                    }
                }
                _ => {}
            }

            // Add output segments to target level
            if job.output.level == 0 {
                // Add to L0
                for segment in &result.output_segments {
                    version.add_to_l0(segment.clone());
                }
            } else {
                // Add to Ln
                let output_level_idx = job.output.level as usize - 1;
                if output_level_idx < version.levels.len() {
                    for (segment, range) in result
                        .output_segments
                        .iter()
                        .zip(result.output_ranges.iter())
                    {
                        version.levels[output_level_idx].add_segment(segment.clone(), range.clone());
                    }
                }
            }
        });

        Ok(())
    }

    /// Returns the base path for segment files
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::levels::VersionSet;
    use crate::version::VersionManager;
    use tempfile::TempDir;

    #[test]
    fn test_executor_creation() {
        let temp_dir = TempDir::new().unwrap();
        let vm = Arc::new(VersionManager::new(7));
        let executor = CompactionExecutor::new(vm, temp_dir.path().to_path_buf());

        assert_eq!(executor.base_path(), temp_dir.path());
    }

    #[test]
    fn test_trivial_move_no_inputs() {
        use crate::compaction::job::{CompactionInput, CompactionOutput};

        let temp_dir = TempDir::new().unwrap();
        let vm = Arc::new(VersionManager::new(7));
        let executor = CompactionExecutor::new(vm, temp_dir.path().to_path_buf());

        let input = CompactionInput {
            level: 1,
            segments: vec![], // Empty
            key_range: KeyRange::new(vec![], vec![], 0),
            total_size: 0,
        };

        let output = CompactionOutput::new(2, 64 * 1024 * 1024);

        let job = CompactionJob::new(1, CompactionJobType::TrivialMove, input, None, output);

        let result = executor.execute_trivial_move(&job);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e, ExecutorError::NoInputSegments));
        }
    }
}
