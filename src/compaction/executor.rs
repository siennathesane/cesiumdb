//! Compaction executor
//!
//! Executes compaction jobs by:
//! - Merging input segments
//! - Writing output segments
//! - Updating the version set

use std::{
    ops::Bound,
    path::{
        Path,
        PathBuf,
    },
    sync::Arc,
};

use parking_lot::Mutex;
use thiserror::Error;

use crate::{
    compact::compact_raw,
    compaction::job::{
        CompactionJob,
        CompactionJobType,
    },
    errs::SegmentError,
    levels::{
        KeyRange,
        VersionSet,
    },
    manifest_writer::ManifestWriter,
    memtable::Memtable,
    segment::Segment,
    segment_reader::SegmentReader,
    utils::Serializer,
    version::{
        VersionEdit,
        VersionManager,
    },
};

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

    #[error("Flush jobs are handled by the background flusher, not the compaction executor")]
    FlushNotRouted,
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

    /// Manifest writer for crash recovery
    manifest: Option<Arc<Mutex<ManifestWriter>>>,

    /// Base directory for segment files
    base_path: PathBuf,
}

impl CompactionExecutor {
    /// Creates a new compaction executor
    pub fn new(
        version_manager: Arc<VersionManager>,
        base_path: PathBuf,
        manifest: Option<Arc<Mutex<ManifestWriter>>>,
    ) -> Self {
        Self {
            version_manager,
            manifest,
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
            | CompactionJobType::TrivialMove => match self.execute_trivial_move(job) {
                | Ok(v) => v,
                | Err(e) => return Err(e),
            },
            | CompactionJobType::Flush => {
                // Flushes go through the background flusher in state.rs, not the executor
                return Err(ExecutorError::FlushNotRouted);
            },
            | CompactionJobType::L0Compaction | CompactionJobType::LevelCompaction => {
                match self.execute_merge_compaction(job) {
                    | Ok(v) => v,
                    | Err(e) => return Err(e),
                }
            },
            | CompactionJobType::Manual => match self.execute_merge_compaction(job) {
                | Ok(v) => v,
                | Err(e) => return Err(e),
            },
        };

        // Verify version hasn't changed
        if !self.version_manager.is_current(initial_seq) {
            return Err(ExecutorError::VersionChanged);
        }

        // Update version set
        if let Err(e) = self.install_compaction_result(job, &result) {
            return Err(e);
        }

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
    fn execute_merge_compaction(
        &self,
        job: &CompactionJob,
    ) -> Result<CompactionResult, ExecutorError> {
        if job.input.segments.is_empty() {
            return Err(ExecutorError::NoInputSegments);
        }

        // Collect all input segments
        let mut all_inputs = job.input.segments.clone();
        if let Some(ref next_level) = job.next_level_input {
            all_inputs.extend(next_level.segments.clone());
        }

        // Create readers and raw iterators for all input segments
        // We need to keep readers alive for the duration of iteration
        let readers: Vec<_> = match all_inputs
            .iter()
            .map(|seg| seg.reader())
            .collect::<Result<Vec<_>, _>>()
        {
            | Ok(v) => v,
            | Err(e) => return Err(ExecutorError::SegmentError(e)),
        };

        let iterators: Vec<_> = readers
            .iter()
            .map(|reader| reader.scan_raw(Bound::Unbounded, Bound::Unbounded))
            .collect();

        // Calculate output path (must match recovery path format)
        let segment_id = job.id; // Use job ID as segment ID for now
        let output_dir = self
            .base_path
            .join(format!("L{}", job.output.level))
            .join("sstables")
            .join(segment_id.to_string());

        // Run the zero-copy raw compaction
        let compact_output = match compact_raw(iterators, output_dir, segment_id) {
            | Ok(v) => v,
            | Err(e) => return Err(ExecutorError::SegmentError(e)),
        };

        let bytes_read = job.total_input_size();
        let bytes_written = compact_output.segment.size_in_bytes();

        // Use key range from compaction output (NO re-scan needed!)
        let output_range =
            KeyRange::new(compact_output.min_key, compact_output.max_key, segment_id);

        let inputs_to_delete: Vec<u64> = all_inputs.iter().map(|s| s.id()).collect();

        Ok(CompactionResult {
            output_segments: vec![compact_output.segment],
            output_ranges: vec![output_range],
            inputs_to_delete,
            entries_processed: compact_output.entry_count,
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
        // Log to manifest BEFORE updating version (write-ahead)
        if let Some(ref manifest_writer) = self.manifest {
            // Log removals
            match job.job_type {
                | CompactionJobType::L0Compaction => {
                    // Remove from L0
                    for segment_id in &result.inputs_to_delete {
                        let edit = VersionEdit::RemoveL0Segment {
                            segment_id: *segment_id,
                        };
                        if let Err(e) = manifest_writer.lock().append_edit(&edit) {
                            tracing::error!(error = ?e, "Failed to log RemoveL0Segment to manifest");
                        }
                    }
                },
                | CompactionJobType::LevelCompaction | CompactionJobType::TrivialMove => {
                    // Remove from source level
                    for segment_id in &result.inputs_to_delete {
                        let edit = VersionEdit::RemoveSegment {
                            level: job.input.level,
                            segment_id: *segment_id,
                        };
                        if let Err(e) = manifest_writer.lock().append_edit(&edit) {
                            tracing::error!(error = ?e, "Failed to log RemoveSegment to manifest");
                        }
                    }

                    // Also remove from next level if present
                    if let Some(ref next_input) = job.next_level_input {
                        for segment_id in &result.inputs_to_delete {
                            let edit = VersionEdit::RemoveSegment {
                                level: next_input.level,
                                segment_id: *segment_id,
                            };
                            if let Err(e) = manifest_writer.lock().append_edit(&edit) {
                                tracing::error!(error = ?e, "Failed to log RemoveSegment to manifest");
                            }
                        }
                    }
                },
                | _ => {},
            }

            // Log additions
            if job.output.level == 0 {
                // Add to L0
                for (segment, range) in result
                    .output_segments
                    .iter()
                    .zip(result.output_ranges.iter())
                {
                    let edit = VersionEdit::AddL0Segment {
                        segment_id: segment.id(),
                        key_range: (range.start.to_vec(), range.end.to_vec()),
                        size: segment.size_in_bytes(),
                    };
                    if let Err(e) = manifest_writer.lock().append_edit(&edit) {
                        tracing::error!(error = ?e, "Failed to log AddL0Segment to manifest");
                    }
                }
            } else {
                // Add to Ln
                for (segment, range) in result
                    .output_segments
                    .iter()
                    .zip(result.output_ranges.iter())
                {
                    let edit = VersionEdit::AddSegment {
                        level: job.output.level,
                        segment_id: segment.id(),
                        key_range: (range.start.to_vec(), range.end.to_vec()),
                        size: segment.size_in_bytes(),
                    };
                    if let Err(e) = manifest_writer.lock().append_edit(&edit) {
                        tracing::error!(error = ?e, "Failed to log AddSegment to manifest");
                    }
                }
            }

            // Sync manifest periodically
            if manifest_writer.lock().entry_count() % 10 == 0 {
                let _ = manifest_writer.lock().sync();
            }
        }

        self.version_manager.update(|version| {
            // Remove input segments
            match job.job_type {
                | CompactionJobType::L0Compaction => {
                    // Remove from L0
                    version
                        .l0
                        .retain(|s| !result.inputs_to_delete.contains(&s.id()));
                },
                | CompactionJobType::LevelCompaction | CompactionJobType::TrivialMove => {
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
                },
                | _ => {},
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
                        version.levels[output_level_idx]
                            .add_segment(segment.clone(), range.clone());
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
    use tempfile::TempDir;

    use super::*;
    use crate::{
        levels::VersionSet,
        version::VersionManager,
    };

    #[test]
    fn test_executor_creation() {
        let temp_dir = TempDir::new().unwrap();
        let vm = Arc::new(VersionManager::new(7));
        let executor = CompactionExecutor::new(vm, temp_dir.path().to_path_buf(), None);

        assert_eq!(executor.base_path(), temp_dir.path());
    }

    #[test]
    fn test_trivial_move_no_inputs() {
        use crate::compaction::job::{
            CompactionInput,
            CompactionOutput,
        };

        let temp_dir = TempDir::new().unwrap();
        let vm = Arc::new(VersionManager::new(7));
        let executor = CompactionExecutor::new(vm, temp_dir.path().to_path_buf(), None);

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
