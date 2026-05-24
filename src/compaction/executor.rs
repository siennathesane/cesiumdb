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
    sync::{
        Arc,
        atomic::{
            AtomicU64,
            Ordering,
        },
    },
};

use parking_lot::Mutex;
use thiserror::Error;

use crate::{
    compact::compact_raw,
    compaction::{
        SegmentRegistry,
        SubcompactionJob,
        SubcompactionPlanner,
        job::{
            CompactionJob,
            CompactionJobType,
        },
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

/// Result of a single subcompaction partition
struct SubcompactionResult {
    segment: Arc<Segment>,
    key_range: KeyRange,
    entry_count: u64,
    bytes_read: u64,
    bytes_written: u64,
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

    /// Segment registry for reference tracking and file deletion
    registry: Arc<SegmentRegistry>,

    /// Subcompaction planner for splitting large compactions
    subcompaction_planner: SubcompactionPlanner,

    /// Cumulative bytes read by compaction jobs
    bytes_read: AtomicU64,

    /// Cumulative bytes written by compaction jobs
    bytes_written: AtomicU64,
}

impl CompactionExecutor {
    /// Creates a new compaction executor
    pub fn new(
        version_manager: Arc<VersionManager>,
        base_path: PathBuf,
        manifest: Option<Arc<Mutex<ManifestWriter>>>,
        registry: Arc<SegmentRegistry>,
    ) -> Self {
        Self::with_planner(
            version_manager,
            base_path,
            manifest,
            registry,
            SubcompactionPlanner::new(),
        )
    }

    pub fn with_planner(
        version_manager: Arc<VersionManager>,
        base_path: PathBuf,
        manifest: Option<Arc<Mutex<ManifestWriter>>>,
        registry: Arc<SegmentRegistry>,
        subcompaction_planner: SubcompactionPlanner,
    ) -> Self {
        Self {
            version_manager,
            manifest,
            base_path,
            registry,
            subcompaction_planner,
            bytes_read: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
        }
    }

    /// Returns cumulative compaction I/O stats: (bytes_read, bytes_written)
    pub fn compaction_io(&self) -> (u64, u64) {
        (
            self.bytes_read.load(Ordering::Relaxed),
            self.bytes_written.load(Ordering::Relaxed),
        )
    }

    /// Checks whether all input segments for a job still exist in the
    /// current version set.  Used after a (potentially long) merge to
    /// decide whether the result is still safe to install.
    fn inputs_still_valid(&self, job: &CompactionJob) -> bool {
        let version = self.version_manager.current();
        match job.job_type {
            | CompactionJobType::L0Compaction => {
                let l0_ids: std::collections::HashSet<u64> =
                    version.l0.iter().map(|s| s.id()).collect();
                job.input.segments.iter().all(|s| l0_ids.contains(&s.id()))
            },
            | CompactionJobType::LevelCompaction | CompactionJobType::TrivialMove => {
                let level_idx = job.input.level as usize;
                if level_idx == 0 {
                    let l0_ids: std::collections::HashSet<u64> =
                        version.l0.iter().map(|s| s.id()).collect();
                    let inputs_ok = job.input.segments.iter().all(|s| l0_ids.contains(&s.id()));
                    let next_ok = job.next_level_input.as_ref().map_or(true, |next| {
                        let next_idx = next.level as usize - 1;
                        if next_idx < version.levels.len() {
                            let next_ids: std::collections::HashSet<u64> = version.levels[next_idx]
                                .segments
                                .iter()
                                .map(|s| s.id())
                                .collect();
                            next.segments.iter().all(|s| next_ids.contains(&s.id()))
                        } else {
                            false
                        }
                    });
                    inputs_ok && next_ok
                } else {
                    let level_idx = level_idx - 1;
                    if level_idx >= version.levels.len() {
                        return false;
                    }
                    let level_ids: std::collections::HashSet<u64> = version.levels[level_idx]
                        .segments
                        .iter()
                        .map(|s| s.id())
                        .collect();
                    let inputs_ok = job
                        .input
                        .segments
                        .iter()
                        .all(|s| level_ids.contains(&s.id()));
                    let next_ok = job.next_level_input.as_ref().map_or(true, |next| {
                        let next_idx = next.level as usize - 1;
                        if next_idx < version.levels.len() {
                            let next_ids: std::collections::HashSet<u64> = version.levels[next_idx]
                                .segments
                                .iter()
                                .map(|s| s.id())
                                .collect();
                            next.segments.iter().all(|s| next_ids.contains(&s.id()))
                        } else {
                            false
                        }
                    });
                    inputs_ok && next_ok
                }
            },
            | _ => true,
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

        // Validate that the compaction result is still safe to install.
        // L0 compactions are serialized, so new flushes cannot invalidate
        // our inputs; we only need to validate level / manual jobs.
        match job.job_type {
            | CompactionJobType::L0Compaction => {
                // L0 jobs are serialized — inputs are guaranteed to still
                // be present because no other compaction can remove them.
            },
            | CompactionJobType::LevelCompaction |
            CompactionJobType::Manual |
            CompactionJobType::TrivialMove => {
                if !self.inputs_still_valid(job) {
                    return Err(ExecutorError::VersionChanged);
                }
            },
            | _ => {},
        }

        // Update version set
        if let Err(e) = self.install_compaction_result(job, &result) {
            return Err(e);
        }

        // Record compaction I/O for throughput metrics
        match job.job_type {
            | CompactionJobType::L0Compaction |
            CompactionJobType::LevelCompaction |
            CompactionJobType::Manual => {
                self.bytes_read
                    .fetch_add(result.bytes_read, Ordering::Relaxed);
                self.bytes_written
                    .fetch_add(result.bytes_written, Ordering::Relaxed);
            },
            | _ => {},
        }

        // Clean up obsolete segments
        let (deleted, bytes_freed) = self.registry.cleanup();
        if deleted > 0 {
            tracing::info!(
                job_id = job.id,
                segments_deleted = deleted,
                bytes_freed = bytes_freed,
                "Cleaned up obsolete segments after compaction"
            );
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

        // Check if this job should be split into parallel subcompactions
        if let Some(subjobs) = self.subcompaction_planner.split(job) {
            if subjobs.len() > 1 {
                return self.execute_subcompactions(job, subjobs);
            }
        }

        self.execute_single_merge(job)
    }

    /// Executes a single merge compaction without splitting.
    fn execute_single_merge(&self, job: &CompactionJob) -> Result<CompactionResult, ExecutorError> {
        // Collect all input segments
        let mut all_inputs = job.input.segments.clone();
        if let Some(ref next_level) = job.next_level_input {
            all_inputs.extend(next_level.segments.clone());
        }

        // Create readers and raw iterators for all input segments
        let readers: Vec<_> = match all_inputs
            .iter()
            .map(|seg| seg.reader())
            .collect::<Result<Vec<_>, _>>()
        {
            | Ok(v) => v,
            | Err(e) => return Err(ExecutorError::SegmentError(e)),
        };

        let iterators: Vec<_> = readers
            .into_iter()
            .map(|reader| reader.scan_raw(Bound::Unbounded, Bound::Unbounded))
            .collect();

        let segment_id = job.allocated_segment_ids[0];
        let output_dir = self
            .base_path
            .join(format!("L{}", job.output.level))
            .join("segments")
            .join(segment_id.to_string());

        let compact_output = match compact_raw(iterators, output_dir, segment_id) {
            | Ok(v) => v,
            | Err(e) => return Err(ExecutorError::SegmentError(e)),
        };

        let bytes_read = job.total_input_size();
        let bytes_written = compact_output.segment.size_in_bytes();

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

    /// Executes a set of subcompactions in parallel, then aggregates results.
    fn execute_subcompactions(
        &self,
        job: &CompactionJob,
        subjobs: Vec<SubcompactionJob>,
    ) -> Result<CompactionResult, ExecutorError> {
        let num_subs = subjobs.len();
        tracing::info!(
            job_id = job.id,
            num_subcompactions = num_subs,
            "Executing subcompactions"
        );

        // Allocate segment IDs for each subcompaction output.
        // Sub-0 uses the pre-allocated ID; additional IDs come from the version
        // manager.
        let mut segment_ids = vec![job.allocated_segment_ids[0]];
        for _ in 1..num_subs {
            segment_ids.push(self.version_manager.next_segment_id());
        }

        // Run subcompactions in parallel using scoped threads.
        let results: Vec<Result<SubcompactionResult, ExecutorError>> =
            std::thread::scope(|scope| {
                let handles: Vec<_> = subjobs
                    .into_iter()
                    .enumerate()
                    .map(|(idx, subjob)| {
                        let segment_id = segment_ids[idx];
                        scope.spawn(move || self.execute_single_subcompaction(&subjob, segment_id))
                    })
                    .collect();

                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });

        // Aggregate outputs
        let mut output_segments = Vec::with_capacity(num_subs);
        let mut output_ranges = Vec::with_capacity(num_subs);
        let mut total_entries = 0u64;
        let mut total_bytes_read = 0u64;
        let mut total_bytes_written = 0u64;

        for result in results {
            let sub = result?;
            output_segments.push(sub.segment);
            output_ranges.push(sub.key_range);
            total_entries += sub.entry_count;
            total_bytes_read += sub.bytes_read;
            total_bytes_written += sub.bytes_written;
        }

        // Inputs to delete are the same as the parent job
        let mut all_inputs = job.input.segments.clone();
        if let Some(ref next_level) = job.next_level_input {
            all_inputs.extend(next_level.segments.clone());
        }
        let inputs_to_delete: Vec<u64> = all_inputs.iter().map(|s| s.id()).collect();

        Ok(CompactionResult {
            output_segments,
            output_ranges,
            inputs_to_delete,
            entries_processed: total_entries,
            bytes_read: total_bytes_read,
            bytes_written: total_bytes_written,
        })
    }

    /// Executes a single subcompaction — one key-range partition of a larger
    /// job.
    fn execute_single_subcompaction(
        &self,
        subjob: &SubcompactionJob,
        segment_id: u64,
    ) -> Result<SubcompactionResult, ExecutorError> {
        // Collect all input segments for this subcompaction
        let mut all_inputs = subjob.input.segments.clone();
        if let Some(ref next_level) = subjob.next_level_input {
            all_inputs.extend(next_level.segments.clone());
        }

        // Create bounded readers — each subcompaction only reads keys in its range
        let readers: Vec<_> = match all_inputs
            .iter()
            .map(|seg| seg.reader())
            .collect::<Result<Vec<_>, _>>()
        {
            | Ok(v) => v,
            | Err(e) => return Err(ExecutorError::SegmentError(e)),
        };

        let start_bound = Bound::Included(subjob.key_range.start.as_slice());
        let end_bound = Bound::Excluded(subjob.key_range.end.as_slice());

        let iterators: Vec<_> = readers
            .into_iter()
            .map(|reader| reader.scan_raw(start_bound, end_bound))
            .collect();

        let output_dir = self
            .base_path
            .join(format!("L{}", subjob.output.level))
            .join("segments")
            .join(segment_id.to_string());

        let compact_output = match compact_raw(iterators, output_dir, segment_id) {
            | Ok(v) => v,
            | Err(e) => return Err(ExecutorError::SegmentError(e)),
        };

        let bytes_read = subjob.input.total_size +
            subjob
                .next_level_input
                .as_ref()
                .map(|i| i.total_size)
                .unwrap_or(0);
        let bytes_written = compact_output.segment.size_in_bytes();

        let key_range = KeyRange::new(compact_output.min_key, compact_output.max_key, segment_id);

        Ok(SubcompactionResult {
            segment: compact_output.segment,
            key_range,
            entry_count: compact_output.entry_count,
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
                    // Remove L0 inputs
                    for segment_id in job.input.segments.iter().map(|s| s.id()) {
                        let edit = VersionEdit::RemoveL0Segment { segment_id };
                        if let Err(e) = manifest_writer.lock().append_edit(&edit) {
                            tracing::error!(error = ?e, "Failed to log RemoveL0Segment to manifest");
                        }
                    }
                    // Remove L1 inputs (if any overlap)
                    if let Some(ref next) = job.next_level_input {
                        for segment_id in next.segments.iter().map(|s| s.id()) {
                            let edit = VersionEdit::RemoveSegment {
                                level: next.level,
                                segment_id,
                            };
                            if let Err(e) = manifest_writer.lock().append_edit(&edit) {
                                tracing::error!(error = ?e, "Failed to log RemoveSegment to manifest");
                            }
                        }
                    }
                },
                | CompactionJobType::LevelCompaction | CompactionJobType::TrivialMove => {
                    // Remove from source level
                    for segment_id in job.input.segments.iter().map(|s| s.id()) {
                        let edit = VersionEdit::RemoveSegment {
                            level: job.input.level,
                            segment_id,
                        };
                        if let Err(e) = manifest_writer.lock().append_edit(&edit) {
                            tracing::error!(error = ?e, "Failed to log RemoveSegment to manifest");
                        }
                    }

                    // Also remove from next level if present
                    if let Some(ref next_input) = job.next_level_input {
                        for segment_id in next_input.segments.iter().map(|s| s.id()) {
                            let edit = VersionEdit::RemoveSegment {
                                level: next_input.level,
                                segment_id,
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
                    // Remove L0 inputs
                    let l0_ids: std::collections::HashSet<u64> =
                        job.input.segments.iter().map(|s| s.id()).collect();
                    version.l0.retain(|s| !l0_ids.contains(&s.id()));

                    // Remove L1 inputs (if any overlap)
                    if let Some(ref next) = job.next_level_input {
                        let l1_idx = next.level as usize - 1;
                        if l1_idx < version.levels.len() {
                            for seg in &next.segments {
                                version.levels[l1_idx].remove_segment(seg.id());
                            }
                        }
                    }
                },
                | CompactionJobType::LevelCompaction | CompactionJobType::TrivialMove => {
                    // Remove from source level
                    let level_idx = job.input.level as usize - 1;
                    if level_idx < version.levels.len() {
                        for seg in &job.input.segments {
                            version.levels[level_idx].remove_segment(seg.id());
                        }
                    }

                    // Also remove from next level if present
                    if let Some(ref next_input) = job.next_level_input {
                        let next_level_idx = next_input.level as usize - 1;
                        if next_level_idx < version.levels.len() {
                            for seg in &next_input.segments {
                                version.levels[next_level_idx].remove_segment(seg.id());
                            }
                        }
                    }
                },
                | _ => {},
            }

            // Add output segments to target level
            if job.output.level == 0 {
                // Add to L0 with key ranges
                for (segment, range) in result
                    .output_segments
                    .iter()
                    .zip(result.output_ranges.iter())
                {
                    version.add_to_l0(segment.clone(), range.clone());
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

        // Mark input segments for deletion (merge compactions only)
        // Trivial moves reuse the same segments in a different level,
        // so they must NOT be marked for deletion.
        match job.job_type {
            | CompactionJobType::L0Compaction |
            CompactionJobType::LevelCompaction |
            CompactionJobType::Manual => {
                for id in &result.inputs_to_delete {
                    self.registry.mark_for_deletion(*id);
                }
            },
            | _ => {},
        }

        // Register new output segments with the registry (merge compactions only)
        // Trivial moves reuse existing segments that are already registered.
        match job.job_type {
            | CompactionJobType::L0Compaction |
            CompactionJobType::LevelCompaction |
            CompactionJobType::Manual => {
                for segment in &result.output_segments {
                    let path = if job.output.level == 0 {
                        self.base_path
                            .join("segments")
                            .join(segment.id().to_string())
                    } else {
                        self.base_path
                            .join(format!("L{}", job.output.level))
                            .join("segments")
                            .join(segment.id().to_string())
                    };
                    self.registry.register(segment.clone(), path);
                }
            },
            | _ => {},
        }

        Ok(())
    }

    /// Returns the base path for segment files
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use tempfile::TempDir;

    use super::*;
    use crate::{
        compact::flush_memtable,
        compaction::job::{
            CompactionInput,
            CompactionOutput,
        },
        hlc::{
            HLC,
            HybridLogicalClock,
        },
        keypair::{
            DEFAULT_NS,
            KeyBytes,
            ValueBytes,
        },
        levels::VersionSet,
        memtable::Memtable,
        version::VersionManager,
    };

    #[test]
    fn test_executor_creation() {
        let temp_dir = TempDir::new().unwrap();
        let vm = Arc::new(VersionManager::new(7));
        let registry = Arc::new(SegmentRegistry::new(temp_dir.path().to_path_buf()));
        let executor = CompactionExecutor::new(vm, temp_dir.path().to_path_buf(), None, registry);

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
        let registry = Arc::new(SegmentRegistry::new(temp_dir.path().to_path_buf()));
        let executor = CompactionExecutor::new(vm, temp_dir.path().to_path_buf(), None, registry);

        let input = CompactionInput {
            level: 1,
            segments: vec![], // Empty
            key_range: KeyRange::new(vec![], vec![], 0),
            total_size: 0,
        };

        let output = CompactionOutput::new(2, 64 * 1024 * 1024);

        let job = CompactionJob::new(
            1,
            CompactionJobType::TrivialMove,
            input,
            None,
            output,
            vec![],
        );

        let result = executor.execute_trivial_move(&job);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e, ExecutorError::NoInputSegments));
        }
    }

    #[test]
    fn test_merge_compaction_deletes_input_files() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path().to_path_buf();
        let clock = HybridLogicalClock::new();

        // Create two L0 segments by flushing memtables
        let l0_path = base_path.join("segments");

        // Segment 1: keys "a" through "j"
        let memtable1 = Arc::new(Memtable::new(1, 1024 * 1024));
        for i in 0..10 {
            let key = KeyBytes::new(
                DEFAULT_NS,
                Bytes::from(format!("key_{:02}", i)),
                clock.time(),
            );
            let val = ValueBytes::new(DEFAULT_NS, Bytes::from(format!("value_{:02}", i)));
            memtable1.put(key, val).unwrap();
        }
        memtable1.freeze();
        let seg1_path = l0_path.join("1");
        let (segment1, min_key1, max_key1) =
            flush_memtable(memtable1, seg1_path.clone(), 1).unwrap();

        // Segment 2: keys "k" through "t"
        let memtable2 = Arc::new(Memtable::new(2, 1024 * 1024));
        for i in 10..20 {
            let key = KeyBytes::new(
                DEFAULT_NS,
                Bytes::from(format!("key_{:02}", i)),
                clock.time(),
            );
            let val = ValueBytes::new(DEFAULT_NS, Bytes::from(format!("value_{:02}", i)));
            memtable2.put(key, val).unwrap();
        }
        memtable2.freeze();
        let seg2_path = l0_path.join("3");
        let (segment2, min_key2, max_key2) =
            flush_memtable(memtable2, seg2_path.clone(), 3).unwrap();

        // Set up version manager with the L0 segments
        let version_manager = Arc::new(VersionManager::new(7));
        let key_range1 = KeyRange::new(min_key1, max_key1, segment1.id());
        let key_range2 = KeyRange::new(min_key2, max_key2, segment2.id());
        version_manager.update(|version| {
            version.add_to_l0(segment1.clone(), key_range1.clone());
            version.add_to_l0(segment2.clone(), key_range2.clone());
        });

        // Create registry and register both segments
        let registry = Arc::new(SegmentRegistry::new(base_path.clone()));
        registry.register(segment1.clone(), seg1_path.clone());
        registry.register(segment2.clone(), seg2_path.clone());

        // Extract IDs and sizes before moving segments into job
        let seg1_id = segment1.id();
        let seg2_id = segment2.id();
        let total_size = segment1.size_in_bytes() + segment2.size_in_bytes();

        // Build compaction input (this moves the Arcs)
        let input = CompactionInput {
            level: 0,
            segments: vec![segment1, segment2],
            key_range: KeyRange::new(key_range1.start.clone(), key_range2.end.clone(), seg1_id),
            total_size,
        };

        let output = CompactionOutput::new(1, 64 * 1024 * 1024);
        let job = CompactionJob::new(
            1,
            CompactionJobType::L0Compaction,
            input,
            None,
            output,
            vec![5], // Pre-allocated output segment ID
        );

        // Create executor with a planner that won't split (high threshold)
        let executor = CompactionExecutor::with_planner(
            Arc::clone(&version_manager),
            base_path.clone(),
            None,
            Arc::clone(&registry),
            crate::compaction::SubcompactionPlanner::with_config(
                crate::compaction::SubcompactionConfig {
                    min_size_for_split: u64::MAX,
                    ..Default::default()
                },
            ),
        );

        let result = executor.execute(&job).unwrap();

        // Verify output segment was created
        assert_eq!(result.output_segments.len(), 1);
        let output_segment = &result.output_segments[0];
        assert_eq!(output_segment.id(), 5);

        // Verify output directory exists
        let output_path = base_path.join("L1").join("segments").join("5");
        assert!(
            output_path.exists(),
            "output segment directory should exist"
        );

        // Drop the job (which holds input segment Arcs) so registry can clean up
        drop(job);

        // Run cleanup again now that all external references are dropped
        let (deleted, _bytes_freed) = registry.cleanup();
        assert_eq!(deleted, 2, "both input segments should be deleted");

        // Verify input directories were deleted
        assert!(
            !seg1_path.exists(),
            "input segment 1 directory should be deleted"
        );
        assert!(
            !seg2_path.exists(),
            "input segment 2 directory should be deleted"
        );

        // Verify registry state
        assert_eq!(registry.live_count(), 1); // Only output segment
        assert_eq!(registry.pending_deletion_count(), 0);

        // Verify version set was updated
        let version = version_manager.current();
        assert_eq!(version.l0.len(), 0); // L0 should be empty
        assert!(version.levels[0].segments.iter().any(|s| s.id() == 5)); // L1 should have output
    }
}
