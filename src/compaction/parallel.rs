//! Parallel compaction coordinator
//!
//! Enables multiple compaction jobs to run concurrently when they have
//! non-overlapping key ranges, maximizing throughput.

use std::{
    collections::HashSet,
    sync::Arc,
};

use parking_lot::RwLock;

use crate::{
    compaction::job::CompactionJob,
    levels::KeyRange,
};

/// Tracks active compaction jobs to prevent conflicts
pub struct ParallelCompactionCoordinator {
    /// Currently active key ranges being compacted
    active_ranges: RwLock<Vec<KeyRange>>,

    /// Active job IDs
    active_jobs: RwLock<HashSet<u64>>,

    /// Maximum number of parallel compaction jobs
    max_parallel: usize,
}

impl ParallelCompactionCoordinator {
    /// Creates a new coordinator
    pub fn new(max_parallel: usize) -> Self {
        Self {
            active_ranges: RwLock::new(Vec::new()),
            active_jobs: RwLock::new(HashSet::new()),
            max_parallel,
        }
    }

    /// Attempts to acquire permission to run a compaction job
    ///
    /// Returns true if the job can proceed (no overlap with active ranges),
    /// false if it should wait.
    pub fn try_acquire(&self, job: &CompactionJob) -> bool {
        let active_ranges = self.active_ranges.read();
        let active_jobs = self.active_jobs.read();

        // Check if we've hit the parallel limit
        if active_jobs.len() >= self.max_parallel {
            return false;
        }

        // Check if job is already running
        if active_jobs.contains(&job.id) {
            return false;
        }

        // Check for key range overlap with active jobs
        let job_range = &job.input.key_range;

        for active_range in active_ranges.iter() {
            if self.ranges_overlap(job_range, active_range) {
                return false;
            }
        }

        // No conflicts, can proceed
        true
    }

    /// Marks a job as active
    ///
    /// Should be called after try_acquire returns true, before starting work.
    pub fn mark_active(&self, job: &CompactionJob) {
        let mut active_ranges = self.active_ranges.write();
        let mut active_jobs = self.active_jobs.write();

        active_jobs.insert(job.id);
        active_ranges.push(job.input.key_range.clone());

        // Also add next level input range if present
        if let Some(ref next_level) = job.next_level_input {
            active_ranges.push(next_level.key_range.clone());
        }
    }

    /// Marks a job as complete, releasing its ranges
    pub fn mark_complete(&self, job_id: u64) {
        let mut active_jobs = self.active_jobs.write();
        active_jobs.remove(&job_id);

        // Note: We keep ranges in active_ranges for simplicity
        // In production, you'd want to remove only the ranges for this job
        // For now, we'll clear all ranges when no jobs are active
        if active_jobs.is_empty() {
            let mut active_ranges = self.active_ranges.write();
            active_ranges.clear();
        }
    }

    /// Checks if two key ranges overlap
    fn ranges_overlap(&self, a: &KeyRange, b: &KeyRange) -> bool {
        // If either range is empty, no overlap
        if a.start.is_empty() || a.end.is_empty() || b.start.is_empty() || b.end.is_empty() {
            return false;
        }

        // Use the built-in overlaps method
        a.overlaps(b)
    }

    /// Returns the number of currently active jobs
    pub fn active_count(&self) -> usize {
        self.active_jobs.read().len()
    }

    /// Returns statistics about parallel execution
    pub fn stats(&self) -> ParallelStats {
        let active_jobs = self.active_jobs.read();
        let active_ranges = self.active_ranges.read();

        ParallelStats {
            active_jobs: active_jobs.len(),
            active_ranges: active_ranges.len(),
            max_parallel: self.max_parallel,
            utilization: active_jobs.len() as f64 / self.max_parallel as f64,
        }
    }
}

/// Statistics about parallel compaction execution
#[derive(Debug, Clone, Copy)]
pub struct ParallelStats {
    /// Number of jobs currently running
    pub active_jobs: usize,

    /// Number of active key ranges
    pub active_ranges: usize,

    /// Maximum parallel jobs allowed
    pub max_parallel: usize,

    /// Utilization (0.0-1.0)
    pub utilization: f64,
}

impl std::fmt::Display for ParallelStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Parallel: {}/{} jobs ({:.0}% util), {} ranges",
            self.active_jobs,
            self.max_parallel,
            self.utilization * 100.0,
            self.active_ranges
        )
    }
}

/// Parallel compaction manager
///
/// Wraps the coordinator with convenient APIs for managing parallel execution.
pub struct ParallelCompactionManager {
    coordinator: Arc<ParallelCompactionCoordinator>,
}

impl ParallelCompactionManager {
    /// Creates a new manager
    pub fn new(max_parallel: usize) -> Self {
        Self {
            coordinator: Arc::new(ParallelCompactionCoordinator::new(max_parallel)),
        }
    }

    /// Returns the coordinator
    pub fn coordinator(&self) -> Arc<ParallelCompactionCoordinator> {
        Arc::clone(&self.coordinator)
    }

    /// Checks if a job can run in parallel
    pub fn can_run_parallel(&self, job: &CompactionJob) -> bool {
        // Check the job's can_parallelize flag
        if !job.can_parallelize {
            return false;
        }

        // Check with coordinator
        self.coordinator.try_acquire(job)
    }

    /// Executes a job with parallel coordination
    pub fn execute_parallel<F, R>(&self, job: &CompactionJob, f: F) -> R
    where
        F: FnOnce() -> R, {
        // Mark as active
        self.coordinator.mark_active(job);

        // Execute the job
        let result = f();

        // Mark as complete
        self.coordinator.mark_complete(job.id);

        result
    }

    /// Returns statistics
    pub fn stats(&self) -> ParallelStats {
        self.coordinator.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::job::{
        CompactionInput,
        CompactionJobType,
        CompactionOutput,
    };

    fn create_test_job(
        id: u64,
        start: Vec<u8>,
        end: Vec<u8>,
        can_parallelize: bool,
    ) -> CompactionJob {
        let input = CompactionInput {
            level: 0,
            segments: vec![],
            key_range: KeyRange::new(start, end, id),
            total_size: 0,
        };

        let output = CompactionOutput::new(1, 1024 * 1024);

        CompactionJob {
            id,
            job_type: CompactionJobType::LevelCompaction,
            input,
            next_level_input: None,
            output,
            score: 1.0,
            can_parallelize,
        }
    }

    #[test]
    fn test_coordinator_creation() {
        let coordinator = ParallelCompactionCoordinator::new(4);
        assert_eq!(coordinator.active_count(), 0);
    }

    #[test]
    fn test_non_overlapping_jobs() {
        let coordinator = ParallelCompactionCoordinator::new(4);

        let job1 = create_test_job(1, b"a".to_vec(), b"m".to_vec(), true);
        let job2 = create_test_job(2, b"n".to_vec(), b"z".to_vec(), true);

        // Both should be acquirable
        assert!(coordinator.try_acquire(&job1));
        assert!(coordinator.try_acquire(&job2));

        coordinator.mark_active(&job1);
        coordinator.mark_active(&job2);

        assert_eq!(coordinator.active_count(), 2);
    }

    #[test]
    fn test_overlapping_jobs() {
        let coordinator = ParallelCompactionCoordinator::new(4);

        let job1 = create_test_job(1, b"a".to_vec(), b"m".to_vec(), true);
        let job2 = create_test_job(2, b"k".to_vec(), b"z".to_vec(), true);

        // First job can be acquired
        assert!(coordinator.try_acquire(&job1));
        coordinator.mark_active(&job1);

        // Second job overlaps, should be blocked
        assert!(!coordinator.try_acquire(&job2));
    }

    #[test]
    fn test_max_parallel_limit() {
        let coordinator = ParallelCompactionCoordinator::new(2);

        let job1 = create_test_job(1, b"a".to_vec(), b"d".to_vec(), true);
        let job2 = create_test_job(2, b"e".to_vec(), b"h".to_vec(), true);
        let job3 = create_test_job(3, b"i".to_vec(), b"z".to_vec(), true);

        assert!(coordinator.try_acquire(&job1));
        coordinator.mark_active(&job1);

        assert!(coordinator.try_acquire(&job2));
        coordinator.mark_active(&job2);

        // Third job blocked by max parallel limit
        assert!(!coordinator.try_acquire(&job3));
    }

    #[test]
    fn test_mark_complete() {
        let coordinator = ParallelCompactionCoordinator::new(4);

        let job1 = create_test_job(1, b"a".to_vec(), b"m".to_vec(), true);

        assert!(coordinator.try_acquire(&job1));
        coordinator.mark_active(&job1);
        assert_eq!(coordinator.active_count(), 1);

        coordinator.mark_complete(1);
        assert_eq!(coordinator.active_count(), 0);
    }

    #[test]
    fn test_stats() {
        let coordinator = ParallelCompactionCoordinator::new(4);

        let stats = coordinator.stats();
        assert_eq!(stats.active_jobs, 0);
        assert_eq!(stats.max_parallel, 4);
        assert_eq!(stats.utilization, 0.0);

        let job1 = create_test_job(1, b"a".to_vec(), b"m".to_vec(), true);
        coordinator.mark_active(&job1);

        let stats = coordinator.stats();
        assert_eq!(stats.active_jobs, 1);
        assert_eq!(stats.utilization, 0.25);
    }

    #[test]
    fn test_manager_creation() {
        let manager = ParallelCompactionManager::new(4);
        let stats = manager.stats();
        assert_eq!(stats.max_parallel, 4);
    }

    #[test]
    fn test_manager_can_run_parallel() {
        let manager = ParallelCompactionManager::new(4);

        let job1 = create_test_job(1, b"a".to_vec(), b"m".to_vec(), true);
        let job2 = create_test_job(2, b"a".to_vec(), b"m".to_vec(), false);

        // Job with can_parallelize = true
        assert!(manager.can_run_parallel(&job1));

        // Job with can_parallelize = false
        assert!(!manager.can_run_parallel(&job2));
    }

    #[test]
    fn test_manager_execute_parallel() {
        let manager = ParallelCompactionManager::new(4);

        let job1 = create_test_job(1, b"a".to_vec(), b"m".to_vec(), true);

        let result = manager.execute_parallel(&job1, || 42);

        assert_eq!(result, 42);
        assert_eq!(manager.stats().active_jobs, 0);
    }

    #[test]
    fn test_empty_ranges() {
        let coordinator = ParallelCompactionCoordinator::new(4);

        let job1 = create_test_job(1, vec![], vec![], true);
        let job2 = create_test_job(2, b"a".to_vec(), b"z".to_vec(), true);

        // Empty ranges shouldn't overlap
        assert!(coordinator.try_acquire(&job1));
        coordinator.mark_active(&job1);
        assert!(coordinator.try_acquire(&job2));
    }

    #[test]
    fn test_adjacent_ranges() {
        let coordinator = ParallelCompactionCoordinator::new(4);

        let job1 = create_test_job(1, b"a".to_vec(), b"m".to_vec(), true);
        let job2 = create_test_job(2, b"m".to_vec(), b"z".to_vec(), true);

        assert!(coordinator.try_acquire(&job1));
        coordinator.mark_active(&job1);

        // Adjacent ranges overlap at boundary
        assert!(!coordinator.try_acquire(&job2));
    }
}
