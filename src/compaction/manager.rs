//! Compaction manager - orchestrates all compaction operations
//!
//! This is the main entry point for the compaction system, coordinating
//! scheduling, execution, and background threads.

use std::{
    collections::HashSet,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{
            AtomicBool,
            AtomicU64,
            Ordering,
        },
    },
    thread,
    time::Duration,
};

use parking_lot::{Mutex, RwLock};

use crate::{
    compaction::{
        AdaptationPolicy,
        AdaptiveExecutor,
        CompactionExecutor,
        CompactionJob,
        CompactionQueue,
        CompactionScheduler,
        ParallelCompactionManager,
        ResourceLimits,
        SegmentRegistry,
        SubcompactionPlanner,
        WorkloadAdaptor,
        WorkloadStats,
    },
    manifest_writer::ManifestWriter,
    version::VersionManager,
};

/// Main compaction manager
///
/// Coordinates all compaction activities including:
/// - Background compaction thread
/// - Adaptive scheduling and execution
/// - Manual compaction requests
pub struct CompactionManager {
    /// Scheduler for picking compaction jobs
    scheduler: Arc<CompactionScheduler>,

    /// Adaptive executor for running jobs
    executor: Option<AdaptiveExecutor>,

    /// Compaction queue
    queue: Arc<CompactionQueue>,

    /// Version manager
    version_manager: Arc<VersionManager>,

    /// Segment registry for reference tracking
    registry: Arc<SegmentRegistry>,

    /// Parallel compaction coordinator
    parallel_manager: Arc<ParallelCompactionManager>,

    /// Subcompaction planner
    subcompaction_planner: Arc<SubcompactionPlanner>,

    /// Workload statistics
    workload_stats: Arc<WorkloadStats>,

    /// Workload adaptor for dynamic strategy
    workload_adaptor: parking_lot::Mutex<WorkloadAdaptor>,

    /// Background thread handle
    bg_thread: Option<thread::JoinHandle<()>>,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Counter for failed compaction jobs
    failed_jobs: Arc<AtomicU64>,

    /// Tracks segments currently being compacted
    /// 
    /// Prevents duplicate job scheduling by tracking which segments
    /// are already in-flight. Cleared when jobs complete.
    in_flight_segments: Arc<RwLock<HashSet<u64>>>,
}

impl CompactionManager {
    /// Creates a new compaction manager
    pub fn new(
        base_path: PathBuf,
        version_manager: Arc<VersionManager>,
        manifest: Option<Arc<Mutex<ManifestWriter>>>,
    ) -> Self {
        let queue = Arc::new(CompactionQueue::new());
        let scheduler = Arc::new(CompactionScheduler::new(Arc::clone(&version_manager)));
        let registry = Arc::new(SegmentRegistry::new());
        let parallel_manager = Arc::new(ParallelCompactionManager::new(4));
        let subcompaction_planner = Arc::new(SubcompactionPlanner::new());
        let workload_stats = Arc::new(WorkloadStats::new());
        let workload_adaptor = parking_lot::Mutex::new(WorkloadAdaptor::new(
            Arc::clone(&workload_stats),
            AdaptationPolicy::default(),
        ));

        let executor_impl = Arc::new(CompactionExecutor::new(
            Arc::clone(&version_manager),
            base_path,
            manifest,
        ));

        let limits = ResourceLimits::default();
        let executor = AdaptiveExecutor::new(
            executor_impl,
            Arc::clone(&queue),
            Arc::clone(&version_manager),
            limits,
        );

        Self {
            scheduler,
            executor: Some(executor),
            queue,
            version_manager,
            registry,
            parallel_manager,
            subcompaction_planner,
            workload_stats,
            workload_adaptor,
            bg_thread: None,
            shutdown: Arc::new(AtomicBool::new(false)),
            failed_jobs: Arc::new(AtomicU64::new(0)),
            in_flight_segments: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    /// Starts the background compaction thread
    pub fn start(&mut self) {
        if self.bg_thread.is_some() {
            return; // Already started
        }

        let queue = Arc::clone(&self.queue);
        let scheduler = Arc::clone(&self.scheduler);
        let shutdown = Arc::clone(&self.shutdown);
        let version_manager = Arc::clone(&self.version_manager);

        let handle = thread::spawn(move || {
            Self::background_compaction_loop(queue, scheduler, shutdown, version_manager);
        });

        self.bg_thread = Some(handle);
    }

    /// Background compaction loop
    fn background_compaction_loop(
        queue: Arc<CompactionQueue>,
        scheduler: Arc<CompactionScheduler>,
        shutdown: Arc<AtomicBool>,
        version_manager: Arc<VersionManager>,
    ) {
        while !shutdown.load(Ordering::Relaxed) {
            // Check if we need to schedule new compactions
            let version = version_manager.current();

            // Try to pick a compaction job
            if let Some(job) = scheduler.pick_compaction(&version) {
                // Enqueue the job
                queue.enqueue(job);
            }

            // Sleep briefly before next check
            thread::sleep(Duration::from_millis(100));
        }
    }

    /// Triggers a manual compaction (user-requested)
    ///
    /// This will compact the entire database or a specific key range.
    pub fn compact(&self) {
        // Try to schedule compactions repeatedly until no more are needed
        // This will compact multiple levels if necessary
        for _ in 0..10 {
            // Get FRESH version each iteration (not stale snapshot)
            let version = self.version_manager.current();

            if let Some(job) = self.scheduler.pick_compaction(&version) {
                // Check for duplicates
                if self.is_duplicate_job(&job) {
                    continue; // Skip, try next iteration
                }

                // Mark segments as in-flight
                self.mark_in_flight(&job);
                self.queue.enqueue(job);
            } else {
                break; // No more compactions needed
            }
        }
    }

    /// Notifies the compaction manager of a memtable flush
    ///
    /// This triggers L0 compaction checks.
    pub fn notify_flush(&self) {
        let version = self.version_manager.current();

        // Check if any compaction is needed (prioritizes L0)
        if let Some(job) = self.scheduler.pick_compaction(&version) {
            self.queue.enqueue(job);
        }
    }

    /// Returns true if writes should be stalled due to too many L0 files.
    ///
    /// When L0 has too many files, compaction can't keep up and writes should
    /// be slowed down to give compaction time to catch up.
    pub fn should_stall_writes(&self) -> bool {
        const L0_STALL_TRIGGER: usize = 20;
        let version = self.version_manager.current();
        version.l0.len() >= L0_STALL_TRIGGER
    }

    /// Records a read operation for workload tracking
    pub fn record_read(&self, bytes_read: u64) {
        self.workload_stats.record_get(bytes_read);
    }

    /// Records a write operation for workload tracking
    pub fn record_write(&self, bytes_written: u64) {
        self.workload_stats.record_put(bytes_written);
    }

    /// Records a scan operation for workload tracking
    pub fn record_scan(&self, num_keys: u64, bytes_read: u64) {
        self.workload_stats.record_scan(num_keys, bytes_read);
    }

    /// Returns current compaction statistics
    pub fn stats(&self) -> CompactionStats {
        let queue_stats = self.queue.stats();
        let parallel_stats = self.parallel_manager.stats();
        let workload_analysis = self.workload_stats.analyze();

        CompactionStats {
            queued_jobs: queue_stats.queued,
            in_progress_jobs: queue_stats.in_progress,
            completed_jobs: queue_stats.completed,
            failed_jobs: self.failed_jobs.load(Ordering::Relaxed),
            parallel_utilization: parallel_stats.utilization,
            workload_pattern: format!("{:?}", workload_analysis.pattern),
        }
    }

    /// Shuts down the compaction manager
    pub fn shutdown(mut self) {
        // Signal shutdown
        self.shutdown.store(true, Ordering::Relaxed);

        // Shutdown queue
        self.queue.shutdown();

        // Wait for background thread
        if let Some(handle) = self.bg_thread.take() {
            let _result = handle.join();
        }

        // Shutdown executor
        if let Some(executor) = self.executor.take() {
            executor.shutdown();
        }
    }

    /// Checks if any input segments are already being compacted
    fn is_duplicate_job(&self, job: &CompactionJob) -> bool {
        let in_flight = self.in_flight_segments.read();

        // Check input segments
        for seg in &job.input.segments {
            if in_flight.contains(&seg.id()) {
                return true;
            }
        }

        // Check next level inputs
        if let Some(ref next_input) = job.next_level_input {
            for seg in &next_input.segments {
                if in_flight.contains(&seg.id()) {
                    return true;
                }
            }
        }

        false
    }

    /// Marks job segments as in-flight
    fn mark_in_flight(&self, job: &CompactionJob) {
        let mut in_flight = self.in_flight_segments.write();

        for seg in &job.input.segments {
            in_flight.insert(seg.id());
        }

        if let Some(ref next_input) = job.next_level_input {
            for seg in &next_input.segments {
                in_flight.insert(seg.id());
            }
        }
    }

    /// Clears in-flight status after job completes
    pub fn clear_in_flight(&self, job: &CompactionJob) {
        let mut in_flight = self.in_flight_segments.write();

        for seg in &job.input.segments {
            in_flight.remove(&seg.id());
        }

        if let Some(ref next_input) = job.next_level_input {
            for seg in &next_input.segments {
                in_flight.remove(&seg.id());
            }
        }
    }
}

impl Drop for CompactionManager {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        self.queue.shutdown();

        // Join background thread to prevent orphaned threads
        if let Some(handle) = self.bg_thread.take() {
            let _ = handle.join();
        }

        // Shutdown executor if still present
        if let Some(executor) = self.executor.take() {
            executor.shutdown();
        }
    }
}

/// Statistics about compaction operations
#[derive(Debug, Clone)]
pub struct CompactionStats {
    /// Number of jobs waiting to be processed
    pub queued_jobs: usize,

    /// Number of jobs currently being processed
    pub in_progress_jobs: usize,

    /// Total number of jobs completed
    pub completed_jobs: u64,

    /// Total number of failed compaction jobs
    pub failed_jobs: u64,

    /// Parallel execution utilization (0.0-1.0)
    pub parallel_utilization: f64,

    /// Current workload pattern
    pub workload_pattern: String,
}

impl std::fmt::Display for CompactionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Compaction: queued={}, in_progress={}, completed={}, failed={}, parallel_util={:.0}%, pattern={}",
            self.queued_jobs,
            self.in_progress_jobs,
            self.completed_jobs,
            self.failed_jobs,
            self.parallel_utilization * 100.0,
            self.workload_pattern
        )
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn test_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let version_manager = Arc::new(VersionManager::new(7));

        let manager = CompactionManager::new(temp_dir.path().to_path_buf(), version_manager, None);

        assert!(manager.bg_thread.is_none());
    }

    #[test]
    fn test_manager_start() {
        let temp_dir = TempDir::new().unwrap();
        let version_manager = Arc::new(VersionManager::new(7));

        let mut manager =
            CompactionManager::new(temp_dir.path().to_path_buf(), version_manager, None);

        manager.start();
        assert!(manager.bg_thread.is_some());

        manager.shutdown();
    }

    #[test]
    fn test_workload_tracking() {
        let temp_dir = TempDir::new().unwrap();
        let version_manager = Arc::new(VersionManager::new(7));

        let manager = CompactionManager::new(temp_dir.path().to_path_buf(), version_manager, None);

        manager.record_read(1000);
        manager.record_write(2000);
        manager.record_scan(10, 5000);

        let stats = manager.stats();
        assert_eq!(stats.queued_jobs, 0);
    }

    #[test]
    fn test_stats() {
        let temp_dir = TempDir::new().unwrap();
        let version_manager = Arc::new(VersionManager::new(7));

        let manager = CompactionManager::new(temp_dir.path().to_path_buf(), version_manager, None);

        let stats = manager.stats();
        assert_eq!(stats.queued_jobs, 0);
        assert_eq!(stats.in_progress_jobs, 0);
        assert_eq!(stats.completed_jobs, 0);
    }
}
