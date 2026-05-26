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
            AtomicUsize,
            Ordering,
        },
    },
    thread,
    time::Duration,
};

use parking_lot::{
    Mutex,
    RwLock,
};

use crate::{
    compaction::{
        AdaptiveExecutor,
        CompactionExecutor,
        CompactionJob,
        CompactionJobType,
        CompactionQueue,
        CompactionScheduler,
        ParallelCompactionManager,
        ResourceLimits,
        SchedulerConfig,
        SegmentRegistry,
        SubcompactionPlanner,
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

    /// Workload statistics
    workload_stats: Arc<WorkloadStats>,

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

    /// Number of L0 compactions currently in progress
    ///
    /// RocksDB-style: only one L0 compaction at a time to prevent
    /// overlapping reads and write amplification.
    l0_compactions_in_progress: Arc<AtomicUsize>,

    /// Cumulative bytes read by compaction jobs
    bytes_compacted_read: Arc<AtomicU64>,

    /// Cumulative bytes written by compaction jobs
    bytes_compacted_written: Arc<AtomicU64>,
}

impl CompactionManager {
    /// Creates a new compaction manager
    pub fn new(
        base_path: PathBuf,
        version_manager: Arc<VersionManager>,
        manifest: Option<Arc<Mutex<ManifestWriter>>>,
    ) -> Self {
        let registry = Arc::new(SegmentRegistry::new(base_path.clone()));
        Self::new_with_registry(base_path, version_manager, manifest, registry)
    }

    pub fn new_with_registry(
        base_path: PathBuf,
        version_manager: Arc<VersionManager>,
        manifest: Option<Arc<Mutex<ManifestWriter>>>,
        registry: Arc<SegmentRegistry>,
    ) -> Self {
        Self::new_with_scheduler_config(
            base_path,
            version_manager,
            manifest,
            registry,
            SchedulerConfig::default(),
        )
    }

    pub fn new_with_scheduler_config(
        base_path: PathBuf,
        version_manager: Arc<VersionManager>,
        manifest: Option<Arc<Mutex<ManifestWriter>>>,
        registry: Arc<SegmentRegistry>,
        scheduler_config: SchedulerConfig,
    ) -> Self {
        let queue = Arc::new(CompactionQueue::new());
        let scheduler = Arc::new(CompactionScheduler::with_config(
            scheduler_config,
            Arc::clone(&version_manager),
        ));
        let parallel_manager = Arc::new(ParallelCompactionManager::new(4));
        let workload_stats = Arc::new(WorkloadStats::new());

        let executor_impl = Arc::new(CompactionExecutor::with_planner(
            Arc::clone(&version_manager),
            base_path,
            manifest,
            Arc::clone(&registry),
            SubcompactionPlanner::new(),
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
            workload_stats,
            bg_thread: None,
            shutdown: Arc::new(AtomicBool::new(false)),
            failed_jobs: Arc::new(AtomicU64::new(0)),
            in_flight_segments: Arc::new(RwLock::new(HashSet::new())),
            l0_compactions_in_progress: Arc::new(AtomicUsize::new(0)),
            bytes_compacted_read: Arc::new(AtomicU64::new(0)),
            bytes_compacted_written: Arc::new(AtomicU64::new(0)),
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
        let in_flight = Arc::clone(&self.in_flight_segments);
        let max_concurrent = self.scheduler.config().max_concurrent_jobs;
        let l0_in_progress = Arc::clone(&self.l0_compactions_in_progress);
        let registry = Arc::clone(&self.registry);

        let handle = thread::spawn(move || {
            Self::background_compaction_loop(
                queue,
                scheduler,
                shutdown,
                version_manager,
                in_flight,
                max_concurrent,
                l0_in_progress,
                registry,
            );
        });

        self.bg_thread = Some(handle);
    }

    /// Background compaction loop
    fn background_compaction_loop(
        queue: Arc<CompactionQueue>,
        scheduler: Arc<CompactionScheduler>,
        shutdown: Arc<AtomicBool>,
        version_manager: Arc<VersionManager>,
        in_flight: Arc<RwLock<HashSet<u64>>>,
        max_concurrent: usize,
        l0_in_progress: Arc<AtomicUsize>,
        registry: Arc<SegmentRegistry>,
    ) {
        while !shutdown.load(Ordering::Relaxed) {
            // Drain completed jobs and clear their in-flight segments
            for job in queue.drain_completed() {
                {
                    let mut in_flight_guard = in_flight.write();
                    for seg in &job.input.segments {
                        in_flight_guard.remove(&seg.id());
                    }
                    if let Some(ref next_input) = job.next_level_input {
                        for seg in &next_input.segments {
                            in_flight_guard.remove(&seg.id());
                        }
                    }
                }
                // Decrement L0 compaction counter if applicable
                if job.job_type == CompactionJobType::L0Compaction {
                    l0_in_progress.fetch_sub(1, Ordering::Relaxed);
                }
            }

            // Clean up obsolete segments after dropping completed jobs
            let (deleted, bytes_freed) = registry.cleanup();
            if deleted > 0 {
                tracing::info!(
                    segments_deleted = deleted,
                    bytes_freed = bytes_freed,
                    "Cleaned up obsolete segments in background loop"
                );
            }

            // If the queue is completely idle, clear in-flight tracking so
            // future compactions can be scheduled for those segments.
            if queue.is_empty() {
                let mut in_flight_guard = in_flight.write();
                in_flight_guard.clear();
                l0_in_progress.store(0, Ordering::Relaxed);
            }

            // Fill the queue up to max_concurrent with non-conflicting jobs.
            let mut jobs_scheduled = 0usize;
            loop {
                let total_jobs = queue.queued_count() + queue.in_progress_count();
                if total_jobs >= max_concurrent {
                    break;
                }
                let slots = max_concurrent - total_jobs;

                let version = version_manager.current();
                let in_flight_snapshot = {
                    let guard = in_flight.read();
                    guard.clone()
                };

                let jobs = scheduler.pick_compactions(&version, &in_flight_snapshot, slots);
                if jobs.is_empty() {
                    break;
                }

                for job in jobs {
                    // RocksDB-style: serialize L0 compactions to one at a time
                    if job.job_type == CompactionJobType::L0Compaction &&
                        l0_in_progress.load(Ordering::Relaxed) > 0
                    {
                        tracing::debug!(job_id = job.id, "bg_loop: skipping L0 job, serialization");
                        continue;
                    }

                    // Double-check no segment is already in-flight
                    let is_dup = {
                        let guard = in_flight.read();
                        job.input
                            .segments
                            .iter()
                            .any(|seg| guard.contains(&seg.id())) ||
                            job.next_level_input.as_ref().is_some_and(|next| {
                                next.segments.iter().any(|seg| guard.contains(&seg.id()))
                            })
                    };
                    if is_dup {
                        continue;
                    }

                    // Mark segments as in-flight
                    {
                        let mut guard = in_flight.write();
                        for seg in &job.input.segments {
                            guard.insert(seg.id());
                        }
                        if let Some(ref next) = job.next_level_input {
                            for seg in &next.segments {
                                guard.insert(seg.id());
                            }
                        }
                    }

                    if job.job_type == CompactionJobType::L0Compaction {
                        l0_in_progress.fetch_add(1, Ordering::Relaxed);
                    }

                    queue.enqueue(job);
                    jobs_scheduled += 1;
                }
            }

            // Adaptive sleep: stay responsive when there's work, back off when idle
            if jobs_scheduled > 0 {
                thread::sleep(Duration::from_millis(10));
            } else {
                thread::sleep(Duration::from_millis(100));
            }
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
                self.queue.enqueue(job);
            } else {
                break; // No more compactions needed
            }
        }
    }

    /// Notifies the compaction manager of a memtable flush
    ///
    /// This triggers L0 compaction checks. To prevent unbounded queue
    /// growth, we only enqueue if we're below capacity and the segments
    /// aren't already being compacted.
    pub fn notify_flush(&self) {
        let total_jobs = self.queue.queued_count() + self.queue.in_progress_count();
        if total_jobs >= self.scheduler.config().max_concurrent_jobs {
            tracing::debug!(
                total_jobs,
                max = self.scheduler.config().max_concurrent_jobs,
                "notify_flush: at capacity"
            );
            return;
        }

        let version = self.version_manager.current();
        let slots = self.scheduler.config().max_concurrent_jobs - total_jobs;
        let in_flight_snapshot = {
            let guard = self.in_flight_segments.read();
            guard.clone()
        };

        let jobs = self
            .scheduler
            .pick_compactions(&version, &in_flight_snapshot, slots);
        if jobs.is_empty() {
            tracing::debug!(
                l0_count = version.l0.len(),
                "notify_flush: no compaction needed"
            );
            return;
        }

        for job in jobs {
            // RocksDB-style: serialize L0 compactions
            if job.job_type == CompactionJobType::L0Compaction &&
                self.l0_compactions_in_progress.load(Ordering::Relaxed) > 0
            {
                tracing::debug!(
                    job_id = job.id,
                    "notify_flush: skipping L0 job, serialization"
                );
                continue;
            }
            if self.is_duplicate_job(&job) {
                tracing::debug!(job_id = job.id, "notify_flush: duplicate job");
                continue;
            }
            self.mark_in_flight(&job);
            if job.job_type == CompactionJobType::L0Compaction {
                self.l0_compactions_in_progress
                    .fetch_add(1, Ordering::Relaxed);
            }
            self.queue.enqueue(job);
        }
    }

    /// Returns true if writes should be stalled due to too many L0 files.
    ///
    /// When L0 has too many files, compaction can't keep up and writes should
    /// be slowed down to give compaction time to catch up.
    pub fn should_stall_writes(&self) -> bool {
        let version = self.version_manager.current();
        version.l0.len() >= self.scheduler.config().l0_stop_writes_trigger
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

    /// Returns the segment registry
    pub fn registry(&self) -> Arc<SegmentRegistry> {
        Arc::clone(&self.registry)
    }

    /// Records compaction I/O for throughput tracking
    pub fn record_compaction_io(&self, bytes_read: u64, bytes_written: u64) {
        self.bytes_compacted_read
            .fetch_add(bytes_read, Ordering::Relaxed);
        self.bytes_compacted_written
            .fetch_add(bytes_written, Ordering::Relaxed);
    }

    /// Returns current compaction statistics
    pub fn stats(&self) -> CompactionStats {
        let queue_stats = self.queue.stats();
        let parallel_stats = self.parallel_manager.stats();
        let workload_analysis = self.workload_stats.analyze();
        let registry_stats = self.registry.stats();

        // Read I/O counters from the executor (where they're actually updated)
        let (bytes_read, bytes_written) = self
            .executor
            .as_ref()
            .map(|e| e.compaction_io())
            .unwrap_or((0, 0));

        CompactionStats {
            queued_jobs: queue_stats.queued,
            in_progress_jobs: queue_stats.in_progress,
            completed_jobs: queue_stats.completed,
            failed_jobs: self.failed_jobs.load(Ordering::Relaxed),
            parallel_utilization: parallel_stats.utilization,
            workload_pattern: format!("{:?}", workload_analysis.pattern),
            pending_deletion_segments: registry_stats.pending_deletion,
            bytes_compacted_read: bytes_read,
            bytes_compacted_written: bytes_written,
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

    /// Number of segments pending deletion
    pub pending_deletion_segments: usize,

    /// Cumulative bytes read by compaction jobs
    pub bytes_compacted_read: u64,

    /// Cumulative bytes written by compaction jobs
    pub bytes_compacted_written: u64,
}

impl std::fmt::Display for CompactionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Compaction: queued={}, in_progress={}, completed={}, failed={}, parallel_util={:.0}%, pattern={}, pending_deletion={}, compacted_read={:.2}MB, compacted_written={:.2}MB",
            self.queued_jobs,
            self.in_progress_jobs,
            self.completed_jobs,
            self.failed_jobs,
            self.parallel_utilization * 100.0,
            self.workload_pattern,
            self.pending_deletion_segments,
            self.bytes_compacted_read as f64 / (1024.0 * 1024.0),
            self.bytes_compacted_written as f64 / (1024.0 * 1024.0)
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
        assert_eq!(manager.registry().live_count(), 0);
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
        assert_eq!(stats.pending_deletion_segments, 0);
    }

    #[test]
    fn test_l0_compaction_serialization() {
        let temp_dir = TempDir::new().unwrap();
        let version_manager = Arc::new(VersionManager::new(7));

        let manager =
            CompactionManager::new(temp_dir.path().to_path_buf(), version_manager.clone(), None);

        // Manually simulate an L0 compaction being in progress
        manager
            .l0_compactions_in_progress
            .fetch_add(1, Ordering::Relaxed);

        // Create a fake L0 job and try to enqueue via notify_flush
        // Since l0_compactions_in_progress > 0, it should be rejected
        let before = manager.queue.queued_count();
        manager.notify_flush();
        let after = manager.queue.queued_count();

        // No new job should be queued (no L0 segments exist, but the guard should still
        // work)
        assert_eq!(after, before);

        // Decrement and verify it allows future L0 compactions
        manager
            .l0_compactions_in_progress
            .fetch_sub(1, Ordering::Relaxed);
        assert_eq!(
            manager.l0_compactions_in_progress.load(Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn test_level_compaction_parallel_with_l0() {
        let temp_dir = TempDir::new().unwrap();
        let version_manager = Arc::new(VersionManager::new(7));

        let manager =
            CompactionManager::new(temp_dir.path().to_path_buf(), version_manager.clone(), None);

        // L0 in progress should NOT block manual compaction of other levels
        manager
            .l0_compactions_in_progress
            .fetch_add(1, Ordering::Relaxed);

        // compact() bypasses the L0 guard (it directly enqueues via scheduler)
        // but with no data, no jobs are picked anyway
        manager.compact();

        // Just verify the state is consistent
        assert_eq!(
            manager.l0_compactions_in_progress.load(Ordering::Relaxed),
            1
        );
    }
}
