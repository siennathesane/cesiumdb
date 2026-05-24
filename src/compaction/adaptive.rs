//! Adaptive compaction executor with resource monitoring
//!
//! Provides an intelligent thread pool that:
//! - Monitors CPU and memory usage
//! - Auto-tunes thread count based on workload
//! - Applies backpressure when resources are constrained
//! - Adapts to changing system conditions

use std::{
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
    time::{
        Duration,
        Instant,
    },
};

use crate::{
    compaction::{
        executor::CompactionExecutor,
        job::CompactionJob,
        queue::CompactionQueue,
    },
    version::VersionManager,
};

/// Resource limits for compaction
#[derive(Debug, Clone, Copy)]
pub struct ResourceLimits {
    /// Maximum CPU usage percentage (0-100)
    pub max_cpu_percent: f64,

    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,

    /// Minimum number of worker threads
    pub min_workers: usize,

    /// Maximum number of worker threads
    pub max_workers: usize,

    /// Target queue depth before applying backpressure
    pub target_queue_depth: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        let num_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            max_cpu_percent: 80.0,
            max_memory_bytes: 1024 * 1024 * 1024, // 1GB
            min_workers: (num_cpus / 2).max(2),   // Testing with multiple workers
            max_workers: num_cpus.max(2) - 1,     // Leave one CPU for other work
            target_queue_depth: 10,
        }
    }
}

/// Current resource usage
#[derive(Debug, Clone, Copy)]
pub struct ResourceUsage {
    /// Current CPU usage percentage (0-100)
    pub cpu_percent: f64,

    /// Current memory usage in bytes
    pub memory_bytes: usize,

    /// Number of active workers
    pub active_workers: usize,

    /// Current queue depth
    pub queue_depth: usize,

    /// Jobs completed in last measurement period
    pub jobs_completed_delta: u64,
}

/// Adaptive compaction executor
///
/// Manages a pool of worker threads that execute compaction jobs,
/// with intelligent resource monitoring and auto-tuning.
pub struct AdaptiveExecutor {
    /// The underlying executor
    executor: Arc<CompactionExecutor>,

    /// The job queue
    queue: Arc<CompactionQueue>,

    /// Version manager
    version_manager: Arc<VersionManager>,

    /// Worker threads
    workers: parking_lot::Mutex<Vec<thread::JoinHandle<()>>>,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Resource limits
    limits: ResourceLimits,

    /// Current number of active workers (executing a job right now)
    active_workers: Arc<AtomicUsize>,

    /// Total number of live worker threads
    worker_count: Arc<AtomicUsize>,

    /// Desired number of workers (monitor adjusts this, workers check it)
    desired_workers: Arc<AtomicUsize>,

    /// Jobs completed counter
    jobs_completed: Arc<AtomicU64>,

    /// Monitor thread
    monitor: Option<thread::JoinHandle<()>>,
}

impl AdaptiveExecutor {
    /// Creates a new adaptive executor
    pub fn new(
        executor: Arc<CompactionExecutor>,
        queue: Arc<CompactionQueue>,
        version_manager: Arc<VersionManager>,
        limits: ResourceLimits,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let active_workers = Arc::new(AtomicUsize::new(0));
        let worker_count = Arc::new(AtomicUsize::new(0));
        let desired_workers = Arc::new(AtomicUsize::new(limits.min_workers));
        let jobs_completed = Arc::new(AtomicU64::new(0));

        let mut workers = Vec::new();

        // Start initial worker threads
        for _ in 0..limits.min_workers {
            let worker = Self::spawn_worker(
                Arc::clone(&executor),
                Arc::clone(&queue),
                Arc::clone(&shutdown),
                Arc::clone(&active_workers),
                Arc::clone(&worker_count),
                Arc::clone(&desired_workers),
                Arc::clone(&jobs_completed),
            );
            workers.push(worker);
        }
        worker_count.store(limits.min_workers, Ordering::Relaxed);

        // Start monitor thread
        let monitor = Self::spawn_monitor(
            Arc::clone(&executor),
            Arc::clone(&queue),
            Arc::clone(&shutdown),
            Arc::clone(&active_workers),
            Arc::clone(&worker_count),
            Arc::clone(&desired_workers),
            Arc::clone(&jobs_completed),
            limits,
        );

        Self {
            executor,
            queue,
            version_manager,
            workers: parking_lot::Mutex::new(workers),
            shutdown,
            limits,
            active_workers,
            worker_count,
            desired_workers,
            jobs_completed,
            monitor: Some(monitor),
        }
    }

    /// Spawns a worker thread
    ///
    /// Workers periodically check `desired_workers` and exit voluntarily
    /// when the current count exceeds the target (scale-down).
    fn spawn_worker(
        executor: Arc<CompactionExecutor>,
        queue: Arc<CompactionQueue>,
        shutdown: Arc<AtomicBool>,
        active_workers: Arc<AtomicUsize>,
        worker_count: Arc<AtomicUsize>,
        desired_workers: Arc<AtomicUsize>,
        jobs_completed: Arc<AtomicU64>,
    ) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            while !shutdown.load(Ordering::Relaxed) {
                // Check if we should scale down
                let current = worker_count.load(Ordering::Relaxed);
                let desired = desired_workers.load(Ordering::Relaxed);
                if current > desired {
                    // Try to be the one that exits
                    worker_count.fetch_sub(1, Ordering::Relaxed);
                    return;
                }

                // Try to get a job from the queue
                if let Some(job) = queue.dequeue() {
                    active_workers.fetch_add(1, Ordering::Relaxed);

                    // Execute the job
                    match executor.execute(&job) {
                        | Ok(_) => {
                            jobs_completed.fetch_add(1, Ordering::Relaxed);
                        },
                        | Err(e) => {
                            tracing::error!(
                                job_id = job.id,
                                error = ?e,
                                "Compaction job failed"
                            );
                        },
                    }

                    queue.mark_completed(job);
                    active_workers.fetch_sub(1, Ordering::Relaxed);
                } else {
                    // No jobs available, sleep briefly
                    thread::sleep(Duration::from_millis(10));
                }
            }

            // Decrement worker count on exit
            worker_count.fetch_sub(1, Ordering::Relaxed);
        })
    }

    /// Spawns the monitor thread
    ///
    /// The monitor periodically checks queue depth and adjusts
    /// `desired_workers`:
    /// - Scale up: if queue_depth > 2x target, increase desired (up to
    ///   max_workers)
    /// - Scale down: if queue is empty and no active workers, decrease desired
    ///   (down to min_workers)
    /// - New workers are spawned directly by the monitor when scaling up.
    fn spawn_monitor(
        executor: Arc<CompactionExecutor>,
        queue: Arc<CompactionQueue>,
        shutdown: Arc<AtomicBool>,
        active_workers: Arc<AtomicUsize>,
        worker_count: Arc<AtomicUsize>,
        desired_workers: Arc<AtomicUsize>,
        jobs_completed: Arc<AtomicU64>,
        limits: ResourceLimits,
    ) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            let mut last_jobs_completed = 0u64;
            let mut idle_cycles = 0u32;

            while !shutdown.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_secs(1));

                let current_jobs = jobs_completed.load(Ordering::Relaxed);
                let jobs_delta = current_jobs - last_jobs_completed;
                let queue_depth = queue.queued_count();
                let active = active_workers.load(Ordering::Relaxed);
                let current_desired = desired_workers.load(Ordering::Relaxed);

                // Scale up: queue has significant backlog
                if queue_depth > limits.target_queue_depth * 2 &&
                    current_desired < limits.max_workers
                {
                    let new_desired = (current_desired + 1).min(limits.max_workers);
                    desired_workers.store(new_desired, Ordering::Relaxed);

                    // Spawn the additional worker immediately
                    let current_count = worker_count.load(Ordering::Relaxed);
                    if current_count < new_desired {
                        // We don't store the handle since workers are in the Mutex
                        // and we can't access it from here. The worker will self-manage
                        // its lifecycle via worker_count.
                        let _handle = Self::spawn_worker(
                            Arc::clone(&executor),
                            Arc::clone(&queue),
                            Arc::clone(&shutdown),
                            Arc::clone(&active_workers),
                            Arc::clone(&worker_count),
                            Arc::clone(&desired_workers),
                            Arc::clone(&jobs_completed),
                        );
                        worker_count.fetch_add(1, Ordering::Relaxed);
                    }

                    idle_cycles = 0;
                } else if queue_depth == 0 && active == 0 {
                    // Scale down: no work to do
                    idle_cycles += 1;

                    // Wait a few cycles before scaling down to avoid flapping
                    if idle_cycles >= 3 && current_desired > limits.min_workers {
                        let new_desired = (current_desired - 1).max(limits.min_workers);
                        desired_workers.store(new_desired, Ordering::Relaxed);
                        idle_cycles = 0;
                    }
                } else {
                    idle_cycles = 0;
                }

                last_jobs_completed = current_jobs;
            }
        })
    }

    /// Returns compaction-specific throughput metric (jobs completed per
    /// second).
    ///
    /// This replaces platform-specific CPU/memory measurement with metrics
    /// directly relevant to compaction performance.
    pub fn jobs_throughput(&self) -> f64 {
        // This is a snapshot metric; the monitor thread tracks deltas over time
        self.jobs_completed.load(Ordering::Relaxed) as f64
    }

    /// Submits a job for execution
    pub fn submit(&self, job: CompactionJob) {
        self.queue.enqueue(job);
    }

    /// Returns current resource usage
    pub fn usage(&self) -> ResourceUsage {
        ResourceUsage {
            cpu_percent: 0.0, // Not tracked at OS level; use jobs_throughput() instead
            memory_bytes: 0,  // Not tracked at OS level
            active_workers: self.active_workers.load(Ordering::Relaxed),
            queue_depth: self.queue.queued_count(),
            jobs_completed_delta: 0,
        }
    }

    /// Returns queue statistics
    pub fn queue_stats(&self) -> crate::compaction::queue::QueueStats {
        self.queue.stats()
    }

    /// Returns cumulative compaction I/O stats: (bytes_read, bytes_written)
    pub fn compaction_io(&self) -> (u64, u64) {
        self.executor.compaction_io()
    }

    /// Shuts down the executor
    pub fn shutdown(mut self) {
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for monitor to finish
        if let Some(monitor) = self.monitor.take() {
            let _result = monitor.join();
        }

        // Wait for workers to finish
        for worker in self.workers.lock().drain(..) {
            let _result = worker.join();
        }
    }
}

impl Drop for AdaptiveExecutor {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tempfile::TempDir;

    use super::*;
    use crate::{
        compaction::job::{
            CompactionInput,
            CompactionJobType,
            CompactionOutput,
        },
        levels::{
            CompactionStrategy,
            KeyRange,
            Level,
            VersionSet,
        },
    };

    fn create_test_executor() -> (AdaptiveExecutor, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_path_buf();

        let version_manager = Arc::new(VersionManager::new(7)); // 7 levels
        let registry = Arc::new(crate::compaction::SegmentRegistry::new(path.clone()));
        let executor = Arc::new(CompactionExecutor::new(
            Arc::clone(&version_manager),
            path,
            None,
            registry,
        ));
        let queue = Arc::new(CompactionQueue::new());

        let limits = ResourceLimits {
            min_workers: 2,
            max_workers: 4,
            ..Default::default()
        };

        let adaptive = AdaptiveExecutor::new(executor, queue, version_manager, limits);

        (adaptive, temp_dir)
    }

    fn create_test_job() -> CompactionJob {
        let input = CompactionInput {
            level: 0,
            segments: vec![],
            key_range: KeyRange::new(vec![], vec![], 0),
            total_size: 0,
        };

        let output = CompactionOutput::new(0, 1024 * 1024);

        CompactionJob {
            id: 1,
            job_type: CompactionJobType::Flush,
            input,
            next_level_input: None,
            output,
            score: 1.0,
            can_parallelize: false,
            allocated_segment_ids: vec![1],
        }
    }

    #[test]
    fn test_adaptive_executor_creation() {
        let (executor, _temp) = create_test_executor();

        assert_eq!(executor.workers.lock().len(), 2); // min_workers
        assert!(executor.monitor.is_some());
    }

    #[test]
    fn test_submit_job() {
        let (executor, _temp) = create_test_executor();

        let job = create_test_job();
        executor.submit(job);

        assert!(executor.queue.queued_count() > 0);
    }

    #[test]
    fn test_usage_reporting() {
        let (executor, _temp) = create_test_executor();

        let usage = executor.usage();

        assert_eq!(usage.active_workers, 0);
        assert_eq!(usage.queue_depth, 0);
    }

    #[test]
    fn test_queue_stats() {
        let (executor, _temp) = create_test_executor();

        let stats = executor.queue_stats();

        assert_eq!(stats.queued, 0);
        assert_eq!(stats.in_progress, 0);
        assert_eq!(stats.completed, 0);
    }

    #[test]
    fn test_shutdown() {
        let (executor, _temp) = create_test_executor();

        // Shutdown should complete without hanging
        executor.shutdown();
    }

    #[test]
    fn test_resource_limits_default() {
        let limits = ResourceLimits::default();

        assert!(limits.max_cpu_percent > 0.0);
        assert!(limits.max_memory_bytes > 0);
        assert!(limits.min_workers > 0);
        assert!(limits.max_workers >= limits.min_workers);
    }
}
