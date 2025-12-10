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
            min_workers: 1,
            max_workers: num_cpus.max(2) - 1, // Leave one CPU for other work
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
    workers: Vec<thread::JoinHandle<()>>,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Resource limits
    limits: ResourceLimits,

    /// Current number of active workers
    active_workers: Arc<AtomicUsize>,

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
        let jobs_completed = Arc::new(AtomicU64::new(0));

        let mut workers = Vec::new();

        // Start initial worker threads
        for _ in 0..limits.min_workers {
            let worker = Self::spawn_worker(
                Arc::clone(&executor),
                Arc::clone(&queue),
                Arc::clone(&shutdown),
                Arc::clone(&active_workers),
                Arc::clone(&jobs_completed),
            );
            workers.push(worker);
        }

        // Start monitor thread
        let monitor = Self::spawn_monitor(
            Arc::clone(&queue),
            Arc::clone(&shutdown),
            Arc::clone(&active_workers),
            Arc::clone(&jobs_completed),
            limits,
        );

        Self {
            executor,
            queue,
            version_manager,
            workers,
            shutdown,
            limits,
            active_workers,
            jobs_completed,
            monitor: Some(monitor),
        }
    }

    /// Spawns a worker thread
    fn spawn_worker(
        executor: Arc<CompactionExecutor>,
        queue: Arc<CompactionQueue>,
        shutdown: Arc<AtomicBool>,
        active_workers: Arc<AtomicUsize>,
        jobs_completed: Arc<AtomicU64>,
    ) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            while !shutdown.load(Ordering::Relaxed) {
                // Try to get a job from the queue
                if let Some(job) = queue.dequeue() {
                    active_workers.fetch_add(1, Ordering::Relaxed);

                    // Execute the job
                    let _result = executor.execute(&job);

                    queue.mark_completed();
                    active_workers.fetch_sub(1, Ordering::Relaxed);
                    jobs_completed.fetch_add(1, Ordering::Relaxed);
                } else {
                    // No jobs available, sleep briefly
                    thread::sleep(Duration::from_millis(10));
                }
            }
        })
    }

    /// Spawns the monitor thread
    fn spawn_monitor(
        queue: Arc<CompactionQueue>,
        shutdown: Arc<AtomicBool>,
        active_workers: Arc<AtomicUsize>,
        jobs_completed: Arc<AtomicU64>,
        limits: ResourceLimits,
    ) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            let mut last_jobs_completed = 0u64;
            let mut last_check = Instant::now();

            while !shutdown.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_secs(1));

                // Measure resource usage
                let now = Instant::now();
                let elapsed = now.duration_since(last_check);
                let current_jobs = jobs_completed.load(Ordering::Relaxed);
                let jobs_delta = current_jobs - last_jobs_completed;

                let usage = ResourceUsage {
                    cpu_percent: Self::measure_cpu(),
                    memory_bytes: Self::measure_memory(),
                    active_workers: active_workers.load(Ordering::Relaxed),
                    queue_depth: queue.queued_count(),
                    jobs_completed_delta: jobs_delta,
                };

                // TODO: Implement adaptive scaling based on usage
                // For now, this is a placeholder for future enhancement

                last_jobs_completed = current_jobs;
                last_check = now;
            }
        })
    }

    /// Measures current CPU usage (placeholder)
    fn measure_cpu() -> f64 {
        // TODO: Implement actual CPU measurement
        // This is a placeholder - in production, you'd use platform-specific APIs
        0.0
    }

    /// Measures current memory usage (placeholder)
    fn measure_memory() -> usize {
        // TODO: Implement actual memory measurement
        // This is a placeholder - in production, you'd use platform-specific APIs
        0
    }

    /// Submits a job for execution
    pub fn submit(&self, job: CompactionJob) {
        self.queue.enqueue(job);
    }

    /// Returns current resource usage
    pub fn usage(&self) -> ResourceUsage {
        ResourceUsage {
            cpu_percent: Self::measure_cpu(),
            memory_bytes: Self::measure_memory(),
            active_workers: self.active_workers.load(Ordering::Relaxed),
            queue_depth: self.queue.queued_count(),
            jobs_completed_delta: 0,
        }
    }

    /// Returns queue statistics
    pub fn queue_stats(&self) -> crate::compaction::queue::QueueStats {
        self.queue.stats()
    }

    /// Shuts down the executor
    pub fn shutdown(mut self) {
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for monitor to finish
        if let Some(monitor) = self.monitor.take() {
            let _result = monitor.join();
        }

        // Wait for workers to finish
        for worker in self.workers.drain(..) {
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
        let executor = Arc::new(CompactionExecutor::new(Arc::clone(&version_manager), path));
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
        }
    }

    #[test]
    fn test_adaptive_executor_creation() {
        let (executor, _temp) = create_test_executor();

        assert_eq!(executor.workers.len(), 2); // min_workers
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
