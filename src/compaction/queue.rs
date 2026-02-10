//! Lock-free job queue for compaction coordination
//!
//! This module provides a thread-safe, lock-free queue for coordinating
//! compaction jobs across multiple worker threads.

use std::sync::{
    Arc,
    atomic::{
        AtomicBool,
        AtomicU64,
        AtomicUsize,
        Ordering,
    },
};

use crossbeam_queue::SegQueue;

use crate::compaction::job::CompactionJob;

/// Priority for a compaction job
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum JobPriority {
    /// Low priority (background cleanup)
    Low      = 0,
    /// Normal priority (regular compactions)
    Normal   = 1,
    /// High priority (L0 compaction)
    High     = 2,
    /// Critical priority (flush, blocks writes)
    Critical = 3,
}

impl JobPriority {
    /// Determines priority based on job score
    pub fn from_score(score: f64) -> Self {
        if score >= 100.0 {
            Self::Critical
        } else if score >= 50.0 {
            Self::High
        } else if score >= 10.0 {
            Self::Normal
        } else {
            Self::Low
        }
    }
}

/// A job in the queue with priority
struct QueuedJob {
    job: CompactionJob,
    priority: JobPriority,
    enqueued_at: std::time::Instant,
}

/// Lock-free compaction job queue
///
/// Uses separate queues per priority level for efficient scheduling.
/// Jobs are processed in priority order: Critical > High > Normal > Low.
pub struct CompactionQueue {
    /// Critical priority queue (flushes that block writes)
    critical: SegQueue<Arc<CompactionJob>>,

    /// High priority queue (L0 compactions)
    high: SegQueue<Arc<CompactionJob>>,

    /// Normal priority queue (regular level compactions)
    normal: SegQueue<Arc<CompactionJob>>,

    /// Low priority queue (cleanup, background tasks)
    low: SegQueue<Arc<CompactionJob>>,

    /// Total number of queued jobs across all priorities
    total_queued: AtomicUsize,

    /// Number of jobs currently being processed
    in_progress: AtomicUsize,

    /// Total number of jobs completed
    completed: AtomicU64,

    /// Whether the queue is shutting down
    shutdown: AtomicBool,
}

impl CompactionQueue {
    /// Creates a new empty compaction queue
    pub fn new() -> Self {
        Self {
            critical: SegQueue::new(),
            high: SegQueue::new(),
            normal: SegQueue::new(),
            low: SegQueue::new(),
            total_queued: AtomicUsize::new(0),
            in_progress: AtomicUsize::new(0),
            completed: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
        }
    }

    /// Enqueues a compaction job
    ///
    /// The job is placed in the appropriate priority queue based on its score.
    pub fn enqueue(&self, job: CompactionJob) {
        if self.shutdown.load(Ordering::Acquire) {
            return; // Don't accept new jobs during shutdown
        }

        let priority = JobPriority::from_score(job.score);
        let job = Arc::new(job);

        match priority {
            | JobPriority::Critical => self.critical.push(job),
            | JobPriority::High => self.high.push(job),
            | JobPriority::Normal => self.normal.push(job),
            | JobPriority::Low => self.low.push(job),
        }

        self.total_queued.fetch_add(1, Ordering::Release);
    }

    /// Dequeues the highest-priority job
    ///
    /// Returns None if the queue is empty or shutting down.
    pub fn dequeue(&self) -> Option<Arc<CompactionJob>> {
        if self.shutdown.load(Ordering::Acquire) {
            return None;
        }

        // Try queues in priority order
        let job = self
            .critical
            .pop()
            .or_else(|| self.high.pop())
            .or_else(|| self.normal.pop())
            .or_else(|| self.low.pop());

        let job = match job {
            | Some(j) => j,
            | None => return None,
        };

        self.total_queued.fetch_sub(1, Ordering::Release);
        self.in_progress.fetch_add(1, Ordering::Release);

        Some(job)
    }

    /// Marks a job as completed
    ///
    /// Should be called after a job finishes executing (success or failure).
    pub fn mark_completed(&self) {
        self.in_progress.fetch_sub(1, Ordering::Release);
        self.completed.fetch_add(1, Ordering::Release);
    }

    /// Returns the number of queued jobs (not yet started)
    pub fn queued_count(&self) -> usize {
        self.total_queued.load(Ordering::Acquire)
    }

    /// Returns the number of jobs currently being processed
    pub fn in_progress_count(&self) -> usize {
        self.in_progress.load(Ordering::Acquire)
    }

    /// Returns the total number of completed jobs
    pub fn completed_count(&self) -> u64 {
        self.completed.load(Ordering::Acquire)
    }

    /// Returns whether the queue is empty (no queued or in-progress jobs)
    pub fn is_empty(&self) -> bool {
        self.queued_count() == 0 && self.in_progress_count() == 0
    }

    /// Initiates graceful shutdown
    ///
    /// No new jobs will be accepted, but queued jobs will continue processing.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
    }

    /// Returns whether the queue is shutting down
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }

    /// Drains all pending jobs from the queue
    ///
    /// Returns all jobs that were queued but not yet started.
    /// Useful for cleanup during shutdown.
    pub fn drain(&self) -> Vec<Arc<CompactionJob>> {
        let mut jobs = Vec::new();

        while let Some(job) = self.critical.pop() {
            jobs.push(job);
        }
        while let Some(job) = self.high.pop() {
            jobs.push(job);
        }
        while let Some(job) = self.normal.pop() {
            jobs.push(job);
        }
        while let Some(job) = self.low.pop() {
            jobs.push(job);
        }

        self.total_queued.store(0, Ordering::Release);
        jobs
    }

    /// Returns statistics about the queue
    pub fn stats(&self) -> QueueStats {
        QueueStats {
            queued: self.queued_count(),
            in_progress: self.in_progress_count(),
            completed: self.completed_count(),
            is_shutdown: self.is_shutdown(),
        }
    }
}

impl Default for CompactionQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the compaction queue
#[derive(Debug, Clone, Copy)]
pub struct QueueStats {
    /// Number of jobs waiting to be processed
    pub queued: usize,
    /// Number of jobs currently being processed
    pub in_progress: usize,
    /// Total number of jobs completed since queue creation
    pub completed: u64,
    /// Whether the queue is in shutdown mode
    pub is_shutdown: bool,
}

impl std::fmt::Display for QueueStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Queue: {} queued, {} running, {} completed{}",
            self.queued,
            self.in_progress,
            self.completed,
            if self.is_shutdown { " [SHUTDOWN]" } else { "" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        compaction::job::{
            CompactionInput,
            CompactionJobType,
            CompactionOutput,
        },
        levels::KeyRange,
    };

    fn create_test_job(score: f64) -> CompactionJob {
        let input = CompactionInput {
            level: 0,
            segments: vec![],
            key_range: KeyRange::new(vec![], vec![], 0),
            total_size: 0,
        };
        let output = CompactionOutput::new(1, 64 * 1024 * 1024);

        CompactionJob::new(1, CompactionJobType::Flush, input, None, output)
    }

    #[test]
    fn test_queue_creation() {
        let queue = CompactionQueue::new();
        assert_eq!(queue.queued_count(), 0);
        assert_eq!(queue.in_progress_count(), 0);
        assert_eq!(queue.completed_count(), 0);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_priority_from_score() {
        assert_eq!(JobPriority::from_score(150.0), JobPriority::Critical);
        assert_eq!(JobPriority::from_score(75.0), JobPriority::High);
        assert_eq!(JobPriority::from_score(25.0), JobPriority::Normal);
        assert_eq!(JobPriority::from_score(5.0), JobPriority::Low);
    }

    #[test]
    fn test_enqueue_dequeue() {
        let queue = CompactionQueue::new();
        let mut job = create_test_job(50.0);
        job.score = 50.0; // High priority

        queue.enqueue(job);
        assert_eq!(queue.queued_count(), 1);

        let dequeued = queue.dequeue();
        assert!(dequeued.is_some());
        assert_eq!(queue.queued_count(), 0);
        assert_eq!(queue.in_progress_count(), 1);

        queue.mark_completed();
        assert_eq!(queue.in_progress_count(), 0);
        assert_eq!(queue.completed_count(), 1);
    }

    #[test]
    fn test_priority_ordering() {
        let queue = CompactionQueue::new();

        // Enqueue jobs in reverse priority order
        let mut low_job = create_test_job(1.0);
        low_job.score = 1.0;
        queue.enqueue(low_job);

        let mut normal_job = create_test_job(15.0);
        normal_job.score = 15.0;
        queue.enqueue(normal_job);

        let mut high_job = create_test_job(60.0);
        high_job.score = 60.0;
        queue.enqueue(high_job);

        let mut critical_job = create_test_job(150.0);
        critical_job.score = 150.0;
        queue.enqueue(critical_job);

        assert_eq!(queue.queued_count(), 4);

        // Should dequeue in priority order
        assert_eq!(queue.dequeue().unwrap().score, 150.0); // Critical
        assert_eq!(queue.dequeue().unwrap().score, 60.0); // High
        assert_eq!(queue.dequeue().unwrap().score, 15.0); // Normal
        assert_eq!(queue.dequeue().unwrap().score, 1.0); // Low
        assert!(queue.dequeue().is_none());
    }

    #[test]
    fn test_shutdown() {
        let queue = CompactionQueue::new();
        assert!(!queue.is_shutdown());

        queue.shutdown();
        assert!(queue.is_shutdown());

        // Should reject new jobs after shutdown
        let job = create_test_job(50.0);
        queue.enqueue(job);
        assert_eq!(queue.queued_count(), 0);
    }

    #[test]
    fn test_drain() {
        let queue = CompactionQueue::new();

        for i in 0..10 {
            let mut job = create_test_job(i as f64);
            job.score = i as f64;
            queue.enqueue(job);
        }

        assert_eq!(queue.queued_count(), 10);

        let drained = queue.drain();
        assert_eq!(drained.len(), 10);
        assert_eq!(queue.queued_count(), 0);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_concurrent_enqueue_dequeue() {
        use std::thread;

        let queue = Arc::new(CompactionQueue::new());

        // Spawn producer threads
        let mut producers = vec![];
        for _ in 0..4 {
            let q = queue.clone();
            producers.push(thread::spawn(move || {
                for i in 0..100 {
                    let mut job = create_test_job(i as f64);
                    job.score = i as f64;
                    q.enqueue(job);
                }
            }));
        }

        // Wait for producers to finish first
        for p in producers {
            p.join().unwrap();
        }

        // Now spawn consumer threads
        let mut consumers = vec![];
        for _ in 0..4 {
            let q = queue.clone();
            consumers.push(thread::spawn(move || {
                let mut count = 0;
                while let Some(_job) = q.dequeue() {
                    q.mark_completed();
                    count += 1;
                }
                count
            }));
        }

        // Wait for consumers
        let mut total_consumed = 0;
        for c in consumers {
            total_consumed += c.join().unwrap();
        }

        // All jobs should be processed
        assert_eq!(total_consumed, 400);
        assert_eq!(queue.completed_count(), 400);
    }

    #[test]
    fn test_stats() {
        let queue = CompactionQueue::new();

        let job = create_test_job(50.0);
        queue.enqueue(job);

        let stats = queue.stats();
        assert_eq!(stats.queued, 1);
        assert_eq!(stats.in_progress, 0);
        assert_eq!(stats.completed, 0);
        assert!(!stats.is_shutdown);

        queue.dequeue();
        let stats = queue.stats();
        assert_eq!(stats.queued, 0);
        assert_eq!(stats.in_progress, 1);
    }
}
