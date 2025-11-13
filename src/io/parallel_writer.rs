//! Parallel segment writing for high-throughput compaction
//!
//! This module provides parallel writing capabilities for segments,
//! allowing multiple blocks to be written concurrently during compaction.

use crate::block::{Block, BLOCK_SIZE};
use crate::io::buffer_pool::BufferPool;
use crate::map::Map;
use crossbeam_channel::{bounded, Receiver, Sender};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

/// A write task to be processed
pub struct WriteTask {
    /// Map to write to
    map: Arc<Map>,

    /// Block index to write to
    block_index: usize,

    /// Block to write
    block: Block,

    /// Task ID for tracking completion
    task_id: u64,
}

/// Result of a write operation
pub struct WriteResult {
    /// Task ID that completed
    pub task_id: u64,

    /// Number of bytes written (always BLOCK_SIZE for successful writes)
    pub bytes_written: usize,

    /// Whether the write succeeded
    pub success: bool,
}

/// Configuration for parallel writing
#[derive(Clone)]
pub struct ParallelWriterConfig {
    /// Number of writer threads to use
    pub num_threads: usize,

    /// Size of the work queue
    pub queue_size: usize,

    /// Buffer pool for allocations (currently unused but kept for consistency)
    pub buffer_pool: BufferPool,
}

impl ParallelWriterConfig {
    /// Creates a new config with default values
    pub fn new() -> Self {
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            num_threads,
            queue_size: 128,
            buffer_pool: BufferPool::new(),
        }
    }

    /// Sets the number of threads
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.num_threads = threads.max(1);
        self
    }

    /// Sets the queue size
    pub fn with_queue_size(mut self, size: usize) -> Self {
        self.queue_size = size.max(1);
        self
    }

    /// Sets the buffer pool
    pub fn with_buffer_pool(mut self, pool: BufferPool) -> Self {
        self.buffer_pool = pool;
        self
    }
}

impl Default for ParallelWriterConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Parallel segment writer
///
/// Spawns a pool of writer threads that process write requests
/// concurrently, maximizing I/O throughput during compaction.
pub struct ParallelWriter {
    /// Worker threads (wrapped in Option to allow taking in shutdown)
    workers: Option<Vec<thread::JoinHandle<()>>>,

    /// Channel for submitting write tasks
    task_sender: Sender<Option<WriteTask>>,

    /// Channel for receiving results
    result_receiver: Receiver<WriteResult>,

    /// Configuration
    config: ParallelWriterConfig,

    /// Next task ID
    next_task_id: Arc<AtomicU64>,

    /// Total bytes written (across all threads)
    total_bytes_written: Arc<AtomicU64>,
}

impl ParallelWriter {
    /// Creates a new parallel writer with the given configuration
    pub fn new(config: ParallelWriterConfig) -> Self {
        let (task_sender, task_receiver) = bounded::<Option<WriteTask>>(config.queue_size);
        let (result_sender, result_receiver) = bounded::<WriteResult>(config.queue_size);

        let total_bytes_written = Arc::new(AtomicU64::new(0));
        let mut workers = Vec::with_capacity(config.num_threads);

        // Spawn worker threads
        for worker_id in 0..config.num_threads {
            let task_rx = task_receiver.clone();
            let result_tx = result_sender.clone();
            let bytes_counter = total_bytes_written.clone();

            let worker = thread::Builder::new()
                .name(format!("parallel-writer-{}", worker_id))
                .spawn(move || {
                    Self::worker_loop(task_rx, result_tx, bytes_counter);
                })
                .expect("failed to spawn writer thread");

            workers.push(worker);
        }

        Self {
            workers: Some(workers),
            task_sender,
            result_receiver,
            config,
            next_task_id: Arc::new(AtomicU64::new(0)),
            total_bytes_written,
        }
    }

    /// Worker thread main loop
    fn worker_loop(
        task_rx: Receiver<Option<WriteTask>>,
        result_tx: Sender<WriteResult>,
        bytes_counter: Arc<AtomicU64>,
    ) {
        while let Ok(Some(task)) = task_rx.recv() {
            let offset = task.block_index * BLOCK_SIZE;
            let block_range = offset..(offset + BLOCK_SIZE);

            // Write the block to the map
            let success = task
                .map
                .write_to_range(block_range, |slice| unsafe {
                    // SAFETY: We know the block is exactly BLOCK_SIZE bytes
                    // and the space is available
                    task.block.finalize(slice.as_mut_ptr());
                })
                .is_ok();

            let bytes_written = if success { BLOCK_SIZE } else { 0 };

            // Update the global counter
            if success {
                bytes_counter.fetch_add(BLOCK_SIZE as u64, Ordering::Relaxed);
            }

            // Create result
            let result = WriteResult {
                task_id: task.task_id,
                bytes_written,
                success,
            };

            // Send result back
            // If the receiver is dropped, exit gracefully
            if result_tx.send(result).is_err() {
                break;
            }
        }
    }

    /// Submits a write task
    ///
    /// Returns the task ID if successful, or None if the queue is full.
    pub fn write_block(&self, map: Arc<Map>, block_index: usize, block: Block) -> Option<u64> {
        let task_id = self.next_task_id.fetch_add(1, Ordering::Relaxed);

        let task = WriteTask {
            map,
            block_index,
            block,
            task_id,
        };

        if self.task_sender.send(Some(task)).is_ok() {
            Some(task_id)
        } else {
            None
        }
    }

    /// Tries to receive a completed write result
    ///
    /// Returns None if no results are available.
    pub fn try_recv(&self) -> Option<WriteResult> {
        self.result_receiver.try_recv().ok()
    }

    /// Receives a completed write result (blocking)
    ///
    /// Returns None if all writer threads have exited.
    pub fn recv(&self) -> Option<WriteResult> {
        self.result_receiver.recv().ok()
    }

    /// Returns the number of pending tasks in the queue
    pub fn pending_tasks(&self) -> usize {
        self.task_sender.len()
    }

    /// Returns the number of available results
    pub fn available_results(&self) -> usize {
        self.result_receiver.len()
    }

    /// Returns the total number of bytes written
    pub fn total_bytes_written(&self) -> u64 {
        self.total_bytes_written.load(Ordering::Relaxed)
    }

    /// Returns statistics about the buffer pool
    pub fn buffer_pool_stats(&self) -> crate::io::buffer_pool::BufferPoolStats {
        self.config.buffer_pool.stats()
    }

    /// Shuts down the writer gracefully
    ///
    /// Waits for all in-flight tasks to complete and all threads to exit.
    pub fn shutdown(mut self) {
        // Send shutdown signals (None) to all workers
        for _ in 0..self.config.num_threads {
            let _ = self.task_sender.send(None);
        }

        // Wait for all workers to exit
        if let Some(workers) = self.workers.take() {
            for worker in workers {
                let _ = worker.join();
            }
        }
    }
}

impl Drop for ParallelWriter {
    fn drop(&mut self) {
        // Send shutdown signals to workers
        for _ in 0..self.config.num_threads {
            let _ = self.task_sender.send(None);
        }
        // Note: We don't join threads here since we can't move out of &mut self
        // The threads will exit when they receive None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::Block;
    use crate::map::Map;
    use tempfile::TempDir;

    fn create_test_map() -> Arc<Map> {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.map");
        Arc::new(Map::new(path, 10 * 1024 * 1024).unwrap())
    }

    #[test]
    fn test_writer_creation() {
        let config = ParallelWriterConfig::new();
        let writer = ParallelWriter::new(config);

        assert_eq!(writer.pending_tasks(), 0);
        assert_eq!(writer.available_results(), 0);
        assert_eq!(writer.total_bytes_written(), 0);

        writer.shutdown();
    }

    #[test]
    fn test_config_builder() {
        let config = ParallelWriterConfig::new()
            .with_threads(4)
            .with_queue_size(256);

        assert_eq!(config.num_threads, 4);
        assert_eq!(config.queue_size, 256);
    }

    #[test]
    fn test_write_task_submission() {
        let config = ParallelWriterConfig::new().with_queue_size(10);
        let writer = ParallelWriter::new(config);

        let map = create_test_map();
        let block = Block::new();

        // Submit a write task
        let task_id = writer.write_block(map, 0, block);
        assert!(task_id.is_some());

        writer.shutdown();
    }

    #[test]
    fn test_parallel_writes() {
        let config = ParallelWriterConfig::new()
            .with_threads(2)
            .with_queue_size(20);
        let writer = ParallelWriter::new(config);

        let map = create_test_map();

        // Submit multiple write tasks
        let mut task_ids = Vec::new();
        for i in 0..10 {
            let block = Block::new();
            if let Some(task_id) = writer.write_block(map.clone(), i, block) {
                task_ids.push(task_id);
            }
        }

        assert_eq!(task_ids.len(), 10);

        // Collect results
        let mut results = Vec::new();
        for _ in 0..10 {
            if let Some(result) = writer.recv() {
                results.push(result);
            } else {
                break;
            }
        }

        assert_eq!(results.len(), 10);

        // Verify all tasks succeeded
        for result in &results {
            assert!(result.success);
            assert_eq!(result.bytes_written, BLOCK_SIZE);
        }

        // Check total bytes written
        assert_eq!(writer.total_bytes_written(), (BLOCK_SIZE * 10) as u64);

        writer.shutdown();
    }

    #[test]
    fn test_shutdown() {
        let config = ParallelWriterConfig::new();
        let writer = ParallelWriter::new(config);

        // Shutdown should complete without hanging
        writer.shutdown();
    }

    #[test]
    fn test_task_id_generation() {
        let config = ParallelWriterConfig::new();
        let writer = ParallelWriter::new(config);

        let map = create_test_map();

        // Task IDs should increment
        let id1 = writer.write_block(map.clone(), 0, Block::new()).unwrap();
        let id2 = writer.write_block(map.clone(), 1, Block::new()).unwrap();
        let id3 = writer.write_block(map.clone(), 2, Block::new()).unwrap();

        assert!(id2 > id1);
        assert!(id3 > id2);

        writer.shutdown();
    }

    #[test]
    fn test_concurrent_writes() {
        use std::thread;

        let config = ParallelWriterConfig::new()
            .with_threads(4)
            .with_queue_size(100);
        let writer = Arc::new(ParallelWriter::new(config));

        let map = create_test_map();

        // Spawn multiple threads submitting writes
        let mut handles = vec![];
        for _ in 0..4 {
            let w = writer.clone();
            let m = map.clone();
            handles.push(thread::spawn(move || {
                for i in 0..25 {
                    w.write_block(m.clone(), i, Block::new());
                }
            }));
        }

        // Wait for all submissions
        for h in handles {
            h.join().unwrap();
        }

        // Collect all results
        let mut results = Vec::new();
        for _ in 0..100 {
            if let Some(result) = writer.recv() {
                results.push(result);
            } else {
                break;
            }
        }

        assert_eq!(results.len(), 100);
        assert_eq!(writer.total_bytes_written(), (BLOCK_SIZE * 100) as u64);

        // Can't call shutdown() because we have an Arc
        // The Drop implementation will handle cleanup
    }
}
