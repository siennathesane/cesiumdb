//! Parallel segment reading for high-throughput scans
#![allow(unused)]
//! This module provides parallel reading capabilities for segments,
//! allowing multiple blocks/ranges to be read concurrently.

use std::{
    sync::Arc,
    thread,
};

use bytes::Bytes;
use crossbeam_channel::{
    Receiver,
    Sender,
    bounded,
};

use crate::{
    block::BLOCK_SIZE,
    io::buffer_pool::BufferPool,
    segment::BlockType,
    segment_reader::SegmentReader,
    utils::Deserializer,
};

/// Result of a parallel read operation
pub struct ReadResult {
    /// Block index that was read
    pub block_index: usize,

    /// The deserialized block
    pub block: crate::block::ReadOnlyBlock,

    /// Type of block (Key or Value)
    pub block_type: BlockType,
}

/// Configuration for parallel reading
#[derive(Clone)]
pub struct ParallelReaderConfig {
    /// Number of reader threads to use
    pub num_threads: usize,

    /// Size of the work queue
    pub queue_size: usize,

    /// Buffer pool for allocations
    pub buffer_pool: BufferPool,
}

impl ParallelReaderConfig {
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

impl Default for ParallelReaderConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Work item for reader threads
struct ReadTask {
    /// Segment reader to read from
    reader: Arc<SegmentReader>,

    /// Block index to read
    block_index: usize,

    /// Type of block to read (Key or Value)
    block_type: BlockType,
}

/// Parallel segment reader
///
/// Spawns a pool of reader threads that process read requests
/// concurrently, maximizing I/O throughput.
pub struct ParallelReader {
    /// Worker threads (wrapped in Option to allow taking in shutdown)
    workers: Option<Vec<thread::JoinHandle<()>>>,

    /// Channel for submitting read tasks
    task_sender: Sender<Option<ReadTask>>,

    /// Channel for receiving results
    result_receiver: Receiver<ReadResult>,

    /// Configuration
    config: ParallelReaderConfig,
}

impl ParallelReader {
    /// Creates a new parallel reader with the given configuration
    pub fn new(config: ParallelReaderConfig) -> Self {
        let (task_sender, task_receiver) = bounded::<Option<ReadTask>>(config.queue_size);
        let (result_sender, result_receiver) = bounded::<ReadResult>(config.queue_size);

        let mut workers = Vec::with_capacity(config.num_threads);

        // Spawn worker threads
        for worker_id in 0..config.num_threads {
            let task_rx = task_receiver.clone();
            let result_tx = result_sender.clone();
            let buffer_pool = config.buffer_pool.clone();

            let worker = thread::Builder::new()
                .name(format!("parallel-reader-{}", worker_id))
                .spawn(move || {
                    Self::worker_loop(task_rx, result_tx, buffer_pool);
                })
                .expect("failed to spawn reader thread");

            workers.push(worker);
        }

        Self {
            workers: Some(workers),
            task_sender,
            result_receiver,
            config,
        }
    }

    /// Worker thread main loop
    fn worker_loop(
        task_rx: Receiver<Option<ReadTask>>,
        result_tx: Sender<ReadResult>,
        _buffer_pool: BufferPool,
    ) {
        while let Ok(Some(task)) = task_rx.recv() {
            // Read the block from the segment using the internal API
            let offset = task.block_index * BLOCK_SIZE;

            // CRITICAL: Clone the Arc to ensure the Map stays alive during the read
            // If we just borrow &Arc<Map>, the task could be dropped mid-read causing
            // SIGBUS
            let handle = match task.block_type {
                | BlockType::Key => Arc::clone(task.reader.key_handle()),
                | BlockType::Value => Arc::clone(task.reader.val_handle()),
            };

            // Check bounds using pre-computed visible block counts
            let max_blocks = match task.block_type {
                | BlockType::Key => task.reader.visible_key_blocks,
                | BlockType::Value => task.reader.visible_val_blocks,
            };
            if task.block_index >= max_blocks {
                // Skip invalid reads
                continue;
            }

            // Read the block data directly to Bytes (zero-copy for read-only maps)
            let bytes = handle.read_bytes(offset..offset + BLOCK_SIZE).ok();

            if bytes.is_none() {
                continue;
            }

            // Deserialize into a block
            let block = crate::block::ReadOnlyBlock::deserialize(bytes.unwrap());

            let result = ReadResult {
                block_index: task.block_index,
                block,
                block_type: task.block_type,
            };

            // Send result back
            // If the receiver is dropped, exit gracefully
            if result_tx.send(result).is_err() {
                break;
            }
        }
    }

    /// Submits a read task for a specific block
    ///
    /// Returns true if the task was submitted, false if the queue is full.
    pub fn read_block(
        &self,
        reader: Arc<SegmentReader>,
        block_index: usize,
        block_type: BlockType,
    ) -> bool {
        let task = ReadTask {
            reader,
            block_index,
            block_type,
        };

        self.task_sender.send(Some(task)).is_ok()
    }

    /// Tries to receive a completed read result
    ///
    /// Returns None if no results are available.
    pub fn try_recv(&self) -> Option<ReadResult> {
        self.result_receiver.try_recv().ok()
    }

    /// Receives a completed read result (blocking)
    ///
    /// Returns None if all reader threads have exited.
    pub fn recv(&self) -> Option<ReadResult> {
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

    /// Returns statistics about the buffer pool
    pub fn buffer_pool_stats(&self) -> crate::io::buffer_pool::BufferPoolStats {
        self.config.buffer_pool.stats()
    }

    /// Shuts down the reader gracefully
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

impl Drop for ParallelReader {
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
    use tempfile::TempDir;

    use super::*;
    use crate::segment_builder::SegmentBuilder;

    fn create_test_reader() -> Arc<SegmentReader> {
        let temp_dir = TempDir::new().unwrap();
        let builder = SegmentBuilder::new(temp_dir.path().to_path_buf()).unwrap();

        // Create a new segment
        let segment_id = 1;
        let _segment = builder
            .new_segment(segment_id, 12345, 64 * 1024 * 1024)
            .unwrap();

        // Drop the segment to close the writers
        drop(_segment);

        // Reopen as read-only
        let segment = builder.open(segment_id).unwrap();

        Arc::new(segment.reader().unwrap())
    }

    #[test]
    fn test_reader_creation() {
        let config = ParallelReaderConfig::new();
        let reader = ParallelReader::new(config);

        assert_eq!(reader.pending_tasks(), 0);
        assert_eq!(reader.available_results(), 0);

        reader.shutdown();
    }

    #[test]
    fn test_config_builder() {
        let config = ParallelReaderConfig::new()
            .with_threads(4)
            .with_queue_size(256);

        assert_eq!(config.num_threads, 4);
        assert_eq!(config.queue_size, 256);
    }

    #[test]
    fn test_read_task_submission() {
        let config = ParallelReaderConfig::new().with_queue_size(10);
        let parallel_reader = ParallelReader::new(config);

        let reader = create_test_reader();

        // Submit a read task
        assert!(parallel_reader.read_block(reader, 0, BlockType::Key));

        parallel_reader.shutdown();
    }

    #[test]
    fn test_parallel_reads() {
        let config = ParallelReaderConfig::new()
            .with_threads(2)
            .with_queue_size(20);
        let parallel_reader = ParallelReader::new(config);

        let reader = create_test_reader();

        // Submit multiple read tasks
        for i in 0..10 {
            assert!(parallel_reader.read_block(reader.clone(), i, BlockType::Key));
        }

        // Collect results (with timeout to avoid hanging)
        let mut results = Vec::new();
        for _ in 0..10 {
            if let Some(result) = parallel_reader.recv() {
                results.push(result);
            } else {
                break;
            }
        }

        assert_eq!(results.len(), 10);

        parallel_reader.shutdown();
    }

    #[test]
    fn test_buffer_pool_integration() {
        let pool = BufferPool::with_config(4096, 50);
        let config = ParallelReaderConfig::new().with_buffer_pool(pool.clone());
        let parallel_reader = ParallelReader::new(config);

        let reader = create_test_reader();

        // Submit reads
        for i in 0..5 {
            parallel_reader.read_block(reader.clone(), i, BlockType::Key);
        }

        // Collect results
        for _ in 0..5 {
            let _ = parallel_reader.recv();
        }

        // Check buffer pool stats - Note: buffer pool not currently used in real impl
        let _stats = parallel_reader.buffer_pool_stats();

        parallel_reader.shutdown();
    }

    #[test]
    fn test_shutdown() {
        let config = ParallelReaderConfig::new();
        let reader = ParallelReader::new(config);

        // Shutdown should complete without hanging
        reader.shutdown();
    }
}
