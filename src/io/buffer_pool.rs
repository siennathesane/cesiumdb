//! Buffer pool for reusable I/O buffers
//!
//! This module provides a thread-safe pool of reusable buffers to minimize
//! allocations during compaction and I/O operations.

use std::sync::{
    Arc,
    atomic::{
        AtomicU64,
        AtomicUsize,
        Ordering,
    },
};

use bytes::{
    Bytes,
    BytesMut,
};
use crossbeam_queue::SegQueue;

/// Default buffer size (64 KB)
pub const DEFAULT_BUFFER_SIZE: usize = 64 * 1024;

/// Maximum number of buffers to keep in the pool
pub const DEFAULT_MAX_POOLED: usize = 128;

/// A reusable buffer
///
/// When dropped, the buffer is returned to the pool if there's space.
pub struct PooledBuffer {
    /// The actual buffer data
    data: BytesMut,

    /// Reference to the pool (for returning on drop)
    pool: Arc<BufferPoolInner>,
}

impl PooledBuffer {
    /// Returns a mutable reference to the BytesMut
    #[inline]
    pub fn as_mut(&mut self) -> &mut BytesMut {
        &mut self.data
    }

    /// Returns an immutable slice of the buffer
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Returns the buffer capacity
    #[inline]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Returns the buffer length
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns whether the buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Clears the buffer (sets length to 0, keeps capacity)
    #[inline]
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Freezes the buffer into an immutable Bytes
    pub fn freeze(mut self) -> Bytes {
        // Take the data before drop runs
        let mut data = BytesMut::new();
        std::mem::swap(&mut data, &mut self.data);

        // Prevent the drop from returning to pool
        std::mem::forget(self);

        data.freeze()
    }

    /// Resizes the buffer to the given length
    #[inline]
    pub fn resize(&mut self, new_len: usize, value: u8) {
        self.data.resize(new_len, value);
    }

    /// Extends the buffer with data
    #[inline]
    pub fn extend_from_slice(&mut self, data: &[u8]) {
        self.data.extend_from_slice(data);
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        // Try to return the buffer to the pool
        if self.pool.pooled_count() < self.pool.max_pooled {
            // Clear the buffer before returning to pool
            self.data.clear();

            // Take ownership of the BytesMut
            let mut buffer = BytesMut::new();
            std::mem::swap(&mut buffer, &mut self.data);

            // Return to pool
            self.pool.return_buffer(buffer);
        }
        // Otherwise, drop the buffer (releases memory)
    }
}

impl std::ops::Deref for PooledBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::DerefMut for PooledBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl AsRef<[u8]> for PooledBuffer {
    fn as_ref(&self) -> &[u8] {
        &self.data
    }
}

impl AsMut<[u8]> for PooledBuffer {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

/// Inner buffer pool implementation
struct BufferPoolInner {
    /// Queue of available buffers
    buffers: SegQueue<BytesMut>,

    /// Buffer size for new allocations
    buffer_size: usize,

    /// Maximum number of buffers to keep pooled
    max_pooled: usize,

    /// Current number of pooled buffers
    pooled_count: AtomicUsize,

    /// Total number of buffers allocated
    total_allocated: AtomicU64,

    /// Number of times a buffer was reused from pool
    reuse_count: AtomicU64,

    /// Number of times we had to allocate a new buffer
    allocation_count: AtomicU64,
}

impl BufferPoolInner {
    fn new(buffer_size: usize, max_pooled: usize) -> Self {
        Self {
            buffers: SegQueue::new(),
            buffer_size,
            max_pooled,
            pooled_count: AtomicUsize::new(0),
            total_allocated: AtomicU64::new(0),
            reuse_count: AtomicU64::new(0),
            allocation_count: AtomicU64::new(0),
        }
    }

    fn get_buffer(&self) -> BytesMut {
        if let Some(buffer) = self.buffers.pop() {
            self.pooled_count.fetch_sub(1, Ordering::Relaxed);
            self.reuse_count.fetch_add(1, Ordering::Relaxed);
            buffer
        } else {
            // Allocate new buffer
            self.allocation_count.fetch_add(1, Ordering::Relaxed);
            self.total_allocated.fetch_add(1, Ordering::Relaxed);
            BytesMut::with_capacity(self.buffer_size)
        }
    }

    fn return_buffer(&self, buffer: BytesMut) {
        self.buffers.push(buffer);
        self.pooled_count.fetch_add(1, Ordering::Relaxed);
    }

    fn pooled_count(&self) -> usize {
        self.pooled_count.load(Ordering::Relaxed)
    }
}

/// Thread-safe buffer pool
///
/// The pool maintains a collection of reusable BytesMut buffers to minimize
/// allocation overhead during I/O operations.
#[derive(Clone)]
pub struct BufferPool {
    inner: Arc<BufferPoolInner>,
}

impl BufferPool {
    /// Creates a new buffer pool with default settings
    pub fn new() -> Self {
        Self::with_config(DEFAULT_BUFFER_SIZE, DEFAULT_MAX_POOLED)
    }

    /// Creates a new buffer pool with custom configuration
    ///
    /// # Arguments
    /// * `buffer_size` - Size of each buffer in bytes
    /// * `max_pooled` - Maximum number of buffers to keep in the pool
    pub fn with_config(buffer_size: usize, max_pooled: usize) -> Self {
        Self {
            inner: Arc::new(BufferPoolInner::new(buffer_size, max_pooled)),
        }
    }

    /// Gets a buffer from the pool
    ///
    /// If a buffer is available in the pool, it will be reused.
    /// Otherwise, a new buffer will be allocated.
    pub fn get(&self) -> PooledBuffer {
        let data = self.inner.get_buffer();

        PooledBuffer {
            data,
            pool: self.inner.clone(),
        }
    }

    /// Returns statistics about the buffer pool
    pub fn stats(&self) -> BufferPoolStats {
        BufferPoolStats {
            pooled: self.inner.pooled_count(),
            total_allocated: self.inner.total_allocated.load(Ordering::Relaxed),
            reuse_count: self.inner.reuse_count.load(Ordering::Relaxed),
            allocation_count: self.inner.allocation_count.load(Ordering::Relaxed),
            buffer_size: self.inner.buffer_size,
            max_pooled: self.inner.max_pooled,
        }
    }

    /// Returns the buffer size
    pub fn buffer_size(&self) -> usize {
        self.inner.buffer_size
    }

    /// Returns the maximum number of pooled buffers
    pub fn max_pooled(&self) -> usize {
        self.inner.max_pooled
    }

    /// Returns the current number of pooled buffers
    pub fn pooled_count(&self) -> usize {
        self.inner.pooled_count()
    }

    /// Calculates the reuse ratio (0.0 to 1.0)
    ///
    /// Higher values indicate better buffer reuse.
    pub fn reuse_ratio(&self) -> f64 {
        let reused = self.inner.reuse_count.load(Ordering::Relaxed);
        let allocated = self.inner.allocation_count.load(Ordering::Relaxed);
        let total = reused + allocated;

        if total == 0 {
            0.0
        } else {
            reused as f64 / total as f64
        }
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the buffer pool
#[derive(Debug, Clone, Copy)]
pub struct BufferPoolStats {
    /// Number of buffers currently in the pool
    pub pooled: usize,

    /// Total number of buffers allocated
    pub total_allocated: u64,

    /// Number of times a buffer was reused
    pub reuse_count: u64,

    /// Number of times a new buffer was allocated
    pub allocation_count: u64,

    /// Size of each buffer
    pub buffer_size: usize,

    /// Maximum buffers kept in pool
    pub max_pooled: usize,
}

impl std::fmt::Display for BufferPoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total_gets = self.reuse_count + self.allocation_count;
        let reuse_ratio = if total_gets == 0 {
            0.0
        } else {
            self.reuse_count as f64 / total_gets as f64 * 100.0
        };

        write!(
            f,
            "BufferPool: {} pooled, {} total allocated, {:.1}% reuse ratio",
            self.pooled, self.total_allocated, reuse_ratio
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let pool = BufferPool::new();
        assert_eq!(pool.buffer_size(), DEFAULT_BUFFER_SIZE);
        assert_eq!(pool.max_pooled(), DEFAULT_MAX_POOLED);
        assert_eq!(pool.pooled_count(), 0);
    }

    #[test]
    fn test_buffer_allocation() {
        let pool = BufferPool::new();
        let buffer = pool.get();

        assert_eq!(buffer.capacity(), DEFAULT_BUFFER_SIZE);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_buffer_return_to_pool() {
        let pool = BufferPool::new();

        {
            let _buffer = pool.get();
            assert_eq!(pool.pooled_count(), 0); // Buffer is in use
        } // Buffer dropped, returned to pool

        assert_eq!(pool.pooled_count(), 1); // Buffer returned
    }

    #[test]
    fn test_buffer_reuse() {
        let pool = BufferPool::new();

        // Get and return a buffer
        {
            let _buffer = pool.get();
        }

        assert_eq!(pool.pooled_count(), 1);

        // Get another buffer - should reuse the pooled one
        let buffer = pool.get();
        assert_eq!(pool.pooled_count(), 0);

        let stats = pool.stats();
        assert_eq!(stats.allocation_count, 1); // Only one allocation
        assert_eq!(stats.reuse_count, 1); // One reuse
    }

    #[test]
    fn test_max_pooled_limit() {
        let pool = BufferPool::with_config(1024, 2); // Max 2 buffers

        // Allocate 5 buffers and hold them
        let buffers: Vec<_> = (0..5).map(|_| pool.get()).collect();

        // Drop all buffers
        drop(buffers);

        // Should only keep 2 in pool
        assert_eq!(pool.pooled_count(), 2);
    }

    #[test]
    fn test_buffer_operations() {
        let pool = BufferPool::new();
        let mut buffer = pool.get();

        // Test resize
        buffer.resize(10, 0xff);
        assert_eq!(buffer.len(), 10);
        assert_eq!(buffer[0], 0xff);

        // Test clear
        buffer.clear();
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.capacity(), DEFAULT_BUFFER_SIZE);
    }

    #[test]
    fn test_buffer_extend() {
        let pool = BufferPool::new();
        let mut buffer = pool.get();

        buffer.extend_from_slice(b"hello");
        assert_eq!(buffer.len(), 5);
        assert_eq!(&buffer[..], b"hello");

        buffer.extend_from_slice(b" world");
        assert_eq!(buffer.len(), 11);
        assert_eq!(&buffer[..], b"hello world");
    }

    #[test]
    fn test_buffer_freeze() {
        let pool = BufferPool::new();
        let mut buffer = pool.get();

        buffer.extend_from_slice(b"test data");

        // Freeze into Bytes
        let bytes = buffer.freeze();
        assert_eq!(&bytes[..], b"test data");

        // Buffer should not be returned to pool after freeze
        assert_eq!(pool.pooled_count(), 0);
    }

    #[test]
    fn test_reuse_ratio() {
        let pool = BufferPool::new();

        // First get - allocation
        {
            let _buffer = pool.get();
        }

        // Second get - reuse
        {
            let _buffer = pool.get();
        }

        // Should be 50% reuse (1 alloc, 1 reuse)
        let ratio = pool.reuse_ratio();
        assert!((ratio - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;

        let pool = BufferPool::new();
        let pool_clone = pool.clone();

        // Spawn threads that get and release buffers
        let mut handles = vec![];
        for _ in 0..4 {
            let p = pool.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let mut buffer = p.get();
                    buffer.resize(1024, 0);
                    // Buffer dropped and returned
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Check that buffers were reused
        let stats = pool_clone.stats();
        assert!(stats.reuse_count > 0);
        assert!(stats.allocation_count < 400); // Should have reused many buffers
    }

    #[test]
    fn test_custom_buffer_size() {
        let pool = BufferPool::with_config(1024, 10);

        let buffer = pool.get();
        assert_eq!(buffer.capacity(), 1024);

        let stats = pool.stats();
        assert_eq!(stats.buffer_size, 1024);
    }

    #[test]
    fn test_stats_display() {
        let pool = BufferPool::new();

        {
            let _b1 = pool.get();
        }
        {
            let _b2 = pool.get();
        }

        let stats = pool.stats();
        let display = format!("{}", stats);
        assert!(display.contains("BufferPool"));
        assert!(display.contains("50.0%")); // 50% reuse ratio
    }

    #[test]
    fn test_bytes_mut_as_mut() {
        let pool = BufferPool::new();
        let mut buffer = pool.get();

        // Test mutable access via as_mut()
        buffer.as_mut().extend_from_slice(b"data");
        assert_eq!(buffer.len(), 4);
    }
}
