//! I/O utilities for high-performance operations
//!
//! This module provides utilities for optimizing I/O operations:
//! - Buffer pooling to minimize allocations
//! - Parallel I/O operations
//! - Zero-copy techniques using Bytes

pub mod buffer_pool;
pub mod parallel_reader;
pub mod parallel_writer;

pub use buffer_pool::{
    BufferPool,
    BufferPoolStats,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_MAX_POOLED,
    PooledBuffer,
};
pub(crate) use parallel_reader::ReadResult;
pub use parallel_reader::{
    ParallelReader,
    ParallelReaderConfig,
};
pub use parallel_writer::{
    ParallelWriter,
    ParallelWriterConfig,
    WriteResult,
};
