//! I/O utilities for high-performance operations
//!
//! This module provides utilities for optimizing I/O operations:
//! - Buffer pooling to minimize allocations
//! - Parallel I/O operations
//! - Zero-copy techniques using Bytes

pub mod buffer_pool;
pub mod parallel_reader;
pub mod parallel_writer;
