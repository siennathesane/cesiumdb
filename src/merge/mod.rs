//! Merge iterators for combining sorted streams
//!
//! This module provides different merge iterator implementations:
//! - Basic merge iterator (original implementation)
//! - Zero-copy merge iterator (optimized for compaction)

pub mod basic;
pub mod zero_copy_merge;

pub use basic::MergeIterator;
pub use zero_copy_merge::{ZeroCopyMergeIterator, MergeError, MergeStats, MergeSource};
