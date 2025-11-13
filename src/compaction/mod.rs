//! Multithreaded LSM-tree compaction
//!
//! This module implements the compaction system for the LSM-tree, including:
//! - Job structures for different compaction types
//! - Scheduling logic to pick optimal compactions
//! - Execution engine for running compactions
//! - Background thread coordination

pub mod job;
pub mod scheduler;
pub mod executor;

pub use job::{CompactionJob, CompactionJobType};
pub use scheduler::CompactionScheduler;
pub use executor::CompactionExecutor;
