//! Multithreaded LSM-tree compaction
//!
//! This module implements the compaction system for the LSM-tree, including:
//! - Job structures for different compaction types
//! - Scheduling logic to pick optimal compactions
//! - Execution engine for running compactions
//! - Background thread coordination
//! - Lock-free job queue

pub mod job;
pub mod scheduler;
pub mod executor;
pub mod queue;
pub mod registry;

pub use job::{CompactionJob, CompactionJobType};
pub use scheduler::CompactionScheduler;
pub use executor::CompactionExecutor;
pub use queue::{CompactionQueue, JobPriority, QueueStats};
pub use registry::{SegmentRegistry, RegistryStats};
