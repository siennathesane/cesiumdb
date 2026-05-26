//! Multithreaded LSM-tree compaction
//!
//! This module implements the compaction system for the LSM-tree, including:
//! - Job structures for different compaction types
//! - Scheduling logic to pick optimal compactions
//! - Execution engine for running compactions
//! - Background thread coordination
//! - Lock-free job queue
//! - Adaptive resource management
//! - Workload-aware strategy selection

pub mod adaptive;
pub mod adaptor;
pub mod executor;
pub mod job;
pub mod manager;
pub mod parallel;
pub mod queue;
pub mod range_deletion;
pub mod registry;
pub mod scheduler;
pub mod subcompaction;
pub mod workload;

pub use adaptive::{
    AdaptiveExecutor,
    ResourceLimits,
};
pub use executor::CompactionExecutor;
pub use job::{
    CompactionJob,
    CompactionJobType,
};
pub use manager::{
    CompactionManager,
    CompactionStats,
};
pub use parallel::ParallelCompactionManager;
pub use queue::CompactionQueue;
pub use registry::SegmentRegistry;
pub use scheduler::{
    CompactionScheduler,
    SchedulerConfig,
};
pub use subcompaction::{
    SubcompactionJob,
    SubcompactionPlanner,
};
pub use workload::WorkloadStats;
