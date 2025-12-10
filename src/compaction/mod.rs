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
    ResourceUsage,
};
pub use adaptor::{
    AdaptationPolicy,
    StrategyRecommendation,
    WorkloadAdaptor,
};
pub use executor::CompactionExecutor;
pub use job::{
    CompactionJob,
    CompactionJobType,
};
pub use parallel::{
    ParallelCompactionCoordinator,
    ParallelCompactionManager,
    ParallelStats,
};
pub use queue::{
    CompactionQueue,
    JobPriority,
    QueueStats,
};
pub use range_deletion::{
    RangeDeletionStats,
    RangeTombstone,
    RangeTombstoneManager,
};
pub use registry::{
    RegistryStats,
    SegmentRegistry,
};
pub use scheduler::CompactionScheduler;
pub use subcompaction::{
    SubcompactionConfig,
    SubcompactionJob,
    SubcompactionPlanner,
    SubcompactionStats,
};
pub use workload::{
    WorkloadAnalysis,
    WorkloadPattern,
    WorkloadStats,
};
