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

pub mod job;
pub mod scheduler;
pub mod executor;
pub mod queue;
pub mod registry;
pub mod adaptive;
pub mod workload;
pub mod adaptor;
pub mod parallel;
pub mod subcompaction;
pub mod range_deletion;

pub use job::{CompactionJob, CompactionJobType};
pub use scheduler::CompactionScheduler;
pub use executor::CompactionExecutor;
pub use queue::{CompactionQueue, JobPriority, QueueStats};
pub use registry::{SegmentRegistry, RegistryStats};
pub use adaptive::{AdaptiveExecutor, ResourceLimits, ResourceUsage};
pub use workload::{WorkloadStats, WorkloadPattern, WorkloadAnalysis};
pub use adaptor::{WorkloadAdaptor, AdaptationPolicy, StrategyRecommendation};
pub use parallel::{ParallelCompactionCoordinator, ParallelCompactionManager, ParallelStats};
pub use subcompaction::{SubcompactionPlanner, SubcompactionConfig, SubcompactionJob, SubcompactionStats};
pub use range_deletion::{RangeTombstone, RangeTombstoneManager, RangeDeletionStats};
