//! Workload-adaptive compaction strategy selector
//!
//! Automatically adjusts compaction strategies based on observed workload patterns.

use crate::compaction::workload::{WorkloadAnalysis, WorkloadPattern, WorkloadStats};
use crate::levels::CompactionStrategy;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Adaptation policy configuration
#[derive(Debug, Clone)]
pub struct AdaptationPolicy {
    /// Minimum confidence level to trigger adaptation (0.0-1.0)
    pub min_confidence: f64,

    /// Minimum time between strategy changes
    pub min_adaptation_interval: Duration,

    /// Minimum number of operations before first adaptation
    pub min_ops_before_adapt: u64,

    /// Read amplification threshold for switching strategies
    pub read_amp_threshold: f64,

    /// Write amplification threshold for switching strategies
    pub write_amp_threshold: f64,
}

impl Default for AdaptationPolicy {
    fn default() -> Self {
        Self {
            min_confidence: 0.7,
            min_adaptation_interval: Duration::from_secs(300), // 5 minutes
            min_ops_before_adapt: 10000,
            read_amp_threshold: 5.0,
            write_amp_threshold: 20.0,
        }
    }
}

/// Workload-adaptive compaction strategy selector
pub struct WorkloadAdaptor {
    /// Workload statistics tracker
    stats: Arc<WorkloadStats>,

    /// Adaptation policy
    policy: AdaptationPolicy,

    /// Last adaptation time
    last_adaptation: Option<Instant>,

    /// Current recommended strategy
    current_strategy: Option<CompactionStrategy>,
}

impl WorkloadAdaptor {
    /// Creates a new workload adaptor
    pub fn new(stats: Arc<WorkloadStats>, policy: AdaptationPolicy) -> Self {
        Self {
            stats,
            policy,
            last_adaptation: None,
            current_strategy: None,
        }
    }

    /// Analyzes the current workload and returns recommended strategy
    ///
    /// Returns None if:
    /// - Not enough data to make a confident decision
    /// - Too soon after last adaptation
    /// - Current strategy is already optimal
    pub fn recommend_strategy(&mut self) -> Option<StrategyRecommendation> {
        let analysis = self.stats.analyze();

        // Check if we have enough confidence
        if analysis.confidence < self.policy.min_confidence {
            return None;
        }

        // Check if we have enough operations
        let snapshot = self.stats.snapshot();
        let total_ops = snapshot.gets + snapshot.puts + snapshot.deletes + snapshot.scans;
        if total_ops < self.policy.min_ops_before_adapt {
            return None;
        }

        // Check if enough time has passed since last adaptation
        if let Some(last) = self.last_adaptation {
            if last.elapsed() < self.policy.min_adaptation_interval {
                return None;
            }
        }

        // Determine recommended strategy based on workload pattern
        let recommended = match analysis.pattern {
            WorkloadPattern::WriteHeavy => {
                // Minimize write amplification
                CompactionStrategy::Leveled {
                    fanout: 10,
                    target_file_count: 10,
                }
            }
            WorkloadPattern::ReadHeavy => {
                // Minimize read amplification
                CompactionStrategy::Tiered {
                    size_ratio: 2.0,
                    min_merge_width: 2,
                    max_merge_width: 4,
                }
            }
            WorkloadPattern::ScanHeavy => {
                // Non-overlapping ranges help scans
                CompactionStrategy::Leveled {
                    fanout: 10,
                    target_file_count: 10,
                }
            }
            WorkloadPattern::PointLookup => {
                // Bloom filters help point lookups
                CompactionStrategy::Tiered {
                    size_ratio: 2.0,
                    min_merge_width: 2,
                    max_merge_width: 4,
                }
            }
            WorkloadPattern::Balanced => {
                // Hybrid approach
                CompactionStrategy::Leveled {
                    fanout: 8,
                    target_file_count: 8,
                }
            }
        };

        // Check if we should switch strategies
        let should_switch = if let Some(ref current) = self.current_strategy {
            !self.strategies_equivalent(current, &recommended)
        } else {
            true // First recommendation
        };

        if !should_switch {
            return None;
        }

        // Check amplification thresholds
        let reason = if analysis.read_amplification > self.policy.read_amp_threshold {
            ChangeReason::HighReadAmplification(analysis.read_amplification)
        } else if analysis.write_amplification > self.policy.write_amp_threshold {
            ChangeReason::HighWriteAmplification(analysis.write_amplification)
        } else {
            ChangeReason::WorkloadPatternChange(analysis.pattern)
        };

        self.last_adaptation = Some(Instant::now());
        self.current_strategy = Some(recommended.clone());

        Some(StrategyRecommendation {
            strategy: recommended,
            reason,
            analysis: analysis.clone(),
        })
    }

    /// Checks if two strategies are equivalent
    fn strategies_equivalent(&self, a: &CompactionStrategy, b: &CompactionStrategy) -> bool {
        match (a, b) {
            (
                CompactionStrategy::Tiered { .. },
                CompactionStrategy::Tiered { .. },
            ) => true,
            (
                CompactionStrategy::Leveled { .. },
                CompactionStrategy::Leveled { .. },
            ) => true,
            (
                CompactionStrategy::Universal { .. },
                CompactionStrategy::Universal { .. },
            ) => true,
            _ => false,
        }
    }

    /// Returns the current workload analysis
    pub fn current_analysis(&self) -> WorkloadAnalysis {
        self.stats.analyze()
    }

    /// Resets adaptation state (useful for testing)
    pub fn reset(&mut self) {
        self.last_adaptation = None;
        self.current_strategy = None;
    }
}

/// Reason for strategy change
#[derive(Debug, Clone)]
pub enum ChangeReason {
    /// Workload pattern changed
    WorkloadPatternChange(WorkloadPattern),

    /// Read amplification exceeded threshold
    HighReadAmplification(f64),

    /// Write amplification exceeded threshold
    HighWriteAmplification(f64),
}

impl std::fmt::Display for ChangeReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WorkloadPatternChange(pattern) => {
                write!(f, "Workload pattern changed to {:?}", pattern)
            }
            Self::HighReadAmplification(amp) => {
                write!(f, "High read amplification ({:.2}x)", amp)
            }
            Self::HighWriteAmplification(amp) => {
                write!(f, "High write amplification ({:.2}x)", amp)
            }
        }
    }
}

/// Strategy recommendation with justification
#[derive(Debug, Clone)]
pub struct StrategyRecommendation {
    /// Recommended strategy
    pub strategy: CompactionStrategy,

    /// Reason for the recommendation
    pub reason: ChangeReason,

    /// Workload analysis that led to this recommendation
    pub analysis: WorkloadAnalysis,
}

impl std::fmt::Display for StrategyRecommendation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Recommend {:?} strategy: {} ({})",
            self.strategy, self.reason, self.analysis
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptor_creation() {
        let stats = Arc::new(WorkloadStats::new());
        let policy = AdaptationPolicy::default();
        let adaptor = WorkloadAdaptor::new(stats, policy);

        assert!(adaptor.last_adaptation.is_none());
        assert!(adaptor.current_strategy.is_none());
    }

    #[test]
    fn test_not_enough_confidence() {
        let stats = Arc::new(WorkloadStats::new());
        let policy = AdaptationPolicy {
            min_confidence: 0.9,
            ..Default::default()
        };
        let mut adaptor = WorkloadAdaptor::new(stats, policy);

        // Not enough operations for high confidence
        assert!(adaptor.recommend_strategy().is_none());
    }

    #[test]
    fn test_write_heavy_recommendation() {
        let stats = Arc::new(WorkloadStats::new());
        let policy = AdaptationPolicy {
            min_ops_before_adapt: 100,
            min_confidence: 0.6,
            ..Default::default()
        };
        let mut adaptor = WorkloadAdaptor::new(Arc::clone(&stats), policy);

        // Simulate write-heavy workload
        for _ in 0..800 {
            stats.record_put(1000);
        }
        for _ in 0..200 {
            stats.record_get(1000);
        }

        let recommendation = adaptor.recommend_strategy();
        assert!(recommendation.is_some());

        let rec = recommendation.unwrap();
        assert!(matches!(rec.strategy, CompactionStrategy::Leveled { .. }));
        assert_eq!(rec.analysis.pattern, WorkloadPattern::WriteHeavy);
    }

    #[test]
    fn test_read_heavy_recommendation() {
        let stats = Arc::new(WorkloadStats::new());
        let policy = AdaptationPolicy {
            min_ops_before_adapt: 100,
            min_confidence: 0.6,
            ..Default::default()
        };
        let mut adaptor = WorkloadAdaptor::new(Arc::clone(&stats), policy);

        // Simulate read-heavy workload
        for _ in 0..800 {
            stats.record_get(1000);
        }
        for _ in 0..200 {
            stats.record_put(1000);
        }

        let recommendation = adaptor.recommend_strategy();
        assert!(recommendation.is_some());

        let rec = recommendation.unwrap();
        assert!(matches!(rec.strategy, CompactionStrategy::Tiered { .. }));
        assert_eq!(rec.analysis.pattern, WorkloadPattern::ReadHeavy);
    }

    #[test]
    fn test_scan_heavy_recommendation() {
        let stats = Arc::new(WorkloadStats::new());
        let policy = AdaptationPolicy {
            min_ops_before_adapt: 100,
            min_confidence: 0.6,
            ..Default::default()
        };
        let mut adaptor = WorkloadAdaptor::new(Arc::clone(&stats), policy);

        // Simulate scan-heavy workload
        for _ in 0..500 {
            stats.record_scan(100, 10000);
        }
        for _ in 0..500 {
            stats.record_get(1000);
        }

        let recommendation = adaptor.recommend_strategy();
        assert!(recommendation.is_some());

        let rec = recommendation.unwrap();
        assert!(matches!(rec.strategy, CompactionStrategy::Leveled { .. }));
        assert_eq!(rec.analysis.pattern, WorkloadPattern::ScanHeavy);
    }

    #[test]
    fn test_min_ops_threshold() {
        let stats = Arc::new(WorkloadStats::new());
        let policy = AdaptationPolicy {
            min_ops_before_adapt: 1000,
            ..Default::default()
        };
        let mut adaptor = WorkloadAdaptor::new(Arc::clone(&stats), policy);

        // Not enough operations
        for _ in 0..500 {
            stats.record_put(1000);
        }

        assert!(adaptor.recommend_strategy().is_none());

        // Now enough operations
        for _ in 0..600 {
            stats.record_put(1000);
        }

        assert!(adaptor.recommend_strategy().is_some());
    }

    #[test]
    fn test_adaptation_interval() {
        let stats = Arc::new(WorkloadStats::new());
        let policy = AdaptationPolicy {
            min_ops_before_adapt: 100,
            min_confidence: 0.6,
            min_adaptation_interval: Duration::from_secs(60),
            ..Default::default()
        };
        let mut adaptor = WorkloadAdaptor::new(Arc::clone(&stats), policy);

        // First recommendation
        for _ in 0..800 {
            stats.record_put(1000);
        }
        for _ in 0..200 {
            stats.record_get(1000);
        }

        assert!(adaptor.recommend_strategy().is_some());

        // Second recommendation should be blocked by time interval
        stats.reset();
        for _ in 0..800 {
            stats.record_get(1000);
        }
        for _ in 0..200 {
            stats.record_put(1000);
        }

        assert!(adaptor.recommend_strategy().is_none());
    }

    #[test]
    fn test_no_change_same_strategy() {
        let stats = Arc::new(WorkloadStats::new());
        let policy = AdaptationPolicy {
            min_ops_before_adapt: 100,
            min_confidence: 0.6,
            min_adaptation_interval: Duration::from_secs(0),
            ..Default::default()
        };
        let mut adaptor = WorkloadAdaptor::new(Arc::clone(&stats), policy);

        // First recommendation - write heavy
        for _ in 0..800 {
            stats.record_put(1000);
        }
        for _ in 0..200 {
            stats.record_get(1000);
        }

        assert!(adaptor.recommend_strategy().is_some());

        // Still write heavy - no change
        stats.reset();
        for _ in 0..800 {
            stats.record_put(1000);
        }
        for _ in 0..200 {
            stats.record_get(1000);
        }

        assert!(adaptor.recommend_strategy().is_none());
    }

    #[test]
    fn test_reset() {
        let stats = Arc::new(WorkloadStats::new());
        let policy = AdaptationPolicy::default();
        let mut adaptor = WorkloadAdaptor::new(stats, policy);

        adaptor.last_adaptation = Some(Instant::now());
        adaptor.current_strategy = Some(CompactionStrategy::Leveled {
            fanout: 10,
            target_file_count: 10,
        });

        adaptor.reset();

        assert!(adaptor.last_adaptation.is_none());
        assert!(adaptor.current_strategy.is_none());
    }
}
