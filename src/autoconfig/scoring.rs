//! Scoring function for autoconfigurator benchmark results.

use super::workload::BenchmarkMetrics;

/// Weighted composite score for a configuration.
///
/// Higher is better.  Incorporates write throughput, read throughput,
/// mixed throughput, and penalizes high write amplification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    pub raw: f64,
    pub write_throughput: f64,
    pub read_throughput: f64,
    pub mixed_throughput: f64,
    pub write_amp_penalty: f64,
}

impl Score {
    /// Returns true if this score is meaningfully better than another.
    pub fn is_better_than(&self, other: Score) -> bool {
        self.raw > other.raw * 1.05
    }
}

/// Computes a composite score from three benchmark runs.
pub fn compute_score(
    write: &BenchmarkMetrics,
    read: &BenchmarkMetrics,
    mixed: &BenchmarkMetrics,
) -> Score {
    let write_throughput = write.ops_per_sec;
    let read_throughput = read.ops_per_sec;
    let mixed_throughput = mixed.ops_per_sec;

    // Penalize write amplification above 2.0x
    let write_amp_penalty = {
        let amp = write.write_amplification.max(mixed.write_amplification);
        if amp > 2.0 {
            (amp - 2.0) * 100_000.0
        } else {
            0.0
        }
    };

    let raw = write_throughput * 0.4
        + read_throughput * 0.3
        + mixed_throughput * 0.2
        - write_amp_penalty * 0.1;

    Score {
        raw: raw.max(0.0),
        write_throughput,
        read_throughput,
        mixed_throughput,
        write_amp_penalty,
    }
}
