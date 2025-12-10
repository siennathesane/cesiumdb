//! Workload statistics collection and analysis
//!
//! Tracks database workload patterns to enable adaptive compaction strategy
//! selection.

use std::{
    sync::atomic::{
        AtomicU64,
        AtomicUsize,
        Ordering,
    },
    time::{
        Duration,
        Instant,
    },
};

/// Workload pattern classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadPattern {
    /// Write-heavy workload (>70% writes)
    WriteHeavy,

    /// Read-heavy workload (>70% reads)
    ReadHeavy,

    /// Balanced workload (mixed read/write)
    Balanced,

    /// Scan-heavy workload (large range queries)
    ScanHeavy,

    /// Point lookup workload (mostly Get operations)
    PointLookup,
}

impl WorkloadPattern {
    /// Recommends a compaction strategy for this workload pattern
    pub fn recommended_strategy(&self) -> &'static str {
        match self {
            | Self::WriteHeavy => "Leveled", // Minimize write amplification
            | Self::ReadHeavy => "Tiered",   // Minimize read amplification
            | Self::Balanced => "Hybrid",    // Balance both
            | Self::ScanHeavy => "Leveled",  // Non-overlapping ranges help scans
            | Self::PointLookup => "Tiered", // Bloom filters help point lookups
        }
    }
}

/// Statistics about workload patterns
pub struct WorkloadStats {
    /// Number of Get operations
    gets: AtomicU64,

    /// Number of Put operations
    puts: AtomicU64,

    /// Number of Delete operations
    deletes: AtomicU64,

    /// Number of Scan/Range operations
    scans: AtomicU64,

    /// Number of keys accessed by scans
    scan_keys: AtomicU64,

    /// Total bytes written
    bytes_written: AtomicU64,

    /// Total bytes read
    bytes_read: AtomicU64,

    /// Number of L0 file reads
    l0_reads: AtomicU64,

    /// Number of Ln file reads
    ln_reads: AtomicU64,

    /// Compaction write amplification (total)
    compaction_bytes_written: AtomicU64,
    compaction_bytes_read: AtomicU64,

    /// Number of compactions completed
    compactions_completed: AtomicU64,

    /// Start time for rate calculations
    start_time: Instant,
}

impl WorkloadStats {
    /// Creates a new workload stats tracker
    pub fn new() -> Self {
        Self {
            gets: AtomicU64::new(0),
            puts: AtomicU64::new(0),
            deletes: AtomicU64::new(0),
            scans: AtomicU64::new(0),
            scan_keys: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            l0_reads: AtomicU64::new(0),
            ln_reads: AtomicU64::new(0),
            compaction_bytes_written: AtomicU64::new(0),
            compaction_bytes_read: AtomicU64::new(0),
            compactions_completed: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    /// Records a Get operation
    pub fn record_get(&self, bytes_read: u64) {
        self.gets.fetch_add(1, Ordering::Relaxed);
        self.bytes_read.fetch_add(bytes_read, Ordering::Relaxed);
    }

    /// Records a Put operation
    pub fn record_put(&self, bytes_written: u64) {
        self.puts.fetch_add(1, Ordering::Relaxed);
        self.bytes_written
            .fetch_add(bytes_written, Ordering::Relaxed);
    }

    /// Records a Delete operation
    pub fn record_delete(&self) {
        self.deletes.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a Scan operation
    pub fn record_scan(&self, num_keys: u64, bytes_read: u64) {
        self.scans.fetch_add(1, Ordering::Relaxed);
        self.scan_keys.fetch_add(num_keys, Ordering::Relaxed);
        self.bytes_read.fetch_add(bytes_read, Ordering::Relaxed);
    }

    /// Records an L0 file read
    pub fn record_l0_read(&self) {
        self.l0_reads.fetch_add(1, Ordering::Relaxed);
    }

    /// Records an Ln file read
    pub fn record_ln_read(&self) {
        self.ln_reads.fetch_add(1, Ordering::Relaxed);
    }

    /// Records compaction I/O
    pub fn record_compaction(&self, bytes_read: u64, bytes_written: u64) {
        self.compaction_bytes_read
            .fetch_add(bytes_read, Ordering::Relaxed);
        self.compaction_bytes_written
            .fetch_add(bytes_written, Ordering::Relaxed);
        self.compactions_completed.fetch_add(1, Ordering::Relaxed);
    }

    /// Analyzes the workload and returns the detected pattern
    pub fn analyze(&self) -> WorkloadAnalysis {
        let gets = self.gets.load(Ordering::Relaxed);
        let puts = self.puts.load(Ordering::Relaxed);
        let deletes = self.deletes.load(Ordering::Relaxed);
        let scans = self.scans.load(Ordering::Relaxed);
        let scan_keys = self.scan_keys.load(Ordering::Relaxed);

        let total_ops = gets + puts + deletes + scans;

        if total_ops == 0 {
            return WorkloadAnalysis {
                pattern: WorkloadPattern::Balanced,
                read_ratio: 0.5,
                write_ratio: 0.5,
                scan_ratio: 0.0,
                avg_scan_size: 0.0,
                read_amplification: 0.0,
                write_amplification: 0.0,
                ops_per_second: 0.0,
                confidence: 0.0,
            };
        }

        let read_ops = gets + scans;
        let write_ops = puts + deletes;

        let read_ratio = read_ops as f64 / total_ops as f64;
        let write_ratio = write_ops as f64 / total_ops as f64;
        let scan_ratio = scans as f64 / total_ops as f64;

        let avg_scan_size = if scans > 0 {
            scan_keys as f64 / scans as f64
        } else {
            0.0
        };

        // Calculate amplification factors
        let user_bytes_written = self.bytes_written.load(Ordering::Relaxed);
        let compaction_bytes_written = self.compaction_bytes_written.load(Ordering::Relaxed);

        let write_amplification = if user_bytes_written > 0 {
            (user_bytes_written + compaction_bytes_written) as f64 / user_bytes_written as f64
        } else {
            1.0
        };

        let l0_reads = self.l0_reads.load(Ordering::Relaxed);
        let ln_reads = self.ln_reads.load(Ordering::Relaxed);
        let read_amplification = if gets > 0 {
            (l0_reads + ln_reads) as f64 / gets as f64
        } else {
            1.0
        };

        // Calculate ops/sec
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let ops_per_second = if elapsed > 0.0 {
            total_ops as f64 / elapsed
        } else {
            0.0
        };

        // Determine pattern
        let pattern = if scan_ratio > 0.3 {
            WorkloadPattern::ScanHeavy
        } else if scans > 0 && avg_scan_size < 10.0 && read_ratio > 0.7 {
            WorkloadPattern::PointLookup
        } else if write_ratio > 0.7 {
            WorkloadPattern::WriteHeavy
        } else if read_ratio > 0.7 {
            WorkloadPattern::ReadHeavy
        } else {
            WorkloadPattern::Balanced
        };

        // Confidence based on sample size
        let confidence = if total_ops < 100 {
            0.3
        } else if total_ops < 1000 {
            0.6
        } else if total_ops < 10000 {
            0.8
        } else {
            0.95
        };

        WorkloadAnalysis {
            pattern,
            read_ratio,
            write_ratio,
            scan_ratio,
            avg_scan_size,
            read_amplification,
            write_amplification,
            ops_per_second,
            confidence,
        }
    }

    /// Returns a snapshot of current statistics
    pub fn snapshot(&self) -> WorkloadSnapshot {
        WorkloadSnapshot {
            gets: self.gets.load(Ordering::Relaxed),
            puts: self.puts.load(Ordering::Relaxed),
            deletes: self.deletes.load(Ordering::Relaxed),
            scans: self.scans.load(Ordering::Relaxed),
            scan_keys: self.scan_keys.load(Ordering::Relaxed),
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            l0_reads: self.l0_reads.load(Ordering::Relaxed),
            ln_reads: self.ln_reads.load(Ordering::Relaxed),
            compaction_bytes_written: self.compaction_bytes_written.load(Ordering::Relaxed),
            compaction_bytes_read: self.compaction_bytes_read.load(Ordering::Relaxed),
            compactions_completed: self.compactions_completed.load(Ordering::Relaxed),
            uptime: self.start_time.elapsed(),
        }
    }

    /// Resets all statistics
    pub fn reset(&self) {
        self.gets.store(0, Ordering::Relaxed);
        self.puts.store(0, Ordering::Relaxed);
        self.deletes.store(0, Ordering::Relaxed);
        self.scans.store(0, Ordering::Relaxed);
        self.scan_keys.store(0, Ordering::Relaxed);
        self.bytes_written.store(0, Ordering::Relaxed);
        self.bytes_read.store(0, Ordering::Relaxed);
        self.l0_reads.store(0, Ordering::Relaxed);
        self.ln_reads.store(0, Ordering::Relaxed);
        self.compaction_bytes_written.store(0, Ordering::Relaxed);
        self.compaction_bytes_read.store(0, Ordering::Relaxed);
        self.compactions_completed.store(0, Ordering::Relaxed);
    }
}

impl Default for WorkloadStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Analysis results from workload statistics
#[derive(Debug, Clone)]
pub struct WorkloadAnalysis {
    /// Detected workload pattern
    pub pattern: WorkloadPattern,

    /// Ratio of read operations (0.0-1.0)
    pub read_ratio: f64,

    /// Ratio of write operations (0.0-1.0)
    pub write_ratio: f64,

    /// Ratio of scan operations (0.0-1.0)
    pub scan_ratio: f64,

    /// Average number of keys per scan
    pub avg_scan_size: f64,

    /// Read amplification factor
    pub read_amplification: f64,

    /// Write amplification factor
    pub write_amplification: f64,

    /// Operations per second
    pub ops_per_second: f64,

    /// Confidence in the analysis (0.0-1.0)
    pub confidence: f64,
}

impl std::fmt::Display for WorkloadAnalysis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Pattern: {:?} ({:.0}% confidence), R/W: {:.1}%/{:.1}%, Scans: {:.1}%, RA: {:.2}x, WA: {:.2}x, OPS: {:.0}/s",
            self.pattern,
            self.confidence * 100.0,
            self.read_ratio * 100.0,
            self.write_ratio * 100.0,
            self.scan_ratio * 100.0,
            self.read_amplification,
            self.write_amplification,
            self.ops_per_second
        )
    }
}

/// Snapshot of workload statistics at a point in time
#[derive(Debug, Clone, Copy)]
pub struct WorkloadSnapshot {
    pub gets: u64,
    pub puts: u64,
    pub deletes: u64,
    pub scans: u64,
    pub scan_keys: u64,
    pub bytes_written: u64,
    pub bytes_read: u64,
    pub l0_reads: u64,
    pub ln_reads: u64,
    pub compaction_bytes_written: u64,
    pub compaction_bytes_read: u64,
    pub compactions_completed: u64,
    pub uptime: Duration,
}

impl std::fmt::Display for WorkloadSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total_ops = self.gets + self.puts + self.deletes + self.scans;
        write!(
            f,
            "Ops: {} (G:{} P:{} D:{} S:{}), Bytes: R:{:.2}MB W:{:.2}MB, Compaction: R:{:.2}MB W:{:.2}MB, Uptime: {:.1}s",
            total_ops,
            self.gets,
            self.puts,
            self.deletes,
            self.scans,
            self.bytes_read as f64 / (1024.0 * 1024.0),
            self.bytes_written as f64 / (1024.0 * 1024.0),
            self.compaction_bytes_read as f64 / (1024.0 * 1024.0),
            self.compaction_bytes_written as f64 / (1024.0 * 1024.0),
            self.uptime.as_secs_f64()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workload_stats_creation() {
        let stats = WorkloadStats::new();
        let snapshot = stats.snapshot();

        assert_eq!(snapshot.gets, 0);
        assert_eq!(snapshot.puts, 0);
        assert_eq!(snapshot.deletes, 0);
    }

    #[test]
    fn test_record_operations() {
        let stats = WorkloadStats::new();

        stats.record_get(100);
        stats.record_put(200);
        stats.record_delete();
        stats.record_scan(10, 500);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.gets, 1);
        assert_eq!(snapshot.puts, 1);
        assert_eq!(snapshot.deletes, 1);
        assert_eq!(snapshot.scans, 1);
        assert_eq!(snapshot.scan_keys, 10);
        assert_eq!(snapshot.bytes_read, 600);
        assert_eq!(snapshot.bytes_written, 200);
    }

    #[test]
    fn test_write_heavy_pattern() {
        let stats = WorkloadStats::new();

        // 80% writes
        for _ in 0..80 {
            stats.record_put(1000);
        }
        for _ in 0..20 {
            stats.record_get(1000);
        }

        let analysis = stats.analyze();
        assert_eq!(analysis.pattern, WorkloadPattern::WriteHeavy);
        assert!(analysis.write_ratio > 0.7);
    }

    #[test]
    fn test_read_heavy_pattern() {
        let stats = WorkloadStats::new();

        // 80% reads
        for _ in 0..80 {
            stats.record_get(1000);
        }
        for _ in 0..20 {
            stats.record_put(1000);
        }

        let analysis = stats.analyze();
        assert_eq!(analysis.pattern, WorkloadPattern::ReadHeavy);
        assert!(analysis.read_ratio > 0.7);
    }

    #[test]
    fn test_scan_heavy_pattern() {
        let stats = WorkloadStats::new();

        // Many scans
        for _ in 0..50 {
            stats.record_scan(100, 10000);
        }
        for _ in 0..50 {
            stats.record_get(1000);
        }

        let analysis = stats.analyze();
        assert_eq!(analysis.pattern, WorkloadPattern::ScanHeavy);
        assert!(analysis.scan_ratio > 0.3);
    }

    #[test]
    fn test_balanced_pattern() {
        let stats = WorkloadStats::new();

        // Balanced
        for _ in 0..50 {
            stats.record_get(1000);
        }
        for _ in 0..50 {
            stats.record_put(1000);
        }

        let analysis = stats.analyze();
        assert_eq!(analysis.pattern, WorkloadPattern::Balanced);
        assert!(analysis.read_ratio > 0.4 && analysis.read_ratio < 0.6);
    }

    #[test]
    fn test_write_amplification() {
        let stats = WorkloadStats::new();

        stats.record_put(1000);
        stats.record_compaction(1000, 2000); // 2x write amp

        let analysis = stats.analyze();
        assert!((analysis.write_amplification - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_read_amplification() {
        let stats = WorkloadStats::new();

        for _ in 0..10 {
            stats.record_get(1000);
            stats.record_l0_read();
            stats.record_ln_read();
        }

        let analysis = stats.analyze();
        assert!((analysis.read_amplification - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_confidence_levels() {
        let stats = WorkloadStats::new();

        // Low confidence with few ops
        for _ in 0..50 {
            stats.record_get(100);
        }
        let analysis = stats.analyze();
        assert!(analysis.confidence < 0.5);

        // Higher confidence with more ops
        for _ in 0..1000 {
            stats.record_get(100);
        }
        let analysis = stats.analyze();
        assert!(analysis.confidence > 0.7);
    }

    #[test]
    fn test_reset() {
        let stats = WorkloadStats::new();

        stats.record_get(100);
        stats.record_put(200);

        stats.reset();

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.gets, 0);
        assert_eq!(snapshot.puts, 0);
    }

    #[test]
    fn test_pattern_recommendations() {
        assert_eq!(
            WorkloadPattern::WriteHeavy.recommended_strategy(),
            "Leveled"
        );
        assert_eq!(WorkloadPattern::ReadHeavy.recommended_strategy(), "Tiered");
        assert_eq!(WorkloadPattern::Balanced.recommended_strategy(), "Hybrid");
        assert_eq!(WorkloadPattern::ScanHeavy.recommended_strategy(), "Leveled");
        assert_eq!(
            WorkloadPattern::PointLookup.recommended_strategy(),
            "Tiered"
        );
    }
}
