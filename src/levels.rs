//! LSM-tree level management
//!
//! This module implements the leveling structure for the LSM-tree, including:
//! - Immutable version sets (snapshots of the current state)
//! - Per-level compaction strategies (tiered, leveled, universal)
//! - Level metadata and statistics
//! - Key range tracking for efficient lookups

use std::{
    cmp::Ordering,
    sync::Arc,
};

use crate::segment::Segment;

/// Compaction strategy for a level
///
/// Different levels can use different strategies to optimize for
/// different workload patterns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompactionStrategy {
    /// Tiered compaction (Cassandra-style)
    ///
    /// Best for write-heavy workloads. Lower write amplification but
    /// higher read and space amplification.
    ///
    /// Files at the same level can have overlapping key ranges.
    /// Compaction merges multiple sorted runs into a single run.
    Tiered {
        /// Size ratio between adjacent tiers (typically 4-10)
        size_ratio: f64,
        /// Minimum number of files to trigger compaction
        min_merge_width: usize,
        /// Maximum number of files to merge at once
        max_merge_width: usize,
    },

    /// Leveled compaction (RocksDB/LevelDB-style)
    ///
    /// Best for read-heavy workloads. Better read performance and
    /// space amplification, but higher write amplification.
    ///
    /// Files within a level have non-overlapping key ranges.
    /// Compaction merges one file from Ln with overlapping files in Ln+1.
    Leveled {
        /// Size multiplier between levels (typically 10)
        fanout: u32,
        /// Target number of files per level
        target_file_count: usize,
    },

    /// Universal compaction (size-tiered)
    ///
    /// Simplified strategy for smaller datasets or specific use cases.
    /// Merges files of similar size.
    Universal {
        /// Maximum allowed space amplification (typically 1.5-2.0)
        max_size_amplification: f64,
        /// Size ratio for merging (typically 1.0)
        size_ratio: f64,
    },
}

impl CompactionStrategy {
    /// Returns the default strategy for L0 (tiered)
    pub fn default_l0() -> Self {
        Self::Tiered {
            size_ratio: 4.0,
            min_merge_width: 4,
            max_merge_width: 10,
        }
    }

    /// Returns the default strategy for deep levels (leveled)
    pub fn default_leveled() -> Self {
        Self::Leveled {
            fanout: 10,
            target_file_count: 10,
        }
    }

    /// Returns whether this strategy allows overlapping files within a level
    pub fn allows_overlaps(&self) -> bool {
        matches!(self, Self::Tiered { .. } | Self::Universal { .. })
    }
}

/// A key range within a level
///
/// Used for fast lookups to determine which segments might contain a key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyRange {
    /// Smallest key in the range (inclusive)
    pub start: Vec<u8>,
    /// Largest key in the range (inclusive)
    pub end: Vec<u8>,
    /// Segment ID this range belongs to
    pub segment_id: u64,
}

impl KeyRange {
    /// Creates a new key range
    pub fn new(start: Vec<u8>, end: Vec<u8>, segment_id: u64) -> Self {
        debug_assert!(start <= end, "start must be <= end");
        Self {
            start,
            end,
            segment_id,
        }
    }

    /// Checks if this range contains the given key
    pub fn contains(&self, key: &[u8]) -> bool {
        key >= self.start.as_slice() && key <= self.end.as_slice()
    }

    /// Checks if this range overlaps with another range
    pub fn overlaps(&self, other: &KeyRange) -> bool {
        self.start.as_slice() <= other.end.as_slice() &&
            other.start.as_slice() <= self.end.as_slice()
    }

    /// Checks if this range is strictly before another range
    pub fn is_before(&self, other: &KeyRange) -> bool {
        self.end.as_slice() < other.start.as_slice()
    }

    /// Checks if this range is strictly after another range
    pub fn is_after(&self, other: &KeyRange) -> bool {
        self.start.as_slice() > other.end.as_slice()
    }
}

impl PartialOrd for KeyRange {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for KeyRange {
    fn cmp(&self, other: &Self) -> Ordering {
        // Order by start key, then by end key
        self.start
            .cmp(&other.start)
            .then_with(|| self.end.cmp(&other.end))
    }
}

/// Statistics for a level
#[derive(Debug, Clone, Default)]
pub struct LevelStats {
    /// Total size of all segments in this level (bytes)
    pub total_size: u64,
    /// Number of segments in this level
    pub num_segments: usize,
    /// Number of read operations on this level
    pub num_reads: u64,
    /// Number of bytes read from this level
    pub bytes_read: u64,
    /// Number of compactions performed on this level
    pub num_compactions: u64,
    /// Total bytes written during compaction
    pub bytes_written: u64,
}

impl LevelStats {
    /// Calculates the score for this level (higher = more urgent to compact)
    ///
    /// Score > 1.0 means the level should be compacted
    pub fn score(&self, max_size: u64, strategy: &CompactionStrategy) -> f64 {
        match strategy {
            | CompactionStrategy::Tiered { size_ratio, .. } => {
                // Score based on number of files and size ratio
                let size_score = self.total_size as f64 / max_size as f64;
                let file_score = self.num_segments as f64 / 10.0;
                size_score.max(file_score)
            },
            | CompactionStrategy::Leveled { fanout, .. } => {
                // Simple size-based scoring
                self.total_size as f64 / max_size as f64
            },
            | CompactionStrategy::Universal {
                max_size_amplification,
                ..
            } => {
                // Score based on space amplification
                let space_amp = self.total_size as f64 / max_size as f64;
                space_amp / max_size_amplification
            },
        }
    }
}

/// A single level in the LSM-tree
///
/// Each level contains multiple segments (SSTables) and maintains
/// metadata about key ranges and statistics.
#[derive(Clone)]
pub struct Level {
    /// Level number (0 = L0, 1 = L1, etc.)
    pub level_num: u8,

    /// Segments in this level (reference-counted for safe concurrent access)
    pub segments: Vec<Arc<Segment>>,

    /// Key ranges for fast lookups
    ///
    /// For leveled compaction, ranges are non-overlapping and sorted.
    /// For tiered/universal, ranges may overlap.
    pub key_ranges: Vec<KeyRange>,

    /// Compaction strategy for this level
    pub strategy: CompactionStrategy,

    /// Maximum size for this level (bytes)
    pub max_size: u64,

    /// Target size for each file in this level (bytes)
    pub target_file_size: u64,

    /// Statistics for this level
    pub stats: LevelStats,
}

impl Level {
    /// Creates a new empty level
    pub fn new(
        level_num: u8,
        strategy: CompactionStrategy,
        max_size: u64,
        target_file_size: u64,
    ) -> Self {
        Self {
            level_num,
            segments: Vec::new(),
            key_ranges: Vec::new(),
            strategy,
            max_size,
            target_file_size,
            stats: LevelStats::default(),
        }
    }

    /// Returns the total size of all segments in this level
    pub fn total_size(&self) -> u64 {
        self.stats.total_size
    }

    /// Adds a segment to this level
    ///
    /// Updates key ranges and statistics.
    pub fn add_segment(&mut self, segment: Arc<Segment>, key_range: KeyRange) {
        self.stats.total_size += segment.size_in_bytes();
        self.stats.num_segments += 1;

        self.segments.push(segment);
        self.key_ranges.push(key_range);

        // Sort key ranges if this is a leveled strategy
        if !self.strategy.allows_overlaps() {
            self.key_ranges.sort();
        }
    }

    /// Removes a segment from this level by ID
    ///
    /// Returns the removed segment if found.
    pub fn remove_segment(&mut self, segment_id: u64) -> Option<Arc<Segment>> {
        if let Some(idx) = self.segments.iter().position(|s| s.id() == segment_id) {
            let segment = self.segments.remove(idx);
            self.key_ranges.remove(idx);

            self.stats.total_size -= segment.size_in_bytes();
            self.stats.num_segments -= 1;

            Some(segment)
        } else {
            None
        }
    }

    /// Finds all segments that may contain the given key
    ///
    /// Returns segment IDs that overlap with the key.
    pub fn find_segments_for_key(&self, key: &[u8]) -> Vec<u64> {
        self.key_ranges
            .iter()
            .filter(|range| range.contains(key))
            .map(|range| range.segment_id)
            .collect()
    }

    /// Finds all segments that overlap with the given key range
    pub fn find_overlapping_segments(&self, start: &[u8], end: &[u8]) -> Vec<u64> {
        let query_range = KeyRange::new(start.to_vec(), end.to_vec(), 0);

        self.key_ranges
            .iter()
            .filter(|range| range.overlaps(&query_range))
            .map(|range| range.segment_id)
            .collect()
    }

    /// Calculates the compaction score for this level
    pub fn score(&self) -> f64 {
        self.stats.score(self.max_size, &self.strategy)
    }
}

/// An immutable snapshot of the LSM-tree state
///
/// Version sets are immutable and reference-counted, allowing safe
/// concurrent access without locks. When the state changes (e.g., after
/// compaction), a new version set is created and atomically installed.
#[derive(Clone)]
pub struct VersionSet {
    /// Monotonically increasing sequence number
    ///
    /// Used for ordering versions and implementing snapshot isolation.
    pub sequence: u64,

    /// L0 segments (may have overlapping key ranges)
    ///
    /// L0 is special because it contains recently flushed memtables.
    /// Segments in L0 can have overlapping ranges and are always searched
    /// in reverse chronological order (newest first).
    pub l0: Vec<Arc<Segment>>,

    /// L1+ levels (key ranges depend on strategy)
    ///
    /// The number of levels grows dynamically based on data size.
    /// Each level uses a compaction strategy that may differ from other levels.
    pub levels: Vec<Level>,

    /// Total number of segments across all levels
    pub total_segments: usize,

    /// Total size across all levels (bytes)
    pub total_size: u64,
}

impl VersionSet {
    /// Creates a new empty version set
    pub fn new(sequence: u64, num_levels: usize) -> Self {
        let mut levels = Vec::with_capacity(num_levels);

        // L1-L2: Transition zone, can be tiered or leveled depending on workload
        for level_num in 1..=2 {
            levels.push(Level::new(
                level_num,
                CompactionStrategy::default_l0(),
                Self::max_size_for_level(level_num),
                Self::target_file_size_for_level(level_num),
            ));
        }

        // L3+: Leveled compaction for read optimization
        for level_num in 3..=num_levels as u8 {
            levels.push(Level::new(
                level_num,
                CompactionStrategy::default_leveled(),
                Self::max_size_for_level(level_num),
                Self::target_file_size_for_level(level_num),
            ));
        }

        Self {
            sequence,
            l0: Vec::new(),
            levels,
            total_segments: 0,
            total_size: 0,
        }
    }

    /// Calculates maximum size for a level based on level number
    ///
    /// Uses exponential growth: L1 = 64MB, L2 = 640MB, L3 = 6.4GB, etc.
    fn max_size_for_level(level_num: u8) -> u64 {
        const BASE_SIZE: u64 = 64 * 1024 * 1024; // 64 MB
        const FANOUT: u64 = 10;

        BASE_SIZE * FANOUT.pow(level_num as u32 - 1)
    }

    /// Calculates target file size for a level based on level number
    ///
    /// Uses exponential growth: L1 = 64MB, L2 = 64MB, L3+ = 128MB, etc.
    fn target_file_size_for_level(level_num: u8) -> u64 {
        const BASE_FILE_SIZE: u64 = 64 * 1024 * 1024; // 64 MB

        if level_num <= 2 {
            BASE_FILE_SIZE
        } else {
            BASE_FILE_SIZE * 2
        }
    }

    /// Adds a segment to L0
    pub fn add_to_l0(&mut self, segment: Arc<Segment>) {
        self.total_size += segment.size_in_bytes();
        self.total_segments += 1;
        self.l0.push(segment);
    }

    /// Finds the level with the highest compaction score
    ///
    /// Returns (level_num, score) or None if no level needs compaction
    pub fn pick_compaction_level(&self) -> Option<(u8, f64)> {
        // Check L0 file count (special case)
        const L0_COMPACTION_TRIGGER: usize = 4;
        if self.l0.len() >= L0_COMPACTION_TRIGGER {
            let score = self.l0.len() as f64 / L0_COMPACTION_TRIGGER as f64;
            return Some((0, score));
        }

        // Check L1+ by size
        self.levels
            .iter()
            .map(|level| (level.level_num, level.score()))
            .filter(|(_, score)| *score > 1.0)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }

    /// Creates a new version set with an incremented sequence number
    pub fn next_version(&self) -> Self {
        Self {
            sequence: self.sequence + 1,
            ..self.clone()
        }
    }

    /// Returns the number of levels in this version
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_range_contains() {
        let range = KeyRange::new(b"apple".to_vec(), b"banana".to_vec(), 1);

        assert!(range.contains(b"apple"));
        assert!(range.contains(b"banana"));
        assert!(range.contains(b"avocado"));
        assert!(!range.contains(b"aardvark"));
        assert!(!range.contains(b"cherry"));
    }

    #[test]
    fn test_key_range_overlaps() {
        let r1 = KeyRange::new(b"apple".to_vec(), b"banana".to_vec(), 1);
        let r2 = KeyRange::new(b"avocado".to_vec(), b"cherry".to_vec(), 2);
        let r3 = KeyRange::new(b"date".to_vec(), b"elderberry".to_vec(), 3);
        let r4 = KeyRange::new(b"blueberry".to_vec(), b"coconut".to_vec(), 4);

        assert!(r1.overlaps(&r2)); // [apple-banana] overlaps [avocado-cherry]
        assert!(r2.overlaps(&r1)); // symmetric
        assert!(!r1.overlaps(&r3)); // [apple-banana] doesn't overlap [date-elderberry]
        assert!(!r2.overlaps(&r3)); // [avocado-cherry] doesn't overlap [date-elderberry]
        assert!(r2.overlaps(&r4)); // [avocado-cherry] overlaps [blueberry-coconut]
        assert!(r4.overlaps(&r2)); // symmetric
    }

    #[test]
    fn test_key_range_is_before() {
        let r1 = KeyRange::new(b"apple".to_vec(), b"banana".to_vec(), 1);
        let r2 = KeyRange::new(b"cherry".to_vec(), b"date".to_vec(), 2);
        let r3 = KeyRange::new(b"avocado".to_vec(), b"blueberry".to_vec(), 3);

        // r1 [apple-banana] is before r2 [cherry-date]
        assert!(r1.is_before(&r2));
        // r2 is NOT before r1
        assert!(!r2.is_before(&r1));
        // r1 [apple-banana] is NOT before r3 [avocado-blueberry] (they overlap)
        assert!(!r1.is_before(&r3));
        // r3 is NOT before r2 (r3 ends at blueberry, r2 starts at cherry, blueberry < cherry)
        assert!(r3.is_before(&r2));
    }

    #[test]
    fn test_key_range_is_after() {
        let r1 = KeyRange::new(b"apple".to_vec(), b"banana".to_vec(), 1);
        let r2 = KeyRange::new(b"cherry".to_vec(), b"date".to_vec(), 2);
        let r3 = KeyRange::new(b"avocado".to_vec(), b"blueberry".to_vec(), 3);

        // r2 [cherry-date] is after r1 [apple-banana]
        assert!(r2.is_after(&r1));
        // r1 is NOT after r2
        assert!(!r1.is_after(&r2));
        // r3 [avocado-blueberry] is NOT after r1 [apple-banana] (they overlap)
        assert!(!r3.is_after(&r1));
        // r2 is after r3
        assert!(r2.is_after(&r3));
    }

    #[test]
    fn test_key_range_is_before_and_after_edge_cases() {
        // Test with adjacent ranges (touching but not overlapping)
        let r1 = KeyRange::new(b"a".to_vec(), b"b".to_vec(), 1);
        let r2 = KeyRange::new(b"c".to_vec(), b"d".to_vec(), 2);

        assert!(r1.is_before(&r2));
        assert!(r2.is_after(&r1));
        assert!(!r2.is_before(&r1));
        assert!(!r1.is_after(&r2));

        // Test with same range
        let r3 = KeyRange::new(b"a".to_vec(), b"b".to_vec(), 3);
        assert!(!r1.is_before(&r3));
        assert!(!r1.is_after(&r3));
    }

    #[test]
    fn test_key_range_ordering() {
        let r1 = KeyRange::new(b"apple".to_vec(), b"banana".to_vec(), 1);
        let r2 = KeyRange::new(b"cherry".to_vec(), b"date".to_vec(), 2);
        let r3 = KeyRange::new(b"apple".to_vec(), b"avocado".to_vec(), 3);
        let r4 = KeyRange::new(b"apple".to_vec(), b"banana".to_vec(), 4);

        // r1 < r2 (apple < cherry)
        assert!(r1 < r2);
        assert!(r2 > r1);

        // r3 < r1 (same start, but avocado < banana)
        assert!(r3 < r1);
        assert!(r1 > r3);

        // r1 == r4 (same start and end, different segment_id doesn't affect ordering)
        assert_eq!(r1.cmp(&r4), Ordering::Equal);
    }

    #[test]
    fn test_key_range_partial_ord() {
        let r1 = KeyRange::new(b"a".to_vec(), b"b".to_vec(), 1);
        let r2 = KeyRange::new(b"c".to_vec(), b"d".to_vec(), 2);

        // Test partial_cmp
        assert_eq!(r1.partial_cmp(&r2), Some(Ordering::Less));
        assert_eq!(r2.partial_cmp(&r1), Some(Ordering::Greater));
        assert_eq!(r1.partial_cmp(&r1), Some(Ordering::Equal));
    }

    #[test]
    fn test_key_range_new_edge_cases() {
        // Empty range (start == end)
        let r = KeyRange::new(b"same".to_vec(), b"same".to_vec(), 1);
        assert!(r.contains(b"same"));
        assert!(!r.contains(b"other"));

        // Single byte keys
        let r2 = KeyRange::new(vec![0x00], vec![0xFF], 2);
        assert!(r2.contains(&[0x00]));
        assert!(r2.contains(&[0x7F]));
        assert!(r2.contains(&[0xFF]));

        // Empty keys
        let r3 = KeyRange::new(vec![], vec![], 3);
        assert!(r3.contains(&[]));
    }

    #[test]
    fn test_level_add_remove() {
        let level = Level::new(
            1,
            CompactionStrategy::default_leveled(),
            1024 * 1024,
            64 * 1024,
        );

        // Note: We can't easily test with real segments without full setup,
        // so this is a simplified test of the structure
        assert_eq!(level.stats.num_segments, 0);
        assert_eq!(level.total_size(), 0);
    }

    #[test]
    fn test_level_find_segments_for_key_empty() {
        let level = Level::new(
            1,
            CompactionStrategy::default_leveled(),
            1024 * 1024,
            64 * 1024,
        );

        // Finding segments in empty level should return empty vec
        let segments = level.find_segments_for_key(b"test");
        assert!(segments.is_empty());
    }

    #[test]
    fn test_level_find_overlapping_segments_empty() {
        let level = Level::new(
            1,
            CompactionStrategy::default_leveled(),
            1024 * 1024,
            64 * 1024,
        );

        // Finding overlapping segments in empty level should return empty vec
        let segments = level.find_overlapping_segments(b"a", b"z");
        assert!(segments.is_empty());
    }

    #[test]
    fn test_level_score_empty() {
        let level = Level::new(
            1,
            CompactionStrategy::default_leveled(),
            1024 * 1024,
            64 * 1024,
        );

        // Empty level should have score 0
        assert_eq!(level.score(), 0.0);
    }

    #[test]
    fn test_version_set_creation() {
        let version = VersionSet::new(1, 7);

        assert_eq!(version.sequence, 1);
        assert_eq!(version.num_levels(), 7);
        assert_eq!(version.l0.len(), 0);
        assert_eq!(version.total_segments, 0);

        // Check that strategies are set correctly
        assert_eq!(version.levels[0].strategy, CompactionStrategy::default_l0()); // L1
        assert_eq!(version.levels[1].strategy, CompactionStrategy::default_l0()); // L2
        assert_eq!(
            version.levels[2].strategy,
            CompactionStrategy::default_leveled()
        ); // L3
    }

    #[test]
    fn test_compaction_strategy_allows_overlaps() {
        assert!(CompactionStrategy::default_l0().allows_overlaps());
        assert!(!CompactionStrategy::default_leveled().allows_overlaps());
    }

    #[test]
    fn test_compaction_strategy_universal_allows_overlaps() {
        let universal = CompactionStrategy::Universal {
            max_size_amplification: 2.0,
            size_ratio: 1.0,
        };
        assert!(universal.allows_overlaps());
    }

    #[test]
    fn test_level_max_sizes() {
        let v = VersionSet::new(1, 5);

        // L1 = 64MB, L2 = 640MB, L3 = 6.4GB, etc.
        assert_eq!(v.levels[0].max_size, 64 * 1024 * 1024);
        assert_eq!(v.levels[1].max_size, 640 * 1024 * 1024);
        assert_eq!(v.levels[2].max_size, 6400 * 1024 * 1024);
    }

    #[test]
    fn test_level_target_file_sizes() {
        let v = VersionSet::new(1, 5);

        // L1-L2 = 64MB, L3+ = 128MB
        assert_eq!(v.levels[0].target_file_size, 64 * 1024 * 1024); // L1
        assert_eq!(v.levels[1].target_file_size, 64 * 1024 * 1024); // L2
        assert_eq!(v.levels[2].target_file_size, 128 * 1024 * 1024); // L3
        assert_eq!(v.levels[3].target_file_size, 128 * 1024 * 1024); // L4
    }

    #[test]
    fn test_version_set_next_version() {
        let v1 = VersionSet::new(1, 5);
        let v2 = v1.next_version();

        assert_eq!(v2.sequence, 2);
        assert_eq!(v2.num_levels(), v1.num_levels());
        assert_eq!(v2.l0.len(), v1.l0.len());
        assert_eq!(v2.total_segments, v1.total_segments);
        assert_eq!(v2.total_size, v1.total_size);

        // Next version again
        let v3 = v2.next_version();
        assert_eq!(v3.sequence, 3);
    }

    #[test]
    fn test_version_set_pick_compaction_level_empty() {
        let v = VersionSet::new(1, 5);

        // Empty version set should not need compaction
        let result = v.pick_compaction_level();
        assert!(result.is_none());
    }

    #[test]
    fn test_level_stats_default() {
        let stats = LevelStats::default();

        assert_eq!(stats.total_size, 0);
        assert_eq!(stats.num_segments, 0);
        assert_eq!(stats.num_reads, 0);
        assert_eq!(stats.bytes_read, 0);
        assert_eq!(stats.num_compactions, 0);
        assert_eq!(stats.bytes_written, 0);
    }

    #[test]
    fn test_level_stats_score_tiered() {
        let mut stats = LevelStats::default();
        stats.total_size = 100;
        stats.num_segments = 5;

        let strategy = CompactionStrategy::Tiered {
            size_ratio: 4.0,
            min_merge_width: 4,
            max_merge_width: 10,
        };

        // Score should be max of size_score and file_score
        // size_score = 100 / 1000 = 0.1
        // file_score = 5 / 10 = 0.5
        // max = 0.5
        let score = stats.score(1000, &strategy);
        assert!((score - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_level_stats_score_tiered_size_dominated() {
        let mut stats = LevelStats::default();
        stats.total_size = 800;
        stats.num_segments = 2;

        let strategy = CompactionStrategy::Tiered {
            size_ratio: 4.0,
            min_merge_width: 4,
            max_merge_width: 10,
        };

        // size_score = 800 / 1000 = 0.8
        // file_score = 2 / 10 = 0.2
        // max = 0.8
        let score = stats.score(1000, &strategy);
        assert!((score - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_level_stats_score_leveled() {
        let mut stats = LevelStats::default();
        stats.total_size = 500;
        stats.num_segments = 5;

        let strategy = CompactionStrategy::Leveled {
            fanout: 10,
            target_file_count: 10,
        };

        // Score = total_size / max_size = 500 / 1000 = 0.5
        let score = stats.score(1000, &strategy);
        assert!((score - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_level_stats_score_universal() {
        let mut stats = LevelStats::default();
        stats.total_size = 1500;

        let strategy = CompactionStrategy::Universal {
            max_size_amplification: 2.0,
            size_ratio: 1.0,
        };

        // space_amp = 1500 / 1000 = 1.5
        // score = space_amp / max_size_amplification = 1.5 / 2.0 = 0.75
        let score = stats.score(1000, &strategy);
        assert!((score - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_compaction_strategy_default_l0_values() {
        let strategy = CompactionStrategy::default_l0();

        match strategy {
            CompactionStrategy::Tiered {
                size_ratio,
                min_merge_width,
                max_merge_width,
            } => {
                assert!((size_ratio - 4.0).abs() < 0.001);
                assert_eq!(min_merge_width, 4);
                assert_eq!(max_merge_width, 10);
            }
            _ => panic!("Expected Tiered strategy"),
        }
    }

    #[test]
    fn test_compaction_strategy_default_leveled_values() {
        let strategy = CompactionStrategy::default_leveled();

        match strategy {
            CompactionStrategy::Leveled {
                fanout,
                target_file_count,
            } => {
                assert_eq!(fanout, 10);
                assert_eq!(target_file_count, 10);
            }
            _ => panic!("Expected Leveled strategy"),
        }
    }

    #[test]
    fn test_level_new() {
        let level = Level::new(
            3,
            CompactionStrategy::default_leveled(),
            1024 * 1024 * 1024,
            64 * 1024 * 1024,
        );

        assert_eq!(level.level_num, 3);
        assert!(level.segments.is_empty());
        assert!(level.key_ranges.is_empty());
        assert_eq!(level.max_size, 1024 * 1024 * 1024);
        assert_eq!(level.target_file_size, 64 * 1024 * 1024);
        assert_eq!(level.stats.num_segments, 0);
        assert_eq!(level.stats.total_size, 0);
    }

    #[test]
    fn test_version_set_multiple_levels() {
        // Test with different number of levels
        for num_levels in 3..=10 {
            let v = VersionSet::new(1, num_levels);
            assert_eq!(v.num_levels(), num_levels);

            // L1-L2 should be tiered
            if num_levels >= 1 {
                assert!(v.levels[0].strategy.allows_overlaps());
            }
            if num_levels >= 2 {
                assert!(v.levels[1].strategy.allows_overlaps());
            }

            // L3+ should be leveled (no overlaps)
            for i in 2..num_levels {
                assert!(
                    !v.levels[i].strategy.allows_overlaps(),
                    "Level {} should use leveled strategy",
                    i + 1
                );
            }
        }
    }

    #[test]
    fn test_key_range_clone() {
        let r1 = KeyRange::new(b"apple".to_vec(), b"banana".to_vec(), 42);
        let r2 = r1.clone();

        assert_eq!(r1.start, r2.start);
        assert_eq!(r1.end, r2.end);
        assert_eq!(r1.segment_id, r2.segment_id);
    }

    #[test]
    fn test_key_range_debug() {
        let r = KeyRange::new(b"a".to_vec(), b"z".to_vec(), 1);
        let debug_str = format!("{:?}", r);

        assert!(debug_str.contains("KeyRange"));
        assert!(debug_str.contains("start"));
        assert!(debug_str.contains("end"));
        assert!(debug_str.contains("segment_id"));
    }

    #[test]
    fn test_level_stats_clone() {
        let mut stats = LevelStats::default();
        stats.total_size = 100;
        stats.num_segments = 5;
        stats.num_reads = 10;
        stats.bytes_read = 1000;
        stats.num_compactions = 2;
        stats.bytes_written = 500;

        let cloned = stats.clone();

        assert_eq!(cloned.total_size, 100);
        assert_eq!(cloned.num_segments, 5);
        assert_eq!(cloned.num_reads, 10);
        assert_eq!(cloned.bytes_read, 1000);
        assert_eq!(cloned.num_compactions, 2);
        assert_eq!(cloned.bytes_written, 500);
    }

    #[test]
    fn test_compaction_strategy_clone() {
        let s1 = CompactionStrategy::default_l0();
        let s2 = s1.clone();

        assert_eq!(s1, s2);

        let s3 = CompactionStrategy::Universal {
            max_size_amplification: 1.5,
            size_ratio: 0.9,
        };
        let s4 = s3.clone();

        assert_eq!(s3, s4);
    }

    #[test]
    fn test_compaction_strategy_debug() {
        let tiered = CompactionStrategy::default_l0();
        let debug_str = format!("{:?}", tiered);
        assert!(debug_str.contains("Tiered"));

        let leveled = CompactionStrategy::default_leveled();
        let debug_str = format!("{:?}", leveled);
        assert!(debug_str.contains("Leveled"));

        let universal = CompactionStrategy::Universal {
            max_size_amplification: 2.0,
            size_ratio: 1.0,
        };
        let debug_str = format!("{:?}", universal);
        assert!(debug_str.contains("Universal"));
    }

    #[test]
    fn test_compaction_strategy_copy() {
        let s1 = CompactionStrategy::default_l0();
        let s2 = s1; // Copy

        assert_eq!(s1, s2);
    }

    #[test]
    fn test_level_clone() {
        let level = Level::new(
            1,
            CompactionStrategy::default_leveled(),
            1024 * 1024,
            64 * 1024,
        );

        let cloned = level.clone();

        assert_eq!(cloned.level_num, level.level_num);
        assert_eq!(cloned.max_size, level.max_size);
        assert_eq!(cloned.target_file_size, level.target_file_size);
        assert_eq!(cloned.strategy, level.strategy);
    }

    #[test]
    fn test_version_set_clone() {
        let v1 = VersionSet::new(42, 5);
        let v2 = v1.clone();

        assert_eq!(v2.sequence, 42);
        assert_eq!(v2.num_levels(), 5);
        assert_eq!(v2.total_segments, v1.total_segments);
        assert_eq!(v2.total_size, v1.total_size);
    }

    #[test]
    fn test_key_range_overlaps_edge_cases() {
        // Ranges that touch exactly at boundary
        let r1 = KeyRange::new(b"a".to_vec(), b"b".to_vec(), 1);
        let r2 = KeyRange::new(b"b".to_vec(), b"c".to_vec(), 2);

        // They overlap at exactly "b"
        assert!(r1.overlaps(&r2));
        assert!(r2.overlaps(&r1));

        // Self overlap
        assert!(r1.overlaps(&r1));
    }

    #[test]
    fn test_key_range_contains_boundary() {
        let range = KeyRange::new(b"aaa".to_vec(), b"zzz".to_vec(), 1);

        // Exact boundaries
        assert!(range.contains(b"aaa"));
        assert!(range.contains(b"zzz"));

        // Just inside
        assert!(range.contains(b"aab"));
        assert!(range.contains(b"zzy"));

        // Outside - lexicographically less than start
        assert!(!range.contains(b"aa")); // shorter, lexicographically less than "aaa"
        assert!(!range.contains(b"aA")); // 'A' < 'a' in ASCII

        // Inside - "aaaa" > "aaa" and "aaaa" < "zzz"
        assert!(range.contains(b"aaaa"));

        // Outside - lexicographically greater than end
        assert!(!range.contains(b"zzzz")); // "zzzz" > "zzz"
    }

    #[test]
    fn test_level_stats_debug() {
        let stats = LevelStats::default();
        let debug_str = format!("{:?}", stats);

        assert!(debug_str.contains("LevelStats"));
        assert!(debug_str.contains("total_size"));
        assert!(debug_str.contains("num_segments"));
    }

    #[test]
    fn test_version_set_sequence_starts_correctly() {
        let v1 = VersionSet::new(0, 5);
        assert_eq!(v1.sequence, 0);

        let v2 = VersionSet::new(100, 5);
        assert_eq!(v2.sequence, 100);

        let v3 = VersionSet::new(u64::MAX, 5);
        assert_eq!(v3.sequence, u64::MAX);
    }

    #[test]
    fn test_level_total_size_reflects_stats() {
        let mut level = Level::new(
            1,
            CompactionStrategy::default_leveled(),
            1024 * 1024,
            64 * 1024,
        );

        assert_eq!(level.total_size(), 0);

        // Manually modify stats to simulate segments being added
        level.stats.total_size = 12345;
        assert_eq!(level.total_size(), 12345);
    }

    #[test]
    fn test_key_range_eq() {
        let r1 = KeyRange::new(b"a".to_vec(), b"b".to_vec(), 1);
        let r2 = KeyRange::new(b"a".to_vec(), b"b".to_vec(), 1);
        let r3 = KeyRange::new(b"a".to_vec(), b"b".to_vec(), 2); // Different segment_id
        let r4 = KeyRange::new(b"a".to_vec(), b"c".to_vec(), 1); // Different end

        assert_eq!(r1, r2);
        assert_ne!(r1, r3); // segment_id differs
        assert_ne!(r1, r4); // end differs
    }

    #[test]
    fn test_compaction_strategy_partial_eq() {
        let t1 = CompactionStrategy::Tiered {
            size_ratio: 4.0,
            min_merge_width: 4,
            max_merge_width: 10,
        };
        let t2 = CompactionStrategy::Tiered {
            size_ratio: 4.0,
            min_merge_width: 4,
            max_merge_width: 10,
        };
        let t3 = CompactionStrategy::Tiered {
            size_ratio: 5.0, // Different
            min_merge_width: 4,
            max_merge_width: 10,
        };

        assert_eq!(t1, t2);
        assert_ne!(t1, t3);

        let l1 = CompactionStrategy::default_leveled();
        assert_ne!(t1, l1);
    }

    #[test]
    fn test_level_find_segments_for_key_with_ranges() {
        let mut level = Level::new(
            1,
            CompactionStrategy::default_leveled(),
            1024 * 1024,
            64 * 1024,
        );

        // Manually add key ranges (simulating segments being added)
        level
            .key_ranges
            .push(KeyRange::new(b"a".to_vec(), b"d".to_vec(), 1));
        level
            .key_ranges
            .push(KeyRange::new(b"e".to_vec(), b"h".to_vec(), 2));
        level
            .key_ranges
            .push(KeyRange::new(b"i".to_vec(), b"l".to_vec(), 3));

        // Key in first range
        let segments = level.find_segments_for_key(b"b");
        assert_eq!(segments, vec![1]);

        // Key in second range
        let segments = level.find_segments_for_key(b"f");
        assert_eq!(segments, vec![2]);

        // Key in third range
        let segments = level.find_segments_for_key(b"j");
        assert_eq!(segments, vec![3]);

        // Key at boundary
        let segments = level.find_segments_for_key(b"a");
        assert_eq!(segments, vec![1]);

        // Key not in any range
        let segments = level.find_segments_for_key(b"z");
        assert!(segments.is_empty());
    }

    #[test]
    fn test_level_find_segments_for_key_overlapping_ranges() {
        let mut level = Level::new(
            1,
            CompactionStrategy::default_l0(), // Tiered allows overlaps
            1024 * 1024,
            64 * 1024,
        );

        // Add overlapping ranges (valid for tiered compaction)
        level
            .key_ranges
            .push(KeyRange::new(b"a".to_vec(), b"f".to_vec(), 1));
        level
            .key_ranges
            .push(KeyRange::new(b"c".to_vec(), b"h".to_vec(), 2));
        level
            .key_ranges
            .push(KeyRange::new(b"e".to_vec(), b"j".to_vec(), 3));

        // Key "d" is in first two ranges
        let segments = level.find_segments_for_key(b"d");
        assert_eq!(segments.len(), 2);
        assert!(segments.contains(&1));
        assert!(segments.contains(&2));

        // Key "f" is in all three ranges
        let segments = level.find_segments_for_key(b"f");
        assert_eq!(segments.len(), 3);
        assert!(segments.contains(&1));
        assert!(segments.contains(&2));
        assert!(segments.contains(&3));
    }

    #[test]
    fn test_level_find_overlapping_segments_with_ranges() {
        let mut level = Level::new(
            1,
            CompactionStrategy::default_leveled(),
            1024 * 1024,
            64 * 1024,
        );

        // Add non-overlapping key ranges
        level
            .key_ranges
            .push(KeyRange::new(b"a".to_vec(), b"c".to_vec(), 1));
        level
            .key_ranges
            .push(KeyRange::new(b"d".to_vec(), b"f".to_vec(), 2));
        level
            .key_ranges
            .push(KeyRange::new(b"g".to_vec(), b"i".to_vec(), 3));
        level
            .key_ranges
            .push(KeyRange::new(b"j".to_vec(), b"l".to_vec(), 4));

        // Query that overlaps first two
        let segments = level.find_overlapping_segments(b"b", b"e");
        assert_eq!(segments.len(), 2);
        assert!(segments.contains(&1));
        assert!(segments.contains(&2));

        // Query that overlaps last two
        let segments = level.find_overlapping_segments(b"h", b"k");
        assert_eq!(segments.len(), 2);
        assert!(segments.contains(&3));
        assert!(segments.contains(&4));

        // Query that overlaps all
        let segments = level.find_overlapping_segments(b"a", b"z");
        assert_eq!(segments.len(), 4);

        // Query that overlaps none
        let segments = level.find_overlapping_segments(b"m", b"z");
        assert!(segments.is_empty());

        // Query exactly matching one range
        let segments = level.find_overlapping_segments(b"d", b"f");
        assert_eq!(segments, vec![2]);
    }

    #[test]
    fn test_level_score_with_stats() {
        let mut level = Level::new(
            1,
            CompactionStrategy::default_leveled(),
            1000, // max_size = 1000
            64,
        );

        // Empty level - score 0
        assert_eq!(level.score(), 0.0);

        // Half full
        level.stats.total_size = 500;
        assert!((level.score() - 0.5).abs() < 0.001);

        // Exactly full
        level.stats.total_size = 1000;
        assert!((level.score() - 1.0).abs() < 0.001);

        // Over capacity
        level.stats.total_size = 1500;
        assert!((level.score() - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_level_score_tiered_strategy() {
        let mut level = Level::new(
            0,
            CompactionStrategy::Tiered {
                size_ratio: 4.0,
                min_merge_width: 4,
                max_merge_width: 10,
            },
            1000, // max_size
            64,
        );

        // With tiered, score is max of size_score and file_score
        level.stats.total_size = 200; // size_score = 0.2
        level.stats.num_segments = 8; // file_score = 0.8

        // Score should be max(0.2, 0.8) = 0.8
        assert!((level.score() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_version_set_total_size_and_segments() {
        let mut v = VersionSet::new(1, 5);

        assert_eq!(v.total_size, 0);
        assert_eq!(v.total_segments, 0);

        // Simulate adding to levels
        v.levels[0].stats.total_size = 100;
        v.levels[0].stats.num_segments = 2;
        v.levels[1].stats.total_size = 500;
        v.levels[1].stats.num_segments = 5;

        // total_size and total_segments are separate from level stats
        // (they track L0 + would need to be updated separately)
        v.total_size = 600;
        v.total_segments = 7;

        assert_eq!(v.total_size, 600);
        assert_eq!(v.total_segments, 7);
    }

    #[test]
    fn test_version_set_pick_compaction_level_l0_trigger() {
        let v = VersionSet::new(1, 5);

        // Add segments to L0 to trigger compaction (threshold is 4)
        // We need to create mock segments, but since pick_compaction_level
        // only checks l0.len(), we can just add empty Arcs if we had them.
        // For now, let's test the threshold logic directly by checking
        // the function returns None when l0 is empty.
        assert!(v.pick_compaction_level().is_none());

        // The actual L0 compaction trigger test would require adding Arc<Segment>
        // objects, which is complex. The test above verifies the empty case.
    }

    #[test]
    fn test_version_set_pick_compaction_level_l1_plus() {
        let mut v = VersionSet::new(1, 5);

        // Set L1 to exceed its max size (score > 1.0)
        v.levels[0].stats.total_size = v.levels[0].max_size * 2; // 2x over capacity

        let result = v.pick_compaction_level();
        assert!(result.is_some());

        let (level_num, score) = result.unwrap();
        assert_eq!(level_num, 1); // L1 (levels[0])
        assert!(score > 1.0);
    }

    #[test]
    fn test_version_set_pick_compaction_level_highest_priority() {
        let mut v = VersionSet::new(1, 5);

        // Set multiple levels over capacity
        v.levels[0].stats.total_size = v.levels[0].max_size * 2; // 2x
        v.levels[1].stats.total_size = v.levels[1].max_size * 3; // 3x
        v.levels[2].stats.total_size = v.levels[2].max_size * 1; // 1x (not over)

        let result = v.pick_compaction_level();
        assert!(result.is_some());

        let (level_num, score) = result.unwrap();
        // L2 has highest score (3.0), so it should be picked
        assert_eq!(level_num, 2); // L2 (levels[1])
        assert!((score - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_version_set_pick_compaction_level_all_below_threshold() {
        let mut v = VersionSet::new(1, 5);

        // Set all levels below capacity (score < 1.0)
        v.levels[0].stats.total_size = v.levels[0].max_size / 2; // 0.5
        v.levels[1].stats.total_size = v.levels[1].max_size / 4; // 0.25
        v.levels[2].stats.total_size = v.levels[2].max_size / 10; // 0.1

        // Should return None since no level has score > 1.0
        let result = v.pick_compaction_level();
        assert!(result.is_none());
    }

    #[test]
    fn test_key_range_sorting() {
        let mut ranges = vec![
            KeyRange::new(b"cherry".to_vec(), b"date".to_vec(), 3),
            KeyRange::new(b"apple".to_vec(), b"banana".to_vec(), 1),
            KeyRange::new(b"fig".to_vec(), b"grape".to_vec(), 4),
            KeyRange::new(b"apple".to_vec(), b"apricot".to_vec(), 2),
        ];

        ranges.sort();

        // Should be sorted by start key, then end key
        assert_eq!(ranges[0].segment_id, 2); // apple-apricot
        assert_eq!(ranges[1].segment_id, 1); // apple-banana
        assert_eq!(ranges[2].segment_id, 3); // cherry-date
        assert_eq!(ranges[3].segment_id, 4); // fig-grape
    }

    #[test]
    fn test_level_key_ranges_sorted_for_leveled() {
        let mut level = Level::new(
            1,
            CompactionStrategy::default_leveled(), // Non-overlapping
            1024 * 1024,
            64 * 1024,
        );

        // Add ranges out of order
        level
            .key_ranges
            .push(KeyRange::new(b"m".to_vec(), b"p".to_vec(), 3));
        level
            .key_ranges
            .push(KeyRange::new(b"a".to_vec(), b"d".to_vec(), 1));
        level
            .key_ranges
            .push(KeyRange::new(b"e".to_vec(), b"h".to_vec(), 2));

        // For leveled strategy, ranges should be sorted (though we added them directly)
        // The add_segment method would sort them, so let's verify the sort behavior
        level.key_ranges.sort();

        assert_eq!(level.key_ranges[0].segment_id, 1); // a-d
        assert_eq!(level.key_ranges[1].segment_id, 2); // e-h
        assert_eq!(level.key_ranges[2].segment_id, 3); // m-p
    }

    #[test]
    fn test_level_remove_segment_not_found() {
        let mut level = Level::new(
            1,
            CompactionStrategy::default_leveled(),
            1024 * 1024,
            64 * 1024,
        );

        // Try to remove from empty level
        let result = level.remove_segment(999);
        assert!(result.is_none());
    }

    #[test]
    fn test_version_set_max_size_exponential_growth() {
        let v = VersionSet::new(1, 7);

        // Verify exponential growth: L1 = 64MB * 10^0, L2 = 64MB * 10^1, etc.
        let base = 64 * 1024 * 1024u64;

        assert_eq!(v.levels[0].max_size, base); // L1: 64MB
        assert_eq!(v.levels[1].max_size, base * 10); // L2: 640MB
        assert_eq!(v.levels[2].max_size, base * 100); // L3: 6.4GB
        assert_eq!(v.levels[3].max_size, base * 1000); // L4: 64GB
        assert_eq!(v.levels[4].max_size, base * 10000); // L5: 640GB
        assert_eq!(v.levels[5].max_size, base * 100000); // L6: 6.4TB
    }

    #[test]
    fn test_level_stats_all_fields() {
        let mut stats = LevelStats {
            total_size: 1024,
            num_segments: 10,
            num_reads: 100,
            bytes_read: 5000,
            num_compactions: 5,
            bytes_written: 2000,
        };

        // Verify all fields
        assert_eq!(stats.total_size, 1024);
        assert_eq!(stats.num_segments, 10);
        assert_eq!(stats.num_reads, 100);
        assert_eq!(stats.bytes_read, 5000);
        assert_eq!(stats.num_compactions, 5);
        assert_eq!(stats.bytes_written, 2000);

        // Modify and verify
        stats.num_reads += 1;
        assert_eq!(stats.num_reads, 101);
    }

    #[test]
    fn test_key_range_binary_data() {
        // Test with binary data including null bytes
        let start = vec![0x00, 0x01, 0x02];
        let end = vec![0xFF, 0xFE, 0xFD];
        let range = KeyRange::new(start.clone(), end.clone(), 1);

        assert!(range.contains(&[0x00, 0x01, 0x02]));
        assert!(range.contains(&[0x80, 0x80, 0x80]));
        assert!(range.contains(&[0xFF, 0xFE, 0xFD]));
    }

    #[test]
    fn test_compaction_strategy_universal_scoring() {
        let stats = LevelStats {
            total_size: 2000,
            num_segments: 5,
            ..Default::default()
        };

        let strategy = CompactionStrategy::Universal {
            max_size_amplification: 1.5,
            size_ratio: 1.0,
        };

        // space_amp = 2000 / 1000 = 2.0
        // score = 2.0 / 1.5 = 1.333...
        let score = stats.score(1000, &strategy);
        assert!((score - (2.0 / 1.5)).abs() < 0.001);
    }

    #[test]
    fn test_version_set_l0_is_separate() {
        let v = VersionSet::new(1, 5);

        // L0 is separate from levels array
        assert!(v.l0.is_empty());
        assert_eq!(v.levels.len(), 5);

        // levels[0] is L1, not L0
        assert_eq!(v.levels[0].level_num, 1);
        assert_eq!(v.levels[1].level_num, 2);
    }

    #[test]
    fn test_level_num_assignment() {
        let v = VersionSet::new(1, 7);

        // Verify level numbers are assigned correctly
        for (i, level) in v.levels.iter().enumerate() {
            assert_eq!(level.level_num as usize, i + 1);
        }
    }

    #[test]
    fn test_key_range_with_long_keys() {
        // Test with longer keys
        let start = "a".repeat(1000).into_bytes();
        let end = "z".repeat(1000).into_bytes();
        let range = KeyRange::new(start, end, 1);

        let mid = "m".repeat(1000).into_bytes();
        assert!(range.contains(&mid));

        let before = "0".repeat(1000).into_bytes();
        assert!(!range.contains(&before));
    }
}
