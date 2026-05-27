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
            | CompactionStrategy::Tiered { .. } => {
                // Score based on number of files and size ratio
                let size_score = self.total_size as f64 / max_size as f64;
                let file_score = self.num_segments as f64 / 10.0;
                size_score.max(file_score)
            },
            | CompactionStrategy::Leveled { .. } => {
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
/// Each level contains multiple segments (Segments) and maintains
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

    /// Binary-search for the single segment that may contain `key`.
    ///
    /// `key` must be the *user-key prefix* (`[ns:8][user_key]`) without the
    /// 16-byte timestamp suffix.  This is only valid for leveled compaction
    /// where key ranges are non-overlapping and sorted by user key.
    /// Returns `None` if no range contains the key.
    pub fn find_segment_for_key_binary(&self, key: &[u8]) -> Option<u64> {
        if self.key_ranges.is_empty() {
            return None;
        }
        // Binary search on the user-key prefix of `start`.
        let mut lo = 0usize;
        let mut hi = self.key_ranges.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            let start_prefix =
                &self.key_ranges[mid].start[..self.key_ranges[mid].start.len().saturating_sub(16)];
            if start_prefix <= key {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // `lo` is the first range whose start > key, so candidate is `lo - 1`.
        let idx = lo.saturating_sub(1);
        self.key_ranges.get(idx).and_then(|range| {
            let start_prefix = &range.start[..range.start.len().saturating_sub(16)];
            let end_prefix = &range.end[..range.end.len().saturating_sub(16)];
            if key >= start_prefix && key <= end_prefix {
                Some(range.segment_id)
            } else {
                None
            }
        })
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

    /// Key ranges for L0 segments (parallel to l0)
    ///
    /// Each entry corresponds to the key range of the segment at the same
    /// index in the l0 vector. This parallel array is kept in sync with l0.
    pub l0_key_ranges: Vec<KeyRange>,

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
            l0_key_ranges: Vec::new(),
            levels,
            total_segments: 0,
            total_size: 0,
        }
    }

    /// Returns the highest segment ID across all levels
    ///
    /// Used to initialize next_segment_id on recovery to avoid overwriting
    /// existing segments.
    pub fn max_segment_id(&self) -> u64 {
        let mut max_id = 0u64;

        // Check L0
        for segment in &self.l0 {
            max_id = max_id.max(segment.id());
        }

        // Check L1+
        for level in &self.levels {
            for segment in &level.segments {
                max_id = max_id.max(segment.id());
            }
        }

        max_id
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

    /// Adds a segment to L0 with its key range
    pub fn add_to_l0(&mut self, segment: Arc<Segment>, key_range: KeyRange) {
        self.total_size += segment.size_in_bytes();
        self.total_segments += 1;
        self.l0.push(segment);
        self.l0_key_ranges.push(key_range);
    }

    /// Removes a segment from L0 by ID
    ///
    /// Returns the removed segment if found, keeping the parallel key_ranges
    /// array in sync.
    pub fn remove_from_l0(&mut self, segment_id: u64) -> Option<Arc<Segment>> {
        if let Some(idx) = self.l0.iter().position(|s| s.id() == segment_id) {
            let segment = self.l0.remove(idx);
            self.l0_key_ranges.remove(idx);
            self.total_size -= segment.size_in_bytes();
            self.total_segments -= 1;
            Some(segment)
        } else {
            None
        }
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
    fn test_level_max_sizes() {
        let v = VersionSet::new(1, 5);

        // L1 = 64MB, L2 = 640MB, L3 = 6.4GB, etc.
        assert_eq!(v.levels[0].max_size, 64 * 1024 * 1024);
        assert_eq!(v.levels[1].max_size, 640 * 1024 * 1024);
        assert_eq!(v.levels[2].max_size, 6400 * 1024 * 1024);
    }
}
