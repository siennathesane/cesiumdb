//! Range deletion optimization
//!
//! Efficiently handles deletion of large contiguous key ranges by using
//! range tombstones instead of individual per-key tombstones.

use crate::levels::KeyRange;
use bytes::Bytes;
use std::cmp::Ordering;

/// A range tombstone marking deletion of all keys in a range
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangeTombstone {
    /// Start of the range (inclusive)
    pub start: Bytes,

    /// End of the range (inclusive)
    pub end: Bytes,

    /// Sequence number when this range was deleted
    pub sequence: u64,

    /// Level where this tombstone was created
    pub level: u8,
}

impl RangeTombstone {
    /// Creates a new range tombstone
    pub fn new(start: Bytes, end: Bytes, sequence: u64, level: u8) -> Self {
        debug_assert!(start <= end, "start must be <= end");
        Self {
            start,
            end,
            sequence,
            level,
        }
    }

    /// Checks if this tombstone covers the given key
    pub fn covers(&self, key: &[u8]) -> bool {
        key >= self.start.as_ref() && key <= self.end.as_ref()
    }

    /// Checks if this tombstone overlaps with a key range
    pub fn overlaps(&self, range: &KeyRange) -> bool {
        self.start.as_ref() <= range.end.as_slice()
            && range.start.as_slice() <= self.end.as_ref()
    }

    /// Merges this tombstone with another if they overlap or are adjacent
    ///
    /// Returns Some(merged) if merge is possible, None otherwise.
    pub fn try_merge(&self, other: &RangeTombstone) -> Option<RangeTombstone> {
        // Only merge tombstones from the same level
        if self.level != other.level {
            return None;
        }

        // Check if ranges overlap or are adjacent
        if !self.overlaps_or_adjacent(other) {
            return None;
        }

        // Merge to the wider range and newer sequence
        let start = if self.start <= other.start {
            self.start.clone()
        } else {
            other.start.clone()
        };

        let end = if self.end >= other.end {
            self.end.clone()
        } else {
            other.end.clone()
        };

        let sequence = std::cmp::max(self.sequence, other.sequence);

        Some(RangeTombstone::new(start, end, sequence, self.level))
    }

    /// Checks if this tombstone overlaps or is adjacent to another
    fn overlaps_or_adjacent(&self, other: &RangeTombstone) -> bool {
        // Overlapping
        if self.start.as_ref() <= other.end.as_ref()
            && other.start.as_ref() <= self.end.as_ref()
        {
            return true;
        }

        // Adjacent (this.end + 1 == other.start or other.end + 1 == this.start)
        if let Some(next) = self.next_key() {
            if next.as_ref() == other.start.as_ref() {
                return true;
            }
        }

        if let Some(next) = other.next_key() {
            if next.as_ref() == self.start.as_ref() {
                return true;
            }
        }

        false
    }

    /// Computes the next key after end (for adjacency checking)
    fn next_key(&self) -> Option<Bytes> {
        let mut next = self.end.to_vec();

        // Increment the last byte
        for i in (0..next.len()).rev() {
            if next[i] < 255 {
                next[i] += 1;
                return Some(Bytes::from(next));
            }
            next[i] = 0;
        }

        // Overflow - no next key
        None
    }

    /// Splits this tombstone at the given key
    ///
    /// Returns (left, right) where left covers [start, key) and right covers [key, end].
    pub fn split_at(&self, key: &[u8]) -> (Option<RangeTombstone>, Option<RangeTombstone>) {
        if key <= self.start.as_ref() {
            // Key is before range - no left part
            return (None, Some(self.clone()));
        }

        if key > self.end.as_ref() {
            // Key is after range - no right part
            return (Some(self.clone()), None);
        }

        // Split in the middle
        let left = RangeTombstone::new(
            self.start.clone(),
            Bytes::copy_from_slice(key),
            self.sequence,
            self.level,
        );

        let right = RangeTombstone::new(
            Bytes::copy_from_slice(key),
            self.end.clone(),
            self.sequence,
            self.level,
        );

        (Some(left), Some(right))
    }
}

impl PartialOrd for RangeTombstone {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RangeTombstone {
    fn cmp(&self, other: &Self) -> Ordering {
        // Order by start key, then by end key
        self.start
            .cmp(&other.start)
            .then_with(|| self.end.cmp(&other.end))
            .then_with(|| other.sequence.cmp(&self.sequence)) // Newer sequence first
    }
}

/// Manager for range tombstones
pub struct RangeTombstoneManager {
    /// Active range tombstones, sorted by start key
    tombstones: Vec<RangeTombstone>,
}

impl RangeTombstoneManager {
    /// Creates a new empty manager
    pub fn new() -> Self {
        Self {
            tombstones: Vec::new(),
        }
    }

    /// Adds a range tombstone
    pub fn add(&mut self, tombstone: RangeTombstone) {
        self.tombstones.push(tombstone);
        self.tombstones.sort();
    }

    /// Checks if a key is covered by any range tombstone
    pub fn is_deleted(&self, key: &[u8]) -> bool {
        // Binary search for potential covering tombstone
        let idx = self
            .tombstones
            .binary_search_by(|t| {
                if key < t.start.as_ref() {
                    Ordering::Greater
                } else if key > t.end.as_ref() {
                    Ordering::Less
                } else {
                    Ordering::Equal
                }
            });

        match idx {
            Ok(_) => true, // Found exact match
            Err(i) => {
                // Check tombstone before and after insertion point
                if i > 0 && self.tombstones[i - 1].covers(key) {
                    return true;
                }
                if i < self.tombstones.len() && self.tombstones[i].covers(key) {
                    return true;
                }
                false
            }
        }
    }

    /// Returns all tombstones that overlap with the given range
    pub fn overlapping(&self, range: &KeyRange) -> Vec<&RangeTombstone> {
        self.tombstones
            .iter()
            .filter(|t| t.overlaps(range))
            .collect()
    }

    /// Compacts the tombstones by merging overlapping/adjacent ones
    pub fn compact(&mut self) {
        if self.tombstones.len() <= 1 {
            return;
        }

        self.tombstones.sort();

        let mut compacted = Vec::new();
        let mut current = self.tombstones[0].clone();

        for tombstone in self.tombstones.iter().skip(1) {
            if let Some(merged) = current.try_merge(tombstone) {
                current = merged;
            } else {
                compacted.push(current);
                current = tombstone.clone();
            }
        }

        compacted.push(current);
        self.tombstones = compacted;
    }

    /// Returns the number of tombstones
    pub fn len(&self) -> usize {
        self.tombstones.len()
    }

    /// Returns true if there are no tombstones
    pub fn is_empty(&self) -> bool {
        self.tombstones.is_empty()
    }

    /// Returns statistics about range deletions
    pub fn stats(&self) -> RangeDeletionStats {
        let total_tombstones = self.tombstones.len();

        let mut total_range_size = 0u64;
        for tombstone in &self.tombstones {
            // Estimate range size (rough approximation)
            if !tombstone.start.is_empty() && !tombstone.end.is_empty() {
                total_range_size += tombstone.end.len() as u64;
            }
        }

        RangeDeletionStats {
            active_tombstones: total_tombstones,
            total_range_size,
        }
    }
}

impl Default for RangeTombstoneManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about range deletions
#[derive(Debug, Clone, Copy)]
pub struct RangeDeletionStats {
    /// Number of active range tombstones
    pub active_tombstones: usize,

    /// Estimated total size of all ranges
    pub total_range_size: u64,
}

impl std::fmt::Display for RangeDeletionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Range Deletions: {} tombstones, ~{:.2}KB total",
            self.active_tombstones,
            self.total_range_size as f64 / 1024.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tombstone_creation() {
        let tombstone = RangeTombstone::new(
            Bytes::from_static(b"a"),
            Bytes::from_static(b"z"),
            100,
            0,
        );

        assert_eq!(tombstone.start.as_ref(), b"a");
        assert_eq!(tombstone.end.as_ref(), b"z");
        assert_eq!(tombstone.sequence, 100);
        assert_eq!(tombstone.level, 0);
    }

    #[test]
    fn test_tombstone_covers() {
        let tombstone = RangeTombstone::new(
            Bytes::from_static(b"a"),
            Bytes::from_static(b"z"),
            100,
            0,
        );

        assert!(tombstone.covers(b"a"));
        assert!(tombstone.covers(b"m"));
        assert!(tombstone.covers(b"z"));
        assert!(!tombstone.covers(b"A")); // Before range
        assert!(!tombstone.covers(b"zz")); // After range
    }

    #[test]
    fn test_tombstone_overlaps() {
        let tombstone = RangeTombstone::new(
            Bytes::from_static(b"d"),
            Bytes::from_static(b"m"),
            100,
            0,
        );
        let range1 = KeyRange::new(b"a".to_vec(), b"f".to_vec(), 1);
        let range2 = KeyRange::new(b"p".to_vec(), b"z".to_vec(), 2);

        assert!(tombstone.overlaps(&range1)); // Overlaps
        assert!(!tombstone.overlaps(&range2)); // No overlap
    }

    #[test]
    fn test_tombstone_merge() {
        let t1 = RangeTombstone::new(
            Bytes::from_static(b"a"),
            Bytes::from_static(b"m"),
            100,
            0,
        );
        let t2 = RangeTombstone::new(
            Bytes::from_static(b"k"),
            Bytes::from_static(b"z"),
            101,
            0,
        );

        let merged = t1.try_merge(&t2);
        assert!(merged.is_some());

        let merged = merged.unwrap();
        assert_eq!(merged.start.as_ref(), b"a");
        assert_eq!(merged.end.as_ref(), b"z");
        assert_eq!(merged.sequence, 101); // Newer sequence
    }

    #[test]
    fn test_tombstone_no_merge_different_level() {
        let t1 = RangeTombstone::new(
            Bytes::from_static(b"a"),
            Bytes::from_static(b"m"),
            100,
            0,
        );
        let t2 = RangeTombstone::new(
            Bytes::from_static(b"k"),
            Bytes::from_static(b"z"),
            101,
            1,
        );

        let merged = t1.try_merge(&t2);
        assert!(merged.is_none());
    }

    #[test]
    fn test_tombstone_split() {
        let tombstone = RangeTombstone::new(
            Bytes::from_static(b"a"),
            Bytes::from_static(b"z"),
            100,
            0,
        );

        let (left, right) = tombstone.split_at(b"m");

        assert!(left.is_some());
        assert!(right.is_some());

        let left = left.unwrap();
        let right = right.unwrap();

        assert_eq!(left.start.as_ref(), b"a");
        assert_eq!(left.end.as_ref(), b"m");
        assert_eq!(right.start.as_ref(), b"m");
        assert_eq!(right.end.as_ref(), b"z");
    }

    #[test]
    fn test_manager_creation() {
        let manager = RangeTombstoneManager::new();
        assert_eq!(manager.len(), 0);
        assert!(manager.is_empty());
    }

    #[test]
    fn test_manager_add() {
        let mut manager = RangeTombstoneManager::new();

        let t1 = RangeTombstone::new(
            Bytes::from_static(b"a"),
            Bytes::from_static(b"m"),
            100,
            0,
        );
        manager.add(t1);

        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn test_manager_is_deleted() {
        let mut manager = RangeTombstoneManager::new();

        let t1 = RangeTombstone::new(
            Bytes::from_static(b"d"),
            Bytes::from_static(b"m"),
            100,
            0,
        );
        manager.add(t1);

        assert!(manager.is_deleted(b"d"));
        assert!(manager.is_deleted(b"j"));
        assert!(manager.is_deleted(b"m"));
        assert!(!manager.is_deleted(b"a"));
        assert!(!manager.is_deleted(b"z"));
    }

    #[test]
    fn test_manager_overlapping() {
        let mut manager = RangeTombstoneManager::new();

        let t1 = RangeTombstone::new(
            Bytes::from_static(b"a"),
            Bytes::from_static(b"f"),
            100,
            0,
        );
        let t2 = RangeTombstone::new(
            Bytes::from_static(b"p"),
            Bytes::from_static(b"z"),
            101,
            0,
        );
        manager.add(t1);
        manager.add(t2);

        let range = KeyRange::new(b"d".to_vec(), b"r".to_vec(), 1);
        let overlapping = manager.overlapping(&range);

        assert_eq!(overlapping.len(), 2);
    }

    #[test]
    fn test_manager_compact() {
        let mut manager = RangeTombstoneManager::new();

        // Add overlapping tombstones
        manager.add(RangeTombstone::new(
            Bytes::from_static(b"a"),
            Bytes::from_static(b"f"),
            100,
            0,
        ));
        manager.add(RangeTombstone::new(
            Bytes::from_static(b"d"),
            Bytes::from_static(b"m"),
            101,
            0,
        ));
        manager.add(RangeTombstone::new(
            Bytes::from_static(b"k"),
            Bytes::from_static(b"z"),
            102,
            0,
        ));

        assert_eq!(manager.len(), 3);

        manager.compact();

        // Should merge into one tombstone
        assert_eq!(manager.len(), 1);
        assert_eq!(manager.tombstones[0].start.as_ref(), b"a");
        assert_eq!(manager.tombstones[0].end.as_ref(), b"z");
    }

    #[test]
    fn test_manager_compact_no_overlap() {
        let mut manager = RangeTombstoneManager::new();

        // Add non-overlapping tombstones
        manager.add(RangeTombstone::new(
            Bytes::from_static(b"a"),
            Bytes::from_static(b"f"),
            100,
            0,
        ));
        manager.add(RangeTombstone::new(
            Bytes::from_static(b"p"),
            Bytes::from_static(b"z"),
            101,
            0,
        ));

        assert_eq!(manager.len(), 2);

        manager.compact();

        // Should stay separate
        assert_eq!(manager.len(), 2);
    }

    #[test]
    fn test_tombstone_ordering() {
        let t1 = RangeTombstone::new(
            Bytes::from_static(b"a"),
            Bytes::from_static(b"f"),
            100,
            0,
        );
        let t2 = RangeTombstone::new(
            Bytes::from_static(b"p"),
            Bytes::from_static(b"z"),
            101,
            0,
        );

        assert!(t1 < t2);
    }

    #[test]
    fn test_stats() {
        let mut manager = RangeTombstoneManager::new();

        manager.add(RangeTombstone::new(
            Bytes::from_static(b"a"),
            Bytes::from_static(b"z"),
            100,
            0,
        ));
        manager.add(RangeTombstone::new(
            Bytes::from_static(b"aa"),
            Bytes::from_static(b"zz"),
            101,
            0,
        ));

        let stats = manager.stats();
        assert_eq!(stats.active_tombstones, 2);
        assert!(stats.total_range_size > 0);
    }
}
