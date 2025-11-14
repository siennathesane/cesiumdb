//! Zero-copy merge iterator for efficient compaction
//!
//! This module provides a high-performance merge iterator that combines
//! multiple sorted iterators without copying data unnecessarily.

use crate::keypair::{KeyBytes, ValueBytes};
use crate::simd::simd_compare_keys;
use bytes::Bytes;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Entry from a merge source
#[derive(Clone)]
pub struct MergeEntry {
    /// The key (zero-copy reference)
    pub key: KeyBytes,

    /// The value (zero-copy reference)
    pub value: ValueBytes,

    /// Source index (which iterator this came from)
    source_index: usize,
}

/// Wrapper for heap ordering (reverse order for min-heap)
struct HeapEntry {
    entry: MergeEntry,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.entry.key == other.entry.key
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        // Use SIMD-optimized comparison for better performance
        other.entry.key.simd_cmp(&self.entry.key)
            .then_with(|| {
                // For equal keys, prefer newer (higher source index = newer level)
                other.entry.source_index.cmp(&self.entry.source_index)
            })
    }
}

/// Trait for sources that can be merged
pub trait MergeSource: Iterator<Item = Result<(KeyBytes, ValueBytes), MergeError>> {
    /// Peek at the next entry without consuming it
    fn peek(&mut self) -> Option<&Result<(KeyBytes, ValueBytes), MergeError>>;
}

/// Error type for merge operations
#[derive(Debug, Clone)]
pub enum MergeError {
    /// Source iterator returned an error
    SourceError(String),

    /// Corrupt data detected
    CorruptData(String),
}

impl std::fmt::Display for MergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MergeError::SourceError(msg) => write!(f, "Source error: {}", msg),
            MergeError::CorruptData(msg) => write!(f, "Corrupt data: {}", msg),
        }
    }
}

impl std::error::Error for MergeError {}

/// Zero-copy merge iterator
///
/// Merges multiple sorted iterators into a single sorted stream,
/// automatically handling:
/// - Deduplication of keys across sources
/// - Tombstone filtering
/// - Zero-copy references to underlying data
pub struct ZeroCopyMergeIterator<I>
where
    I: Iterator<Item = Result<(KeyBytes, ValueBytes), MergeError>>,
{
    /// Min-heap of current entries from each source
    heap: BinaryHeap<HeapEntry>,

    /// The source iterators
    sources: Vec<I>,

    /// Whether to filter tombstones
    filter_tombstones: bool,

    /// Statistics
    keys_merged: u64,
    keys_deduplicated: u64,
    tombstones_filtered: u64,
}

impl<I> ZeroCopyMergeIterator<I>
where
    I: Iterator<Item = Result<(KeyBytes, ValueBytes), MergeError>>,
{
    /// Creates a new merge iterator from multiple sources
    ///
    /// Sources should be ordered from oldest to newest (L0 last).
    /// For duplicate keys, the newest (last) source wins.
    pub fn new(sources: Vec<I>) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(sources.len()),
            sources,
            filter_tombstones: true,
            keys_merged: 0,
            keys_deduplicated: 0,
            tombstones_filtered: 0,
        }
    }

    /// Creates a merge iterator that includes tombstones
    pub fn with_tombstones(mut self) -> Self {
        self.filter_tombstones = false;
        self
    }

    /// Initializes the heap by pulling the first entry from each source
    fn init(&mut self) -> Result<(), MergeError> {
        for (idx, source) in self.sources.iter_mut().enumerate() {
            if let Some(result) = source.next() {
                let (key, value) = result?;
                self.heap.push(HeapEntry {
                    entry: MergeEntry {
                        key,
                        value,
                        source_index: idx,
                    },
                });
            }
        }
        Ok(())
    }

    /// Returns statistics about the merge
    pub fn stats(&self) -> MergeStats {
        MergeStats {
            keys_merged: self.keys_merged,
            keys_deduplicated: self.keys_deduplicated,
            tombstones_filtered: self.tombstones_filtered,
        }
    }
}

impl<I> Iterator for ZeroCopyMergeIterator<I>
where
    I: Iterator<Item = Result<(KeyBytes, ValueBytes), MergeError>>,
{
    type Item = Result<(KeyBytes, ValueBytes), MergeError>;

    fn next(&mut self) -> Option<Self::Item> {
        // Initialize on first call
        if self.keys_merged == 0 && self.heap.is_empty() {
            if let Err(e) = self.init() {
                return Some(Err(e));
            }
        }

        loop {
            // Pop the smallest key
            let heap_entry = self.heap.pop()?;
            let entry = heap_entry.entry;

            // Advance the source that provided this entry
            let source_idx = entry.source_index;
            if let Some(result) = self.sources[source_idx].next() {
                match result {
                    Ok((key, value)) => {
                        self.heap.push(HeapEntry {
                            entry: MergeEntry {
                                key,
                                value,
                                source_index: source_idx,
                            },
                        });
                    }
                    Err(e) => return Some(Err(e)),
                }
            }

            // Skip duplicate keys (keep only the first, which has the highest timestamp)
            // Compare just (ns, key), not the full KeyBytes including timestamp
            let mut duplicates = 0;
            while let Some(next_entry) = self.heap.peek() {
                let same_logical_key = entry.key.ns() == next_entry.entry.key.ns()
                    && entry.key.key() == next_entry.entry.key.key();

                if same_logical_key {
                    // Same logical key (ignoring timestamp), skip it
                    let dup = self.heap.pop().unwrap();
                    duplicates += 1;

                    // Advance the source
                    let dup_source_idx = dup.entry.source_index;
                    if let Some(result) = self.sources[dup_source_idx].next() {
                        match result {
                            Ok((key, value)) => {
                                self.heap.push(HeapEntry {
                                    entry: MergeEntry {
                                        key,
                                        value,
                                        source_index: dup_source_idx,
                                    },
                                });
                            }
                            Err(e) => return Some(Err(e)),
                        }
                    }
                } else {
                    break;
                }
            }

            self.keys_merged += 1;
            self.keys_deduplicated += duplicates;

            // Filter tombstones if requested
            if self.filter_tombstones && entry.value.is_tombstone() {
                self.tombstones_filtered += 1;
                continue;
            }

            return Some(Ok((entry.key, entry.value)));
        }
    }
}

/// Statistics from a merge operation
#[derive(Debug, Clone, Copy)]
pub struct MergeStats {
    /// Total number of keys merged
    pub keys_merged: u64,

    /// Number of duplicate keys deduplicated
    pub keys_deduplicated: u64,

    /// Number of tombstones filtered
    pub tombstones_filtered: u64,
}

impl std::fmt::Display for MergeStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Merge: {} keys ({} dedup, {} tombstones)",
            self.keys_merged, self.keys_deduplicated, self.tombstones_filtered
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    fn make_key(ns: u64, key: &[u8], ts: u128) -> KeyBytes {
        KeyBytes::new(ns, Bytes::copy_from_slice(key), ts)
    }

    fn make_value(ns: u64, value: &[u8]) -> ValueBytes {
        ValueBytes::new(ns, Bytes::copy_from_slice(value))
    }

    fn make_tombstone(ns: u64) -> ValueBytes {
        ValueBytes::new_tombstone(ns)
    }

    #[test]
    fn test_merge_two_sources() {
        let source1 = vec![
            Ok((make_key(0, b"a", 100), make_value(0, b"v1"))),
            Ok((make_key(0, b"c", 100), make_value(0, b"v3"))),
        ];

        let source2 = vec![
            Ok((make_key(0, b"b", 100), make_value(0, b"v2"))),
            Ok((make_key(0, b"d", 100), make_value(0, b"v4"))),
        ];

        let mut merger = ZeroCopyMergeIterator::new(vec![
            source1.into_iter(),
            source2.into_iter(),
        ]);

        let results: Vec<_> = merger.by_ref().collect::<Result<Vec<_>, _>>().unwrap();

        assert_eq!(results.len(), 4);
        assert_eq!(&results[0].0.key()[..], b"a");
        assert_eq!(&results[1].0.key()[..], b"b");
        assert_eq!(&results[2].0.key()[..], b"c");
        assert_eq!(&results[3].0.key()[..], b"d");

        let stats = merger.stats();
        assert_eq!(stats.keys_merged, 4);
        assert_eq!(stats.keys_deduplicated, 0);
    }

    #[test]
    fn test_deduplication() {
        // Source 1 (older)
        let source1 = vec![
            Ok((make_key(0, b"a", 100), make_value(0, b"old"))),
            Ok((make_key(0, b"b", 100), make_value(0, b"old"))),
        ];

        // Source 2 (newer - should win)
        let source2 = vec![
            Ok((make_key(0, b"a", 200), make_value(0, b"new"))),
            Ok((make_key(0, b"c", 200), make_value(0, b"new"))),
        ];

        let mut merger = ZeroCopyMergeIterator::new(vec![
            source1.into_iter(),
            source2.into_iter(),
        ]);

        let results: Vec<_> = merger.by_ref().collect::<Result<Vec<_>, _>>().unwrap();

        assert_eq!(results.len(), 3);

        // Key "a" should have the newer value
        assert_eq!(&results[0].0.key()[..], b"a");
        assert_eq!(&results[0].1.as_bytes()[..], b"new");

        assert_eq!(&results[1].0.key()[..], b"b");
        assert_eq!(&results[2].0.key()[..], b"c");

        let stats = merger.stats();
        assert_eq!(stats.keys_merged, 3);
        assert_eq!(stats.keys_deduplicated, 1);
    }

    #[test]
    fn test_tombstone_filtering() {
        let source1 = vec![
            Ok((make_key(0, b"a", 100), make_value(0, b"v1"))),
            Ok((make_key(0, b"b", 100), make_tombstone(0))),
            Ok((make_key(0, b"c", 100), make_value(0, b"v3"))),
        ];

        let mut merger = ZeroCopyMergeIterator::new(vec![source1.into_iter()]);

        let results: Vec<_> = merger.by_ref().collect::<Result<Vec<_>, _>>().unwrap();

        // Tombstone should be filtered
        assert_eq!(results.len(), 2);
        assert_eq!(&results[0].0.key()[..], b"a");
        assert_eq!(&results[1].0.key()[..], b"c");

        let stats = merger.stats();
        assert_eq!(stats.tombstones_filtered, 1);
    }

    #[test]
    fn test_tombstone_preservation() {
        let source1 = vec![
            Ok((make_key(0, b"a", 100), make_value(0, b"v1"))),
            Ok((make_key(0, b"b", 100), make_tombstone(0))),
            Ok((make_key(0, b"c", 100), make_value(0, b"v3"))),
        ];

        let mut merger = ZeroCopyMergeIterator::new(vec![source1.into_iter()])
            .with_tombstones();

        let results: Vec<_> = merger.by_ref().collect::<Result<Vec<_>, _>>().unwrap();

        // Tombstone should be kept
        assert_eq!(results.len(), 3);
        assert_eq!(&results[1].0.key()[..], b"b");
        assert!(results[1].1.is_tombstone());

        let stats = merger.stats();
        assert_eq!(stats.tombstones_filtered, 0);
    }

    #[test]
    fn test_empty_sources() {
        let source1: Vec<Result<(KeyBytes, ValueBytes), MergeError>> = vec![];
        let source2: Vec<Result<(KeyBytes, ValueBytes), MergeError>> = vec![];

        let merger = ZeroCopyMergeIterator::new(vec![
            source1.into_iter(),
            source2.into_iter(),
        ]);

        let results: Vec<_> = merger.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_many_sources() {
        let mut sources = Vec::new();

        for i in 0..10 {
            let source = vec![
                Ok((make_key(0, format!("key{:02}", i * 2).as_bytes(), 100), make_value(0, b"value"))),
                Ok((make_key(0, format!("key{:02}", i * 2 + 1).as_bytes(), 100), make_value(0, b"value"))),
            ];
            sources.push(source.into_iter());
        }

        let merger = ZeroCopyMergeIterator::new(sources);
        let results: Vec<_> = merger.collect::<Result<Vec<_>, _>>().unwrap();

        assert_eq!(results.len(), 20);

        // Verify sorted order
        for i in 1..results.len() {
            assert!(results[i - 1].0 < results[i].0);
        }
    }
}
