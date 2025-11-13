// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

use std::{
    fs,
    path::PathBuf,
    sync::Arc,
};

use bytes::Bytes;
use rand::random;
use tracing::instrument;

use crate::{
    errs::SegmentError,
    keypair::{
        KeyBytes,
        ValueBytes,
    },
    memtable::Memtable,
    merge::MergeIterator,
    segment::Segment,
    segment_builder::SegmentBuilder,
    utils::Serializer,
};

/// Generic compaction function that merges multiple iterators into a single
/// segment. This works for both memtable-to-segment (flush) and
/// segment-to-segment (major compaction).
///
/// The function:
/// 1. Merges multiple iterators using MergeIterator
/// 2. Creates a new segment at the output path
/// 3. Writes all merged entries to the segment
/// 4. Closes and returns the segment
///
/// # Arguments
/// * `iterators` - Vec of iterators that yield (KeyBytes, ValueBytes)
/// * `output_path` - Path where the output segment file will be created
/// * `segment_id` - ID for the new segment
///
/// # Returns
/// * `Result<Arc<Segment>, SegmentError>` - The newly created segment
///
/// # Example
/// ```rust,ignore
/// // Compact memtables
/// let iter1 = memtable1.scan(Bound::Unbounded, Bound::Unbounded);
/// let iter2 = memtable2.scan(Bound::Unbounded, Bound::Unbounded);
/// let segment = compact(vec![iter1, iter2], path, 1)?;
///
/// // Compact segments
/// let iter1 = segment1.scan()?;
/// let iter2 = segment2.scan()?;
/// let merged_segment = compact(vec![iter1, iter2], path, 2)?;
/// ```
#[instrument(level = "info", skip(iterators))]
pub fn compact<I>(
    iterators: Vec<I>,
    output_path: PathBuf,
    segment_id: u64,
) -> Result<Arc<Segment>, SegmentError>
where
    I: Iterator<Item = (KeyBytes, ValueBytes)>,
{
    // Ensure the output directory exists
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|e| SegmentError::IoError(e))?;
    }
    fs::create_dir_all(&output_path).map_err(|e| SegmentError::IoError(e))?;

    let merge_iter = MergeIterator::new(iterators);
    let builder = SegmentBuilder::new(output_path)?;
    let seed = random();
    let segment = builder.new_segment(segment_id, seed, 64 * 1024 * 1024)?;

    let mut entry_count = 0u64;

    // Unwrap the Arc to get mutable access
    let segment_mut = Arc::try_unwrap(segment).map_err(|_| {
        SegmentError::CantCreateWriter(crate::segment::BlockType::Key, segment_id)
    })?;

    let seg = segment_mut;

    // Track the last key (namespace + key bytes, without timestamp) to handle duplicates
    let mut last_key: Option<(u64, Bytes)> = None;
    let mut skip_until_new_key = false;

    for (key, value) in merge_iter {
        let current_key = (key.ns(), key.key().clone());

        // Check if this is a new logical key
        let is_new_key = match &last_key {
            None => true,
            Some(prev) => prev != &current_key,
        };

        if is_new_key {
            // New key - check if newest version is a tombstone
            if value.is_tombstone() {
                // Skip this key and all older versions
                skip_until_new_key = true;
                last_key = Some(current_key);
                continue;
            }
            // Not a tombstone - write it and keep older versions
            skip_until_new_key = false;
            last_key = Some(current_key);
        } else if skip_until_new_key {
            // Still processing versions of a key whose newest is a tombstone - skip all
            continue;
        } else if value.is_tombstone() {
            // Older tombstone for a key we're already keeping - just skip it
            continue;
        }

        // Serialize and write
        let key_bytes = key.serialize();
        let val_bytes = value.serialize();
        seg.write(key_bytes.as_ref(), val_bytes.as_ref())?;
        entry_count += 1;
    }

    // Close the segment (writes index and metadata)
    seg.close()?;

    tracing::info!(
        segment_id = segment_id,
        entries = entry_count,
        "Compaction complete"
    );

    Ok(Arc::new(seg))
}

/// Flush a memtable to disk as an L0 segment.
///
/// Unlike `compact()`, this function preserves tombstones because:
/// - Tombstones in the memtable may be deleting keys from older L0 segments
/// - They should only be discarded during major compaction when all versions are merged
///
/// # Arguments
/// * `memtable` - The memtable to flush
/// * `output_path` - Path where the segment files will be created
/// * `segment_id` - ID for the new segment
///
/// # Returns
/// * `Result<Arc<Segment>, SegmentError>` - The newly created L0 segment
///
/// # Example
/// ```rust,ignore
/// let memtable = Memtable::new(1, 1024 * 1024);
/// // ... write data to memtable ...
/// memtable.freeze();
/// let segment = flush_memtable(Arc::new(memtable), path, 1)?;
/// ```
#[instrument(level = "info", skip(memtable))]
pub fn flush_memtable(
    memtable: Arc<Memtable>,
    output_path: PathBuf,
    segment_id: u64,
) -> Result<Arc<Segment>, SegmentError> {
    // Ensure the output directory exists
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|e| SegmentError::IoError(e))?;
    }
    fs::create_dir_all(&output_path).map_err(|e| SegmentError::IoError(e))?;

    let builder = SegmentBuilder::new(output_path)?;
    let seed = random();
    let segment = builder.new_segment(segment_id, seed, 64 * 1024 * 1024)?;

    let mut entry_count = 0u64;

    // Unwrap the Arc to get mutable access
    let segment_mut = Arc::try_unwrap(segment).map_err(|_| {
        SegmentError::CantCreateWriter(crate::segment::BlockType::Key, segment_id)
    })?;

    let seg = segment_mut;

    // Scan all entries in the memtable (including tombstones)
    use std::collections::Bound;
    let iter = memtable.scan(Bound::Unbounded, Bound::Unbounded);

    for (key, value) in iter {
        eprintln!("flush: writing entry {}, tombstone={}", entry_count, value.is_tombstone());
        // NOTE: We do NOT filter tombstones here - they're needed to mask
        // older versions that may exist in L0 segments
        let key_bytes = key.serialize();
        let val_bytes = value.serialize();

        // Write to segment
        seg.write(key_bytes.as_ref(), val_bytes.as_ref())?;
        entry_count += 1;
    }

    // Close the segment (writes index and metadata)
    seg.close()?;

    tracing::info!(
        memtable_id = memtable.id(),
        segment_id = segment_id,
        entries = entry_count,
        "Memtable flush complete"
    );

    Ok(Arc::new(seg))
}

#[cfg(test)]
mod tests {
    use std::collections::Bound;

    use bytes::Bytes;
    use tempfile::tempdir;

    use super::*;
    use crate::{
        hlc::{
            HLC,
            HybridLogicalClock,
        },
        keypair::{
            DEFAULT_NS,
            KeyBytes,
            ValueBytes,
        },
        memtable::Memtable,
    };

    #[test]
    fn test_compact_empty_iterators() {
        let dir = tempdir().unwrap();
        let output_path = dir.path().join("compacted.segment");

        let empty: Vec<Vec<(KeyBytes, ValueBytes)>> = vec![];
        let iters = empty.into_iter().map(IntoIterator::into_iter).collect();

        let result = compact(iters, output_path, 1);
        assert!(result.is_ok(), "Compacting empty iterators should succeed");
    }

    #[test]
    fn test_compact_single_memtable() {
        let dir = tempdir().unwrap();
        let output_path = dir.path().join("compacted.segment");
        let clock = HybridLogicalClock::new();

        let memtable = Memtable::new(1, 1024 * 1024);

        // Insert some data
        for i in 0..10 {
            let key = KeyBytes::new(DEFAULT_NS, Bytes::from(format!("key-{}", i)), clock.time());
            let val = ValueBytes::new(DEFAULT_NS, Bytes::from(format!("value-{}", i)));
            memtable.put(key, val).unwrap();
        }

        let iter = memtable.scan(Bound::Unbounded, Bound::Unbounded);
        let segment = compact(vec![iter], output_path, 1).unwrap();

        assert!(segment.is_read_only());
    }

    #[test]
    fn test_compact_multiple_memtables() {
        let dir = tempdir().unwrap();
        let output_path = dir.path().join("compacted.segment");
        let clock = HybridLogicalClock::new();

        let memtable1 = Memtable::new(1, 1024 * 1024);
        let memtable2 = Memtable::new(2, 1024 * 1024);

        // Insert data into first memtable
        memtable1
            .put(
                KeyBytes::new(DEFAULT_NS, Bytes::from("key1"), clock.time()),
                ValueBytes::new(DEFAULT_NS, Bytes::from("value1_v2")),
            )
            .unwrap();
        memtable1
            .put(
                KeyBytes::new(DEFAULT_NS, Bytes::from("key2"), clock.time()),
                ValueBytes::new(DEFAULT_NS, Bytes::from("value2_v1")),
            )
            .unwrap();

        // Insert data into second memtable
        memtable2
            .put(
                KeyBytes::new(DEFAULT_NS, Bytes::from("key1"), clock.time()),
                ValueBytes::new(DEFAULT_NS, Bytes::from("value1_v3")),
            )
            .unwrap();
        memtable2
            .put(
                KeyBytes::new(DEFAULT_NS, Bytes::from("key3"), clock.time()),
                ValueBytes::new(DEFAULT_NS, Bytes::from("value3_v1")),
            )
            .unwrap();

        let iter1 = memtable1.scan(Bound::Unbounded, Bound::Unbounded);
        let iter2 = memtable2.scan(Bound::Unbounded, Bound::Unbounded);

        let segment = compact(vec![iter1, iter2], output_path, 1).unwrap();

        assert!(segment.is_read_only());
    }

    #[test]
    fn test_compact_preserves_version_order() {
        let dir = tempdir().unwrap();
        let output_path = dir.path().join("compacted.segment");
        let clock = HybridLogicalClock::new();

        let memtable1 = Memtable::new(1, 1024 * 1024);
        let memtable2 = Memtable::new(2, 1024 * 1024);

        // Insert different versions of same key
        let key_name = Bytes::from("versioned-key");

        memtable1
            .put(
                KeyBytes::new(DEFAULT_NS, key_name.clone(), clock.time()),
                ValueBytes::new(DEFAULT_NS, Bytes::from("v1")),
            )
            .unwrap();

        memtable2
            .put(
                KeyBytes::new(DEFAULT_NS, key_name.clone(), clock.time()),
                ValueBytes::new(DEFAULT_NS, Bytes::from("v2")),
            )
            .unwrap();

        let iter1 = memtable1.scan(Bound::Unbounded, Bound::Unbounded);
        let iter2 = memtable2.scan(Bound::Unbounded, Bound::Unbounded);

        let segment = compact(vec![iter1, iter2], output_path, 1).unwrap();

        assert!(segment.is_read_only());
        // The segment should contain both versions in correct order
    }

    #[test]
    fn test_flush_memtable_basic() {
        let dir = tempdir().unwrap();
        let output_path = dir.path().join("flushed.segment");
        let clock = HybridLogicalClock::new();

        let memtable = Arc::new(Memtable::new(1, 1024 * 1024));

        // Insert some data
        for i in 0..10 {
            let key = KeyBytes::new(DEFAULT_NS, Bytes::from(format!("key-{}", i)), clock.time());
            let val = ValueBytes::new(DEFAULT_NS, Bytes::from(format!("value-{}", i)));
            memtable.put(key, val).unwrap();
        }

        // Freeze the memtable
        memtable.freeze();

        // Flush to disk
        let segment = flush_memtable(memtable.clone(), output_path, 1).unwrap();

        assert!(segment.is_read_only());
    }

    #[test]
    fn test_flush_memtable_preserves_tombstones() {
        let dir = tempdir().unwrap();
        let output_path = dir.path().join("flushed.segment");
        let clock = HybridLogicalClock::new();

        let memtable = Arc::new(Memtable::new(1, 1024 * 1024));

        // Insert data and then delete it
        let key = KeyBytes::new(DEFAULT_NS, Bytes::from("key-to-delete"), clock.time());
        let val = ValueBytes::new(DEFAULT_NS, Bytes::from("value"));
        memtable.put(key.clone(), val).unwrap();

        // Now insert a tombstone
        let tombstone_key = KeyBytes::new(DEFAULT_NS, Bytes::from("key-to-delete"), clock.time());
        let tombstone = ValueBytes::new_tombstone(DEFAULT_NS);
        memtable.put(tombstone_key, tombstone).unwrap();

        memtable.freeze();

        // Flush to disk - should preserve tombstones
        let segment = flush_memtable(memtable.clone(), output_path, 1).unwrap();

        assert!(segment.is_read_only());
        // The segment should contain the tombstone (we can't easily verify this here,
        // but the compaction tests verify tombstone filtering works correctly)
    }

    #[test]
    fn test_flush_empty_memtable() {
        let dir = tempdir().unwrap();
        let output_path = dir.path().join("flushed.segment");

        let memtable = Arc::new(Memtable::new(1, 1024 * 1024));
        memtable.freeze();

        let segment = flush_memtable(memtable.clone(), output_path, 1).unwrap();

        assert!(segment.is_read_only());
    }

    #[test]
    fn test_flush_memtable_with_multiple_versions() {
        let dir = tempdir().unwrap();
        let output_path = dir.path().join("flushed.segment");
        let clock = HybridLogicalClock::new();

        let memtable = Arc::new(Memtable::new(1, 1024 * 1024));

        // Insert multiple versions of the same key
        let key_name = Bytes::from("versioned-key");

        memtable
            .put(
                KeyBytes::new(DEFAULT_NS, key_name.clone(), clock.time()),
                ValueBytes::new(DEFAULT_NS, Bytes::from("v1")),
            )
            .unwrap();

        memtable
            .put(
                KeyBytes::new(DEFAULT_NS, key_name.clone(), clock.time()),
                ValueBytes::new(DEFAULT_NS, Bytes::from("v2")),
            )
            .unwrap();

        memtable
            .put(
                KeyBytes::new(DEFAULT_NS, key_name.clone(), clock.time()),
                ValueBytes::new(DEFAULT_NS, Bytes::from("v3")),
            )
            .unwrap();

        memtable.freeze();

        let segment = flush_memtable(memtable.clone(), output_path, 1).unwrap();

        assert!(segment.is_read_only());
        // The segment should contain all three versions
    }

    #[test]
    fn test_flush_reopen_simple() {
        // Test flush_memtable with reopen
        let dir = tempdir().unwrap();
        let output_path = dir.path().join("flush-simple.segment");
        let clock = HybridLogicalClock::new();

        let memtable = Arc::new(Memtable::new(1, 1024 * 1024));
        memtable
            .put(
                KeyBytes::new(DEFAULT_NS, Bytes::from("key1"), clock.time()),
                ValueBytes::new(DEFAULT_NS, Bytes::from("value1")),
            )
            .unwrap();
        memtable
            .put(
                KeyBytes::new(DEFAULT_NS, Bytes::from("key2"), clock.time()),
                ValueBytes::new(DEFAULT_NS, Bytes::from("value2")),
            )
            .unwrap();
        memtable.freeze();

        let segment = flush_memtable(memtable, output_path.clone(), 1).unwrap();
        assert!(segment.is_read_only());

        // Drop and reopen
        drop(segment);
        let builder = SegmentBuilder::new(output_path).unwrap();
        let reopened = builder.open(1).unwrap();
        let reader = reopened.new_reader().unwrap();

        let mut count = 0;
        for result in reader.scan(Bound::Unbounded, Bound::Unbounded) {
            let (_key, _value) = result.unwrap();
            count += 1;
        }
        assert_eq!(count, 2, "Should have 2 entries");
    }

    #[test]
    fn test_compact_reopen_simple() {
        // Simplest possible test: compact 2 entries, then reopen and read them back
        let dir = tempdir().unwrap();
        let output_path = dir.path().join("simple.segment");
        let clock = HybridLogicalClock::new();

        let memtable = Memtable::new(1, 1024 * 1024);
        memtable
            .put(
                KeyBytes::new(DEFAULT_NS, Bytes::from("key1"), clock.time()),
                ValueBytes::new(DEFAULT_NS, Bytes::from("value1")),
            )
            .unwrap();
        memtable
            .put(
                KeyBytes::new(DEFAULT_NS, Bytes::from("key2"), clock.time()),
                ValueBytes::new(DEFAULT_NS, Bytes::from("value2")),
            )
            .unwrap();

        let iter = memtable.scan(Bound::Unbounded, Bound::Unbounded);
        let segment = compact(vec![iter], output_path.clone(), 1).unwrap();
        assert!(segment.is_read_only());

        // Drop and reopen
        drop(segment);
        let builder = SegmentBuilder::new(output_path).unwrap();
        let reopened = builder.open(1).unwrap();
        let reader = reopened.new_reader().unwrap();

        let mut count = 0;
        for result in reader.scan(Bound::Unbounded, Bound::Unbounded) {
            let (_key, _value) = result.unwrap();
            count += 1;
        }
        assert_eq!(count, 2, "Should have 2 entries");
    }

    #[test]
    fn test_compact_filters_tombstones() {
        let dir = tempdir().unwrap();
        let output_path = dir.path().join("compacted.segment");
        let clock = HybridLogicalClock::new();

        let memtable1 = Memtable::new(1, 1024 * 1024);
        let memtable2 = Memtable::new(2, 1024 * 1024);

        // Insert data in first memtable
        memtable1
            .put(
                KeyBytes::new(DEFAULT_NS, Bytes::from("key1"), clock.time()),
                ValueBytes::new(DEFAULT_NS, Bytes::from("value1")),
            )
            .unwrap();

        // Insert a tombstone in second memtable
        memtable2
            .put(
                KeyBytes::new(DEFAULT_NS, Bytes::from("key1"), clock.time()),
                ValueBytes::new_tombstone(DEFAULT_NS),
            )
            .unwrap();

        // Add a regular key too
        memtable2
            .put(
                KeyBytes::new(DEFAULT_NS, Bytes::from("key2"), clock.time()),
                ValueBytes::new(DEFAULT_NS, Bytes::from("value2")),
            )
            .unwrap();

        let iter1 = memtable1.scan(Bound::Unbounded, Bound::Unbounded);
        let iter2 = memtable2.scan(Bound::Unbounded, Bound::Unbounded);

        // Compact should filter out tombstones
        let output_path_clone = output_path.clone();
        let segment = compact(vec![iter1, iter2], output_path, 1).unwrap();

        assert!(segment.is_read_only());

        // Drop the segment to ensure files are closed
        drop(segment);

        // Reopen the segment to read from it
        let builder = SegmentBuilder::new(output_path_clone).unwrap();
        let reopened = builder.open(1).unwrap();
        let reader = reopened.new_reader().unwrap();
        let mut count = 0;
        for result in reader.scan(Bound::Unbounded, Bound::Unbounded) {
            let (_key, value) = result.unwrap();
            // None of the values should be tombstones
            assert!(!value.is_tombstone(), "Tombstones should be filtered during compaction");
            count += 1;
        }

        // We should only have key2 (key1's tombstone should have removed all versions)
        assert_eq!(count, 1, "Should only have one non-tombstone entry after compaction");
    }

    #[test]
    fn test_flush_vs_compact_tombstone_handling() {
        let dir = tempdir().unwrap();
        let clock = HybridLogicalClock::new();

        // Create a memtable with a tombstone
        let memtable = Arc::new(Memtable::new(1, 1024 * 1024));
        memtable
            .put(
                KeyBytes::new(DEFAULT_NS, Bytes::from("deleted-key"), clock.time()),
                ValueBytes::new_tombstone(DEFAULT_NS),
            )
            .unwrap();
        memtable.freeze();

        // Flush should preserve tombstones
        let flush_path = dir.path().join("flushed.segment");
        let flushed_segment = flush_memtable(memtable.clone(), flush_path.clone(), 1).unwrap();

        // Drop the segment to ensure files are closed
        drop(flushed_segment);

        // Reopen the flushed segment to read from it
        let builder = SegmentBuilder::new(flush_path).unwrap();
        let reopened_flush = builder.open(1).unwrap();
        let flush_reader = reopened_flush.new_reader().unwrap();
        let mut found_tombstone = false;
        let mut count = 0;
        for result in flush_reader.scan(Bound::Unbounded, Bound::Unbounded) {
            let (_key, value) = result.unwrap();
            eprintln!("Scanned entry {}: tombstone={}", count, value.is_tombstone());
            if value.is_tombstone() {
                found_tombstone = true;
            }
            count += 1;
        }
        eprintln!("Total scanned: {}, found_tombstone={}", count, found_tombstone);
        assert!(found_tombstone, "Flush should preserve tombstones for L0");

        // Compact should filter tombstones
        let compact_path = dir.path().join("compacted.segment");
        let iter = memtable.scan(Bound::Unbounded, Bound::Unbounded);
        let compacted_segment = compact(vec![iter], compact_path.clone(), 2).unwrap();

        // Drop the segment to ensure files are closed
        drop(compacted_segment);

        // Reopen the compacted segment to read from it
        let builder2 = SegmentBuilder::new(compact_path).unwrap();
        let reopened_compact = builder2.open(2).unwrap();
        let compact_reader = reopened_compact.new_reader().unwrap();
        let mut found_tombstone_in_compact = false;
        for result in compact_reader.scan(Bound::Unbounded, Bound::Unbounded) {
            let (_key, value) = result.unwrap();
            if value.is_tombstone() {
                found_tombstone_in_compact = true;
            }
        }
        assert!(!found_tombstone_in_compact, "Compact should filter tombstones");
    }
}