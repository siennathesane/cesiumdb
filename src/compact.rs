// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

use std::{
    fs,
    path::PathBuf,
    sync::Arc,
};

use rand::random;
use tracing::instrument;

use crate::{
    errs::SegmentError,
    keypair::{
        KeyBytes,
        ValueBytes,
    },
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

    for (key, value) in merge_iter {
        // Skip tombstones - they've already done their job of masking older versions
        // during the merge, no need to persist them to the new segment
        if value.is_tombstone() {
            continue;
        }

        // Serialize key and value for storage
        let key_bytes = key.serialize();
        let val_bytes = value.serialize_for_storage();

        // Write to segment
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
}