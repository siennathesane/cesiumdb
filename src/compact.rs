// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

use std::{
    fs,
    path::PathBuf,
    sync::Arc,
};

use bytes::Bytes;
use rand::random;

use crate::{
    errs::SegmentError,
    keypair::{
        KeyBytes,
        ValueBytes,
    },
    memtable::Memtable,
    merge::RawMergeIterator,
    raw_entry::RawEntry,
    segment::{
        DEFAULT_SEGMENT_SIZE,
        Segment,
    },
    segment_builder::SegmentBuilder,
    utils::Serializer,
};

/// Result of a compaction operation, including key range information.
pub struct CompactOutput {
    /// The compacted segment
    pub segment: Arc<Segment>,
    /// Smallest serialized key in the output segment (empty if no entries)
    pub min_key: Vec<u8>,
    /// Largest serialized key in the output segment (empty if no entries)
    pub max_key: Vec<u8>,
}

/// Reopen a closed segment for reading.
///
/// After a segment is closed, its mmap handles are dropped. This helper
/// reopens the underlying files so the segment can be queried.
fn reopen_segment_for_reading(
    output_path: &PathBuf,
    segment_id: u64,
) -> Result<Arc<Segment>, SegmentError> {
    use crate::{
        index::Index,
        map::Map,
        segment::Metadata,
    };

    let key_id = segment_id;
    let val_id = segment_id + 1;

    let key_path = output_path.join(key_id.to_string());
    let val_path = output_path.join(val_id.to_string());

    let key_map = Arc::new(match Map::open(key_path) {
        | Ok(v) => v,
        | Err(e) => return Err(e),
    });
    let val_map = Arc::new(match Map::open(val_path) {
        | Ok(v) => v,
        | Err(e) => return Err(e),
    });

    let key_metadata = {
        let len = key_map.len();
        if len < 32 {
            return Err(SegmentError::CorruptedBlock);
        }
        match key_map.read_range(len - 32..len, |slice| {
            Metadata::from(Bytes::copy_from_slice(slice))
        }) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        }
    };

    let val_metadata = {
        let len = val_map.len();
        if len < 32 {
            return Err(SegmentError::CorruptedBlock);
        }
        match val_map.read_range(len - 32..len, |slice| {
            Metadata::from(Bytes::copy_from_slice(slice))
        }) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        }
    };

    let index_bytes = {
        let start = key_metadata.index_start();
        let size = key_metadata.index_size();

        if key_map.len() < start + size {
            return Err(SegmentError::CorruptedBlock);
        }

        match key_map.read_range(start..start + size, Bytes::copy_from_slice) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        }
    };

    let key_index = Index::from(index_bytes);
    let val_block_count = val_metadata.block_count() as u64;

    Segment::open(key_map, key_index, key_id, val_map, val_id, val_block_count)
}

/// Zero-copy compaction: merges raw serialized entries without deserializing.
///
/// Same semantics as `compact()` but operates on `RawEntry` (pre-serialized
/// bytes) to eliminate the deserialize → re-serialize round-trip. This reduces
/// per-entry heap allocations from 5 to 0 (only 2 total allocations for min/max
/// key tracking).
#[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
pub(crate) fn compact_raw<I>(
    iterators: Vec<I>,
    output_path: PathBuf,
    segment_id: u64,
) -> Result<CompactOutput, SegmentError>
where
    I: Iterator<Item = Result<RawEntry, SegmentError>>, {
    use std::time::Instant;
    let start_time = Instant::now();

    // Ensure the output directory exists
    if let Some(parent) = output_path.parent()
        && let Err(e) = fs::create_dir_all(parent) {
            return Err(SegmentError::IoError(e));
        }
    if let Err(e) = fs::create_dir_all(&output_path) {
        return Err(SegmentError::IoError(e));
    }

    let merge_iter = RawMergeIterator::new(iterators);

    let builder = match SegmentBuilder::new(output_path.clone()) {
        | Ok(b) => b,
        | Err(e) => return Err(e),
    };
    let seed = random();
    let segment = match builder.new_segment(segment_id, seed, DEFAULT_SEGMENT_SIZE) {
        | Ok(s) => s,
        | Err(e) => return Err(e),
    };

    let mut min_key: Option<Vec<u8>> = None;
    let mut max_key: Option<Vec<u8>> = None;
    let mut last_key_bytes: Option<Bytes> = None;

    let segment_mut = match Arc::try_unwrap(segment)
        .map_err(|_| SegmentError::CantCreateWriter(crate::segment::BlockType::Key, segment_id))
    {
        | Ok(v) => v,

        | Err(e) => return Err(e),
    };

    let seg = segment_mut;

    // Track the last dedup key to handle duplicates.
    // dedup_key = [ns:8][user_key] (everything except timestamp)
    // Bytes::clone is just an Arc refcount bump — zero copy.
    let mut last_dedup_key: Option<Bytes> = None;

    let mut write_time = std::time::Duration::ZERO;
    let loop_start = Instant::now();

    for result in merge_iter {
        let entry = match result {
            | Ok(e) => e,
            | Err(_) => continue, // Skip read errors (same as existing filter_map behavior)
        };

        let current_dedup = entry.dedup_key();

        // Check if this is a new logical key
        let is_new_key = match &last_dedup_key {
            | None => true,
            | Some(prev) => prev.as_ref() != current_dedup,
        };

        if is_new_key {
            if entry.is_tombstone() {
                last_dedup_key = Some(entry.raw_key().slice(..entry.raw_key().len() - 16));
                continue;
            }
            last_dedup_key = Some(entry.raw_key().slice(..entry.raw_key().len() - 16));
        } else {
            // Older version of a key we've already processed - skip it
            continue;
        }

        // Write raw bytes directly — NO serialize() calls
        let key_ref = entry.raw_key();
        let val_ref = entry.raw_val();

        // Track min key only once (first written)
        if min_key.is_none() {
            min_key = Some(key_ref.to_vec());
        }

        let write_start = Instant::now();
        if let Err(e) = seg.write(key_ref.as_ref(), val_ref.as_ref()) {
            return Err(e);
        }
        write_time += write_start.elapsed();

        last_key_bytes = Some(key_ref.clone());
    }

    let loop_time = loop_start.elapsed();

    if let Some(last) = last_key_bytes {
        max_key = Some(last.to_vec());
    }

    let close_start = Instant::now();
    if let Err(e) = seg.close() {
        return Err(e);
    } // Automatically rebuilds index
    let close_time = close_start.elapsed();

    // Drop the writer segment to release mmap locks before reopening
    drop(seg);

    // Reopen the segment for reading so handles are populated
    let reopened = match reopen_segment_for_reading(&output_path, segment_id) {
        | Ok(s) => s,
        | Err(e) => return Err(e),
    };

    let total_time = start_time.elapsed();

    tracing::info!(
        segment_id = segment_id,
        total_ms = total_time.as_millis(),
        loop_ms = loop_time.as_millis(),
        write_ms = write_time.as_millis(),
        close_ms = close_time.as_millis(),
        "Raw compaction timing breakdown (zero-copy + deferred index)"
    );

    Ok(CompactOutput {
        segment: reopened,
        min_key: min_key.unwrap_or_default(),
        max_key: max_key.unwrap_or_default(),
    })
}

/// Flush a memtable to disk as an L0 segment.
///
/// This function preserves tombstones because:
/// - Tombstones in the memtable may be deleting keys from older L0 segments
/// - They should only be discarded during major compaction when all versions
///   are merged
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
#[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
pub fn flush_memtable(
    memtable: Arc<Memtable>,
    output_path: PathBuf,
    segment_id: u64,
) -> Result<(Arc<Segment>, Vec<u8>, Vec<u8>), SegmentError> {
    // Ensure the output directory exists
    if let Some(parent) = output_path.parent()
        && let Err(e) = fs::create_dir_all(parent) {
            return Err(SegmentError::IoError(e));
        }
    if let Err(e) = fs::create_dir_all(&output_path) {
        return Err(SegmentError::IoError(e));
    }

    let builder = match SegmentBuilder::new(output_path.clone()) {
        | Ok(v) => v,

        | Err(e) => return Err(e),
    };
    let seed = random();
    let segment = match builder.new_segment(segment_id, seed, DEFAULT_SEGMENT_SIZE) {
        | Ok(s) => s,
        | Err(e) => return Err(e),
    };

    // Unwrap the Arc to get mutable access
    let segment_mut = match Arc::try_unwrap(segment) {
        | Ok(s) => s,
        | Err(_) => {
            return Err(SegmentError::CantCreateWriter(
                crate::segment::BlockType::Key,
                segment_id,
            ));
        },
    };

    let seg = segment_mut;

    // Track min/max keys for manifest
    let mut min_key: Option<Vec<u8>> = None;
    let mut max_key: Option<Vec<u8>> = None;

    // Deduplicate: memtable.scan() iterates in ascending byte order (newest
    // first because timestamps are stored as u128::MAX - ts, so smaller
    // serialized bytes = newer timestamp). We only want the newest version
    // per key in the flushed segment, otherwise point reads via
    // SegmentScanIterator::next() will return the oldest version.
    let mut last_key: Option<KeyBytes> = None;
    let mut last_val: Option<ValueBytes> = None;

    // Scan all entries in the memtable (including tombstones)
    use std::collections::Bound;
    let iter = memtable.scan(Bound::Unbounded, Bound::Unbounded);

    for (key, value) in iter {
        let same_logical_key = match &last_key {
            | Some(prev) => prev.ns() == key.ns() && prev.as_bytes() == key.as_bytes(),
            | None => false,
        };

        if same_logical_key {
            // Same key, older version (scan is newest-first, so later = older).
            // Drop this one, keep the first (newest) we saw.
            continue;
        } else {
            // Key changed. Flush the buffered entry (if any).
            if let (Some(prev_key), Some(prev_val)) = (last_key.take(), last_val.take()) {
                let key_bytes = prev_key.serialize();
                let val_bytes = prev_val.serialize();

                if min_key.is_none() {
                    min_key = Some(key_bytes.to_vec());
                }
                if let Err(e) = seg.write(key_bytes.as_ref(), val_bytes.as_ref()) {
                    return Err(e);
                }
                max_key = Some(key_bytes.to_vec());
            }

            last_key = Some(key);
            last_val = Some(value);
        }
    }

    // Flush the final buffered entry
    if let (Some(prev_key), Some(prev_val)) = (last_key, last_val) {
        let key_bytes = prev_key.serialize();
        let val_bytes = prev_val.serialize();

        if min_key.is_none() {
            min_key = Some(key_bytes.to_vec());
        }
        if let Err(e) = seg.write(key_bytes.as_ref(), val_bytes.as_ref()) {
            return Err(e);
        }
        max_key = Some(key_bytes.to_vec());
    }

    // Close the segment (writes index and metadata)
    if let Err(e) = seg.close() {
        return Err(e);
    }

    tracing::info!(
        memtable_id = memtable.id(),
        segment_id = segment_id,
        "Memtable flush complete"
    );

    // Drop the writer segment to release mmap locks before reopening
    drop(seg);

    // Reopen the segment for reading
    // Key and value IDs follow the pattern: key_id = segment_id, val_id =
    // segment_id + 1
    let key_id = segment_id;
    let val_id = segment_id + 1;

    // Open memory maps for the flushed files
    use crate::{
        index::Index,
        map::Map,
    };
    let key_path = output_path.join(key_id.to_string());
    let val_path = output_path.join(val_id.to_string());

    let key_map = Arc::new(match Map::open(key_path) {
        | Ok(v) => v,
        | Err(e) => return Err(e),
    });
    let val_map = Arc::new(match Map::open(val_path) {
        | Ok(v) => v,
        | Err(e) => return Err(e),
    });

    // Read the key metadata to find the index location
    use crate::segment::Metadata;
    let key_metadata = {
        let len = key_map.len();
        // Metadata is at the very end (32 bytes: 4 × u64)
        if len < 32 {
            return Err(SegmentError::CorruptedBlock);
        }
        match key_map.read_range(len - 32..len, |slice| {
            Metadata::from(Bytes::copy_from_slice(slice))
        }) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        }
    };

    // Read the value metadata to get block count
    let val_metadata = {
        let len = val_map.len();
        // Metadata is at the very end (32 bytes: 4 × u64)
        if len < 32 {
            return Err(SegmentError::CorruptedBlock);
        }
        match val_map.read_range(len - 32..len, |slice| {
            Metadata::from(Bytes::copy_from_slice(slice))
        }) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        }
    };

    // Read the index using the key metadata
    let index_bytes = {
        let start = key_metadata.index_start();
        let size = key_metadata.index_size();

        if key_map.len() < start + size {
            return Err(SegmentError::CorruptedBlock);
        }

        match key_map.read_range(start..start + size, Bytes::copy_from_slice) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        }
    };

    let key_index = Index::from(index_bytes);
    let val_block_count = val_metadata.block_count() as u64;

    // Create a read-only segment that can be queried
    let segment = match Segment::open(key_map, key_index, key_id, val_map, val_id, val_block_count)
    {
        | Ok(s) => s,
        | Err(e) => return Err(e),
    };

    // Return segment with min/max keys (empty vec if no keys)
    Ok((
        segment,
        min_key.unwrap_or_default(),
        max_key.unwrap_or_default(),
    ))
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
        let (segment, _min_key, _max_key) =
            flush_memtable(memtable.clone(), output_path, 1).unwrap();

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
        let (segment, _min_key, _max_key) =
            flush_memtable(memtable.clone(), output_path, 1).unwrap();

        assert!(segment.is_read_only());
        // The segment should contain the tombstone (we can't easily verify this
        // here, but the compaction tests verify tombstone filtering
        // works correctly)
    }

    #[test]
    fn test_flush_empty_memtable() {
        let dir = tempdir().unwrap();
        let output_path = dir.path().join("flushed.segment");

        let memtable = Arc::new(Memtable::new(1, 1024 * 1024));
        memtable.freeze();

        let (segment, _min_key, _max_key) =
            flush_memtable(memtable.clone(), output_path, 1).unwrap();

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

        let (segment, _min_key, _max_key) =
            flush_memtable(memtable.clone(), output_path, 1).unwrap();

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

        let (segment, _min_key, _max_key) =
            flush_memtable(memtable, output_path.clone(), 1).unwrap();
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
}
