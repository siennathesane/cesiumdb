// Copyright (c) Sienna Meridian Satterwhite
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

//! Scanning and iteration types for CesiumDB.
//!
//! Provides [`DbScanIterator`] for range scans and [`OwnedSegmentIterator`]
//! for internal segment-level iteration, plus [`ReadAmpStats`] for read
//! amplification instrumentation.

use bytes::Bytes;

use crate::{
    keypair::{
        KeyBytes,
        ValueBytes,
    },
    merge::MergeIterator,
    segment_iterator::SegmentScanIterator,
    segment_reader::SegmentReader,
    utils::Serializer,
};

/// Wrapper that owns a SegmentReader and its iterator together.
///
/// This solves the lifetime issue where SegmentScanIterator borrows from
/// SegmentReader by having the iterator own the reader.
pub struct OwnedSegmentIterator {
    // reader is None after the iterator is created (taken by scan)
    pub(crate) reader: Option<SegmentReader>,
    // Store serialized KeyBytes bounds (with namespace + timestamp)
    pub(crate) lower: std::ops::Bound<Bytes>,
    pub(crate) upper: std::ops::Bound<Bytes>,
    pub(crate) inner: Option<SegmentScanIterator>,
}

impl OwnedSegmentIterator {
    /// Create a new owned segment iterator.
    ///
    /// Takes KeyBytes bounds (already serialized with namespace + timestamp).
    pub(crate) fn new(
        reader: SegmentReader,
        lower: std::ops::Bound<KeyBytes>,
        upper: std::ops::Bound<KeyBytes>,
    ) -> Self {
        use std::ops::Bound;

        // Serialize KeyBytes bounds to raw bytes
        let lower_bound = match lower {
            | Bound::Included(k) => Bound::Included(k.serialize()),
            | Bound::Excluded(k) => Bound::Excluded(k.serialize()),
            | Bound::Unbounded => Bound::Unbounded,
        };
        let upper_bound = match upper {
            | Bound::Included(k) => Bound::Included(k.serialize()),
            | Bound::Excluded(k) => Bound::Excluded(k.serialize()),
            | Bound::Unbounded => Bound::Unbounded,
        };

        Self {
            reader: Some(reader),
            lower: lower_bound,
            upper: upper_bound,
            inner: None,
        }
    }
}

impl Iterator for OwnedSegmentIterator {
    type Item = (KeyBytes, ValueBytes);

    fn next(&mut self) -> Option<Self::Item> {
        // Create iterator on first call by taking ownership of the reader
        if self.inner.is_none() {
            let reader = self.reader.take()?;
            let lower_ref = match &self.lower {
                | std::ops::Bound::Included(b) => std::ops::Bound::Included(&b[..]),
                | std::ops::Bound::Excluded(b) => std::ops::Bound::Excluded(&b[..]),
                | std::ops::Bound::Unbounded => std::ops::Bound::Unbounded,
            };
            let upper_ref = match &self.upper {
                | std::ops::Bound::Included(b) => std::ops::Bound::Included(&b[..]),
                | std::ops::Bound::Excluded(b) => std::ops::Bound::Excluded(&b[..]),
                | std::ops::Bound::Unbounded => std::ops::Bound::Unbounded,
            };
            let iter = reader.scan(lower_ref, upper_ref);
            self.inner = Some(iter);
        }

        // Iterate, skipping errors
        loop {
            match self.inner.as_mut()?.next()? {
                | Ok(pair) => return Some(pair),
                | Err(_) => continue, // Skip corrupt entries
            }
        }
    }
}

/// Iterator over a range of key-value pairs from the database.
///
/// This iterator merges results from memtables and all LSM levels,
/// automatically handling deduplication (newer versions shadow older),
/// tombstone filtering, and maintaining sorted order.
pub struct DbScanIterator {
    pub(crate) inner: MergeIterator<Box<dyn Iterator<Item = (KeyBytes, ValueBytes)> + Send>>,
    pub(crate) last_key: Option<(u64, Bytes)>, // (namespace, key) for deduplication
}

impl Iterator for DbScanIterator {
    type Item = (Bytes, Bytes);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.inner.next() {
                | Some((key, value)) => {
                    let current_ns = key.ns();
                    let current_key = key.as_bytes();

                    // Check if this is a duplicate key (different timestamp of same key)
                    if let Some((last_ns, ref last_key_bytes)) = self.last_key
                        && last_ns == current_ns && last_key_bytes == &current_key {
                            // Skip older version of the same key
                            continue;
                        }

                    // Update last seen key
                    self.last_key = Some((current_ns, current_key.clone()));

                    // Filter out tombstones
                    if value.is_tombstone() {
                        continue;
                    }

                    // Convert KeyBytes/ValueBytes to Bytes for public API
                    return Some((current_key, value.as_bytes()));
                },
                | None => return None,
            }
        }
    }
}

/// Read amplification statistics for point lookups.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReadAmpStats {
    /// Total number of db.get operations.
    pub total_gets: u64,
    /// Number of L0 segments checked across all gets.
    pub l0_segments_checked: u64,
    /// Number of L1-L7 segments checked across all gets.
    pub ln_segments_checked: u64,
}
