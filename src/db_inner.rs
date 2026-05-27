// Copyright (c) Sienna Meridian Satterwhite
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

//! Internal database implementation details.
//!
//! [`DbInner`] holds the actual state and implements the core get, scan,
//! batch, and sync logic behind the public [`Db`] API.

use std::{
    sync::{
        Arc,
        atomic::{
            AtomicU64,
            Ordering,
        },
    },
    thread,
    time::Duration,
};

use bytes::Bytes;
use parking_lot::{
    Mutex,
    RwLock,
};

use crate::{
    Batch::{
        Delete,
        DeleteNs,
        Put,
        PutNs,
    },
    CesiumError,
    CesiumError::MemtableError,
    DbScanIterator,
    OwnedSegmentIterator,
    ReadAmpStats,
    VersionStats,
    db_options::Batch,
    errs::MemtableError as MtError,
    keypair::{
        DEFAULT_NS,
        KeyBytes,
        ValueBytes,
    },
    memtable::Memtable,
    merge,
    state::DbStorageState,

    version::VersionManager,
};

/// Internal database state and operations.
#[repr(C)]
pub(crate) struct DbInner {
    pub(crate) state: Mutex<DbStorageState>,
    /// Cached current memtable to avoid state lock on hot write/get paths.
    /// Updated whenever `new_memtable()` is called under state lock.
    pub(crate) curr_memtable: RwLock<Arc<Memtable>>,
    /// Cached frozen-memtables Arc to avoid state lock on the read path.
    /// Points to the same `Mutex<Vec<Arc<Memtable>>>` held by `DbStorageState`.
    pub(crate) frozen_memtables: Arc<Mutex<Vec<Arc<Memtable>>>>,
    /// Version manager for checking L0 size without state lock
    pub(crate) version_manager: Arc<VersionManager>,
    /// Warm thread pool for parallel LSM reads across levels
    pub(crate) read_pool: rayon::ThreadPool,
    /// Cumulative read amplification counters
    pub(crate) total_gets: AtomicU64,
    pub(crate) l0_reads: AtomicU64,
    pub(crate) ln_reads: AtomicU64,
}

impl DbInner {
    pub(crate) fn get(&self, key: KeyBytes) -> Result<Option<ValueBytes>, CesiumError> {
        // Track that we did a get (for read amplification instrumentation)
        self.total_gets.fetch_add(1, Ordering::Relaxed);

        // 1. Check current memtable (hottest data) without state lock
        {
            let mtable = self.curr_memtable.read().clone();
            if let Some(val) = mtable.get(&key) {
                // Return None for tombstones
                if val.is_tombstone() {
                    return Ok(None);
                }
                return Ok(Some(val));
            }
        }

        // 2. Check frozen memtables (newest to oldest) without state lock
        {
            let frozen = self.frozen_memtables.lock();
            for memtable in frozen.iter().rev() {
                if let Some(val) = memtable.get(&key) {
                    if val.is_tombstone() {
                        return Ok(None);
                    }
                    return Ok(Some(val));
                }
            }
        }

        // 3. Check L0-L7 via VersionManager (no state lock)
        {
            let version = self.version_manager.current();

            // Check L0 (newest to oldest - reverse chronological)
            // L0 must be checked sequentially because newer segments override older ones
            // We need to search by key prefix (ns + key) to find any version
            // Since timestamps are stored as (u128::MAX - ts), newest=0, oldest=u128::MAX

            // Prepare key without timestamp for bloom filter checks.
            // Use SmallVec to avoid heap allocation for typical key sizes.
            let mut key_for_bloom = smallvec::SmallVec::<[u8; 64]>::with_capacity(8 + key.as_bytes().len());
            key_for_bloom.extend_from_slice(&key.ns().to_le_bytes());
            key_for_bloom.extend_from_slice(&key.as_bytes());

            // Check L0 segments in reverse chronological order (newest first)
            for segment in version.l0.iter().rev() {
                // Fast bloom filter check without creating a SegmentReader.
                if !segment.may_contain(&key_for_bloom) {
                    continue;
                }

                self.l0_reads.fetch_add(1, Ordering::Relaxed);

                let reader = match segment.reader_cached() {
                    | Ok(r) => r,
                    | Err(e) => return Err(CesiumError::SegmentError(e)),
                };

                // Fast point lookup – touches at most one key block.
                // Bloom was already checked, so use the fast path.
                match reader.get_latest_fast(&key_for_bloom) {
                    | Ok(Some(val_bytes)) => {
                        let val = ValueBytes::deserialize(val_bytes);
                        if val.is_tombstone() {
                            return Ok(None);
                        }
                        return Ok(Some(val));
                    },
                    | Ok(None) => {},
                    | Err(e) => return Err(CesiumError::SegmentError(e)),
                }
            }

            // Check L1-L7 sequentially from newest level to oldest.
            for level in &version.levels {
                if level.strategy.allows_overlaps() {
                    // Tiered / universal levels may have overlapping ranges.
                    // Fall back to scanning all segments with bloom checks.
                    for segment in &level.segments {
                        if !segment.may_contain(&key_for_bloom) {
                            continue;
                        }

                        self.ln_reads.fetch_add(1, Ordering::Relaxed);

                        let reader = match segment.reader_cached() {
                            | Ok(r) => r,
                            | Err(e) => return Err(CesiumError::SegmentError(e)),
                        };

                        match reader.get_latest_fast(&key_for_bloom) {
                            | Ok(Some(val_bytes)) => {
                                let val = ValueBytes::deserialize(val_bytes);
                                if val.is_tombstone() {
                                    return Ok(None);
                                }
                                return Ok(Some(val));
                            },
                            | Ok(None) => {},
                            | Err(e) => return Err(CesiumError::SegmentError(e)),
                        }
                    }
                } else {
                    // Leveled levels have sorted, non-overlapping ranges.
                    // Use binary search to touch at most one segment per level.
                    if let Some(segment_id) = level.find_segment_for_key_binary(&key_for_bloom) {
                        let segment = match level.segments.iter().find(|s| s.id() == segment_id) {
                            | Some(s) => s,
                            | None => continue,
                        };

                        if !segment.may_contain(&key_for_bloom) {
                            continue;
                        }

                        self.ln_reads.fetch_add(1, Ordering::Relaxed);

                        let reader = match segment.reader_cached() {
                            | Ok(r) => r,
                            | Err(e) => return Err(CesiumError::SegmentError(e)),
                        };

                        match reader.get_latest_fast(&key_for_bloom) {
                            | Ok(Some(val_bytes)) => {
                                let val = ValueBytes::deserialize(val_bytes);
                                if val.is_tombstone() {
                                    return Ok(None);
                                }
                                return Ok(Some(val));
                            },
                            | Ok(None) => {},
                            | Err(e) => return Err(CesiumError::SegmentError(e)),
                        }
                    }
                }
            }
        }

        // 4. Not found anywhere
        Ok(None)
    }

    pub(crate) fn scan(
        &self,
        ns: u64,
        lower: std::ops::Bound<&[u8]>,
        upper: std::ops::Bound<&[u8]>,
    ) -> Result<DbScanIterator, CesiumError> {
        use std::ops::Bound;

        // Convert bounds to KeyBytes format (with namespace and timestamp)
        // For namespace isolation, we need to ensure we only scan within the given
        // namespace
        //
        // IMPORTANT: KeyBytes serializes timestamps as `u128::MAX - ts`, so:
        // - ts=0 (newest) serializes to MAX (sorts LAST in byte order)
        // - ts=MAX (oldest) serializes to 0 (sorts FIRST in byte order)
        // Therefore, to scan forward seeing newest versions first, we need ts=MAX in
        // lower bound.
        let lower_key = match lower {
            | Bound::Included(k) => {
                // Start with oldest version (ts=MAX serializes to 0, sorts first)
                Bound::Included(KeyBytes::new(ns, Bytes::copy_from_slice(k), u128::MAX))
            },
            | Bound::Excluded(k) => {
                // Exclude oldest version
                Bound::Excluded(KeyBytes::new(ns, Bytes::copy_from_slice(k), u128::MAX))
            },
            | Bound::Unbounded => {
                // Start from the beginning of this namespace
                Bound::Included(KeyBytes::new(ns, Bytes::new(), u128::MAX))
            },
        };

        let upper_key = match upper {
            | Bound::Included(k) => {
                // Include newest version (ts=0 serializes to MAX, sorts last)
                Bound::Included(KeyBytes::new(ns, Bytes::copy_from_slice(k), 0))
            },
            | Bound::Excluded(k) => {
                // Exclude all versions (ts=MAX serializes to 0, sorts first, so excluded bound
                // excludes all)
                Bound::Excluded(KeyBytes::new(ns, Bytes::copy_from_slice(k), u128::MAX))
            },
            | Bound::Unbounded => {
                // End at the last possible key in this namespace
                // Use next namespace's first key as excluded upper bound
                Bound::Excluded(KeyBytes::new(ns + 1, Bytes::new(), u128::MAX))
            },
        };

        let mut iters: Vec<Box<dyn Iterator<Item = (KeyBytes, ValueBytes)> + Send>> = Vec::new();

        // 1. Add current memtable iterator (without state lock)
        {
            let mtable = self.curr_memtable.read().clone();
            let memtable_iter = mtable.scan(lower_key.clone(), upper_key.clone());
            iters
                .push(Box::new(memtable_iter)
                    as Box<dyn Iterator<Item = (KeyBytes, ValueBytes)> + Send>);
        }

        // 2. Add frozen memtables without state lock
        {
            let frozen = self.frozen_memtables.lock();
            for memtable in frozen.iter().rev() {
                let iter = memtable.scan(lower_key.clone(), upper_key.clone());
                iters.push(Box::new(iter) as Box<dyn Iterator<Item = (KeyBytes, ValueBytes)> + Send>);
            }
        }

        // 3. Add L0-L7 segment iterators under state lock
        {
            let version = self.version_manager.current();

            // Add L0 segments (can overlap, so all must be scanned)
            for segment in &version.l0 {
                let reader = match segment.reader() {
                    | Ok(r) => r,
                    | Err(e) => return Err(CesiumError::SegmentError(e)),
                };
                let owned_iter =
                    OwnedSegmentIterator::new(reader, lower_key.clone(), upper_key.clone());
                iters
                    .push(Box::new(owned_iter)
                        as Box<dyn Iterator<Item = (KeyBytes, ValueBytes)> + Send>);
            }

            // Add segments from L1-L7
            for level in &version.levels {
                for segment in &level.segments {
                    let reader = match segment.reader() {
                        | Ok(r) => r,
                        | Err(e) => return Err(CesiumError::SegmentError(e)),
                    };
                    let owned_iter =
                        OwnedSegmentIterator::new(reader, lower_key.clone(), upper_key.clone());
                    iters.push(Box::new(owned_iter)
                        as Box<dyn Iterator<Item = (KeyBytes, ValueBytes)> + Send>);
                }
            }
        }

        // Create merge iterator
        let merge_iter = merge::MergeIterator::new(iters);

        Ok(DbScanIterator {
            inner: merge_iter,
            last_key: None,
        })
    }

    /// Block until the compaction manager indicates writes may proceed.
    fn stall_if_needed(&self) {
        loop {
            let should_stall = {
                let guard = self.state.lock();
                guard.should_stall_writes()
            };
            if !should_stall {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    /// Helper: write the remaining portion of a batch, swapping memtables as
    /// needed.  Deduplicates the retry logic used by both the partial-write
    /// and the DataExceedsMaximum paths.
    fn write_batch_with_swap(
        &self,
        batch: &[(KeyBytes, ValueBytes)],
        offset: &mut usize,
        last_attempted: &mut Arc<Memtable>,
    ) -> Result<(), CesiumError> {
        while *offset < batch.len() {
            self.wait_for_frozen_capacity();

            let new_mtable = {
                let mut guard = self.state.lock();
                let current = guard.current_memtable();
                if Arc::ptr_eq(last_attempted, &current) {
                    guard.new_memtable();
                }
                let new = guard.current_memtable();
                *self.curr_memtable.write() = new.clone();
                new
            };

            match new_mtable.put_batch(&batch[*offset..]) {
                | Ok(w) => {
                    *offset += w;
                    if *offset >= batch.len() {
                        return Ok(());
                    }
                    *last_attempted = new_mtable;
                },
                | Err(e) => {
                    if matches!(e, MtError::MemtableIsFrozen | MtError::DataExceedsMaximum) {
                        *last_attempted = new_mtable;
                        continue;
                    }
                    return Err(MemtableError(e));
                },
            }
        }
        Ok(())
    }

    pub(crate) fn batch<K: AsRef<[u8]>, V: AsRef<[u8]>>(
        &self,
        ops: &[Batch<K, V>],
    ) -> Result<(), CesiumError> {
        self.stall_if_needed();

        let mut _batch = Vec::with_capacity(ops.len());
        for b in ops.iter() {
            match b {
                | PutNs(ns, k, v, ts) => {
                    _batch.push((
                        KeyBytes::new(*ns, Bytes::copy_from_slice(k.as_ref()), *ts),
                        ValueBytes::new(*ns, Bytes::copy_from_slice(v.as_ref())),
                    ));
                },
                | DeleteNs(ns, k, ts) => {
                    _batch.push((
                        KeyBytes::new(*ns, Bytes::copy_from_slice(k.as_ref()), *ts),
                        ValueBytes::new_tombstone(*ns),
                    ));
                },
                | Put(k, v, ts) => {
                    _batch.push((
                        KeyBytes::new(DEFAULT_NS, Bytes::copy_from_slice(k.as_ref()), *ts),
                        ValueBytes::new(DEFAULT_NS, Bytes::copy_from_slice(v.as_ref())),
                    ));
                },
                | Delete(k, ts) => {
                    _batch.push((
                        KeyBytes::new(DEFAULT_NS, Bytes::copy_from_slice(k.as_ref()), *ts),
                        ValueBytes::new_tombstone(DEFAULT_NS),
                    ));
                },
            }
        }

        // Fast path: try to write entire batch to current memtable
        let mtable = self.curr_memtable.read().clone();

        match mtable.put_batch(_batch.as_ref()) {
            | Ok(written) if written == _batch.len() => {
                // All written, done!
                Ok(())
            },
            | Ok(written) => {
                // Partial write - need to handle remaining with memtable swaps
                let mut offset = written;
                let mut last_attempted = mtable.clone();
                self.write_batch_with_swap(&_batch, &mut offset, &mut last_attempted)
            },
            | Err(e) => {
                match e {
                    | MtError::DataExceedsMaximum => {
                        // First entry doesn't fit - same logic as partial write loop
                        let mut offset = 0usize;
                        let mut last_attempted = mtable.clone();
                        self.write_batch_with_swap(&_batch, &mut offset, &mut last_attempted)
                    },
                    | MtError::MemtableIsFrozen => {
                        // Memtable was frozen during write - get current and retry
                        let new_mtable = {
                            let guard = self.state.lock();
                            let new = guard.current_memtable();
                            *self.curr_memtable.write() = new.clone();
                            new
                        };
                        match new_mtable.put_batch(_batch.as_ref()) {
                            | Ok(_) => Ok(()),
                            | Err(e) => Err(MemtableError(e)),
                        }
                    },
                }
            },
        }
    }

    pub(crate) fn sync(&self) -> Result<(), CesiumError> {
        let mut guard = self.state.lock();
        guard.sync()?;
        // Update cached curr_memtable to match the new empty memtable
        // created by sync(). Without this, db.get would continue reading
        // from the old (flushed) memtable via the stale cache.
        let new_mtable = guard.current_memtable();
        *self.curr_memtable.write() = new_mtable;
        Ok(())
    }

    /// Block until frozen memtables are below the limit.
    /// This prevents unbounded memory growth when the flusher can't keep up.
    pub(crate) fn wait_for_frozen_capacity(&self) {
        let limit = {
            let guard = self.state.lock();
            guard.memtable_limit()
        };
        // If limit is 0, disable backpressure (default behavior)
        if limit == 0 {
            return;
        }
        loop {
            let frozen = {
                let guard = self.state.lock();
                guard.frozen_count()
            };
            if frozen < limit as usize {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
    }

    pub(crate) fn version_stats(&self) -> VersionStats {
        self.state.lock().version_stats()
    }

    pub(crate) fn read_amp_stats(&self) -> ReadAmpStats {
        ReadAmpStats {
            total_gets: self.total_gets.load(Ordering::Relaxed),
            l0_segments_checked: self.l0_reads.load(Ordering::Relaxed),
            ln_segments_checked: self.ln_reads.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn frozen_memtable_count(&self) -> usize {
        self.state.lock().frozen_count()
    }
}
