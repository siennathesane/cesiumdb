// Copyright (c) Sienna Satterwhite, CesiumDB Contributors

// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

#![feature(sync_unsafe_cell)]
#![cfg_attr(target_arch = "aarch64", feature(integer_atomics))]
#![allow(dead_code)]
#![allow(unused)]
#![deny(unused_mut)]
#![deny(clippy::missing_safety_doc)]
#![deny(clippy::undocumented_unsafe_blocks)]
// for @siennathesane's sanity and to make it clear the scope of error handling. and because it's
// super fucking subtle and i'll miss it in code reviews sorry not sorry
#![deny(clippy::question_mark_used)]
// just keeps syntax consistent
#![deny(clippy::needless_borrow)]
// personal preference.
#![allow(bindings_with_variant_name)]

#[cfg(not(unix))]
compile_warn!("cesiumdb is not tested on windows");
#[cfg(not(target_pointer_width = "64"))]
compile_warn!("cesiumdb is not tested on 32-bit systems");

#[allow(unused)]
use std::sync::Arc;

use bytes::Bytes;
use parking_lot::Mutex;

use crate::{
    Batch::{
        Delete,
        DeleteNs,
        Put,
        PutNs,
    },
    errs::{
        CesiumError,
        CesiumError::MemtableError,
    },
    hlc::{
        HLC,
        HybridLogicalClock,
    },
    keypair::{
        DEFAULT_NS,
        KeyBytes,
        ValueBytes,
    },
    state::{
        DbStorageBuilder,
        DbStorageState,
    },
};

#[cfg(feature = "benchmarks")]
pub mod block;
#[cfg(not(feature = "benchmarks"))]
pub(crate) mod block;

mod block_alloc;

#[cfg(feature = "benchmarks")]
pub mod compact;
#[cfg(not(feature = "benchmarks"))]
pub(crate) mod compact;

pub mod compaction;
pub mod errs;
mod hash;
pub mod hlc;
mod index;
pub(crate) mod io;
pub mod keypair;
pub mod levels;
pub(crate) mod manifest;
pub(crate) mod manifest_reader;
pub(crate) mod manifest_writer;

#[cfg(feature = "benchmarks")]
pub mod map;
#[cfg(not(feature = "benchmarks"))]
pub(crate) mod map;

pub mod memtable;
pub mod merge;
pub mod peek;
pub(crate) mod raw_entry;
pub mod segment;
mod segment_builder;
mod segment_iterator;
pub(crate) mod segment_reader;

#[cfg(feature = "benchmarks")]
pub mod segment_writer;
#[cfg(not(feature = "benchmarks"))]
pub(crate) mod segment_writer;

pub mod simd;
pub(crate) mod state;
mod stats;
pub mod utils;
pub mod version;

/// The core Cesium database! The API is simple by design, and focused on
/// performance. It is designed for heavy concurrency, implements sharding, and
/// Multi-Version Concurrency Control (MVCC).
pub struct Db {
    inner: Arc<DbInner>,
    clock: Arc<dyn HLC>,
}

impl Db {
    /// Create or open an existing database.
    pub fn open(opts: DbOptions) -> Arc<Self> {
        opts.build()
    }

    /// Fetches the current time according to the clock. This is designed to be
    /// used for batch operations so callers can set the order of updates.
    /// This provides a bit of determinism for callers and allows for a lot
    /// of different use cases.
    pub fn time(&self) -> u128 {
        self.clock.time()
    }

    /// Put a key into a specific namespace.
    pub fn put_ns(&self, ns: u64, key: &[u8], value: &[u8]) -> Result<(), CesiumError> {
        self.inner
            .batch(&[PutNs(ns, key, value, self.clock.time())])
    }

    /// Get a key from a specific namespace.
    pub fn get_ns(&self, ns: u64, key: &[u8]) -> Result<Option<Bytes>, CesiumError> {
        match self
            .inner
            .get(KeyBytes::new(ns, Bytes::copy_from_slice(key), 0))
        {
            | Ok(v) => match v {
                | None => Ok(None),
                | Some(v) => Ok(Some(v.as_bytes())),
            },
            | Err(e) => Err(e),
        }
    }

    /// Delete a key from a specific namespace.
    pub fn delete_ns(&self, ns: u64, key: &[u8]) -> Result<(), CesiumError> {
        self.inner
            .batch::<&[u8], &[u8]>(&[DeleteNs(ns, key, self.clock.time())])
    }

    /// Put a key.
    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<(), CesiumError> {
        self.put_ns(DEFAULT_NS, key, value)
    }

    /// Get a key.
    pub fn get(&self, key: &[u8]) -> Result<Option<Bytes>, CesiumError> {
        self.get_ns(DEFAULT_NS, key)
    }

    /// Delete a key.
    pub fn delete(&self, key: &[u8]) -> Result<(), CesiumError> {
        self.delete_ns(DEFAULT_NS, key)
    }

    /// Write a batch of records to the database. It is safe to mix namespaced
    /// and un-namespaced records.
    pub fn batch<K: AsRef<[u8]>, V: AsRef<[u8]>>(
        &self,
        ops: &[Batch<K, V>],
    ) -> Result<(), CesiumError> {
        let _ops = ops
            .iter()
            .map(|b| match b {
                | Put(k, v, ts) => PutNs(DEFAULT_NS, k, v, *ts),
                | PutNs(ns, k, v, ts) => PutNs(*ns, k, v, *ts),
                | Delete(k, ts) => DeleteNs(DEFAULT_NS, k, *ts),
                | DeleteNs(ns, k, ts) => DeleteNs(*ns, k, *ts),
            })
            .collect::<Vec<_>>();
        self.inner.batch(&_ops)
    }

    /// Sync the database to disk. This is a blocking operation and will cause
    /// delays under heavy write scenarios.
    pub fn sync(&self) -> Result<(), CesiumError> {
        self.inner.sync()
    }

    /// Close the database. This performs an orderly shutdown:
    /// 1. Freezes the current memtable
    /// 2. Waits for background flusher to drain all frozen memtables
    /// 3. Shuts down the compaction manager
    pub fn close(&self) -> Result<(), CesiumError> {
        self.inner.state.lock().shutdown()
    }

    /// Triggers a manual compaction of the entire database.
    ///
    /// This is a synchronous operation that will compact all levels.
    /// Useful for:
    /// - Reclaiming space after deletions
    /// - Optimizing read performance
    /// - Forcing cleanup of old versions
    pub fn compact(&self) -> Result<(), CesiumError> {
        let guard = self.inner.state.lock();
        guard.compact();
        Ok(())
    }

    /// Returns current compaction statistics.
    ///
    /// This provides insights into:
    /// - Number of queued/in-progress/completed jobs
    /// - Parallel execution utilization
    /// - Current workload pattern
    pub fn compaction_stats(&self) -> Result<crate::compaction::CompactionStats, CesiumError> {
        let guard = self.inner.state.lock();
        match guard.compaction_stats() {
            | Some(stats) => Ok(stats),
            | None => Err(CesiumError::CompactionError(
                crate::errs::CompactionError::NotInitialized,
            )),
        }
    }
}

/// Configuration options for Cesium.
#[repr(C)]
pub struct DbOptions {
    engine_opts: DbStorageBuilder,
    clock: Arc<dyn HLC>,
}

impl DbOptions {
    pub fn new() -> Self {
        Self {
            engine_opts: DbStorageBuilder::default(),
            clock: Arc::new(HybridLogicalClock::new()),
        }
    }

    pub fn engine(&mut self, engine: DbStorageBuilder) -> &mut Self {
        self.engine_opts = engine;
        self
    }

    /// **The Hybrid Linear Clock** *(and how MVCC works in LSM-trees)*
    ///
    /// By default, CesiumDB used the bundled hybrid linear clock, which
    /// provides a perfectly incrementing clock, to determine when writes
    /// happen. The clock implementation is "client-side", so CesiumDB
    /// assumes a write happened when the caller said it did. This is
    /// overrideable behaviour, and consumers can implement their own clock
    /// via the [`HLC`] trait. Theoretically a provided implementation can
    /// move the clock to an earlier time than when the DB comes online,
    /// however that could result in older keys get overwritten.
    ///
    /// In an LSM-tree, multiple versions of a key can exist until flushing and
    /// compaction events. When you call `Db.put(b"key", b"value")`, it
    /// attaches an internal timestamp based on when that API is called and
    /// then encodes the reversed timestamp into the key value, along with a
    /// namespace. As LSM-trees are append-only data structures,
    /// `Db.get(b"key")` will always return the latest value. When flushing
    /// happens, the memtables are merged into N sorted string tables (not
    /// actual strings) and duplicate key versions are merged into "latest"
    /// to produce a single key for the sstables. When compaction happens,
    /// the various levels of sstables (and various sstables in a specific
    /// level) are merged together and the same key duplication is checked.
    ///
    /// If you provide your own clock source, in order to ensure that the most
    /// recent version of your keys is updated on `Db.put`, you need to make
    /// sure that your most recently updated key's - the last key written to
    /// the database before `Db.close` is called - timestamp is less than
    /// `HLC.time` before any other key is updated. If this happens, it is
    /// considered undefined behavior and is not protected against.
    ///
    /// It's recommended to use the provided HLC as it has a general resolution
    /// of 2-3ns on average.
    pub fn clock(&mut self, clock: Arc<dyn HLC>) -> &mut Self {
        self.clock = clock;
        self
    }

    /// Sets the data directory for persistent storage.
    ///
    /// When set, enables:
    /// - Background compaction threads
    /// - Persistent SSTable storage
    /// - Automatic flush-to-disk
    pub fn data_dir(&mut self, path: std::path::PathBuf) -> &mut Self {
        self.engine_opts = self.engine_opts.clone().base_path(path);
        self
    }

    /// Sets the memtable size in bytes (default: configured in memtable
    /// module).
    ///
    /// Smaller memtables = more frequent flushes, less memory usage
    /// Larger memtables = fewer flushes, more memory usage
    pub fn memtable_size(&mut self, size: u64) -> &mut Self {
        // Note: This would need to be added to DbStorageBuilder
        // For now, this is a placeholder
        self
    }

    /// Sets the maximum number of memtables before blocking writes.
    ///
    /// This is the num_memtable_limit parameter.
    pub fn max_memtables(&mut self, count: u64) -> &mut Self {
        self.engine_opts = self.engine_opts.clone().num_memtable_limit(count);
        self
    }

    pub fn build(&self) -> Arc<Db> {
        let mut builder = DbStorageBuilder::new()
            .block_size(self.engine_opts.block_size)
            .target_sst_size(self.engine_opts.target_sst_size)
            .num_memtable_limit(self.engine_opts.num_memtable_limit);

        if let Some(ref path) = self.engine_opts.base_path {
            builder = builder.base_path(path.clone());
        }

        let state = builder.build();

        // Create warm thread pool for parallel LSM reads
        // Use half the available cores for reads to leave room for writes
        let num_read_threads = std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(2))
            .unwrap_or(4);

        let read_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_read_threads)
            .thread_name(|i| format!("lsm-reader-{}", i))
            .build()
            .expect("failed to create read thread pool");

        let inner = DbInner { state, read_pool };

        Arc::new(Db {
            inner: Arc::new(inner),
            clock: self.clock.clone(),
        })
    }
}

impl Default for DbOptions {
    fn default() -> Self {
        Self::new()
    }
}

#[repr(C)]
pub enum Batch<K: AsRef<[u8]>, V: AsRef<[u8]>> {
    Put(K, V, u128),
    Delete(K, u128),
    PutNs(u64, K, V, u128),
    DeleteNs(u64, K, u128),
}

#[repr(C)]
struct DbInner {
    state: Mutex<DbStorageState>,
    /// Warm thread pool for parallel LSM reads across levels
    read_pool: rayon::ThreadPool,
}

impl DbInner {
    fn get(&self, key: KeyBytes) -> Result<Option<ValueBytes>, CesiumError> {
        // 1. Check current memtable (hottest data)
        {
            let guard = self.state.lock();
            let val = guard.current_memtable().get(key.clone());
            if let Some(val) = val {
                // Return None for tombstones
                if val.is_tombstone() {
                    return Ok(None);
                }
                return Ok(Some(val));
            }
        }

        // 2. Check frozen memtables (newest to oldest)
        {
            let guard = self.state.lock();
            if let Some(val) = guard.get_from_frozen(key.clone()) {
                if val.is_tombstone() {
                    return Ok(None);
                }
                return Ok(Some(val));
            }
        }

        // 3. Check L0-L7 via VersionManager (parallelized)
        {
            use rayon::prelude::*;

            use crate::utils::Serializer;

            let guard = self.state.lock();
            let version = guard.version_manager.current();
            let key_bytes = key.serialize();

            // Check L0 (newest to oldest - reverse chronological)
            // L0 must be checked sequentially because newer segments override older ones
            // We need to search by key prefix (ns + key) to find any version
            // Since timestamps are stored as (u128::MAX - ts), newest=0, oldest=u128::MAX
            let key_prefix_lower = {
                use bytes::{
                    BufMut,
                    BytesMut,
                };
                let mut bytes = BytesMut::with_capacity(8 + key.as_bytes().len() + 16);
                bytes.put_u64_le(key.ns());
                bytes.put_slice(key.as_bytes().as_ref());
                bytes.put_u128_le(0); // newest possible (u128::MAX - u128::MAX = 0)
                bytes.freeze()
            };
            let key_prefix_upper = {
                use bytes::{
                    BufMut,
                    BytesMut,
                };
                let mut bytes = BytesMut::with_capacity(8 + key.as_bytes().len() + 16);
                bytes.put_u64_le(key.ns());
                bytes.put_slice(key.as_bytes().as_ref());
                bytes.put_u128_le(u128::MAX); // oldest possible (u128::MAX - 0 = u128::MAX)
                bytes.freeze()
            };

            // Prepare key without timestamp for bloom filter checks (L1-L7 only)
            let key_for_bloom = {
                use bytes::{
                    BufMut,
                    BytesMut,
                };
                let mut bytes = BytesMut::with_capacity(8 + key.as_bytes().len());
                bytes.put_u64_le(key.ns());
                bytes.put_slice(key.as_bytes().as_ref());
                bytes.freeze()
            };

            // Check L0 segments in reverse chronological order (newest first)
            for segment in version.l0.iter().rev() {
                let reader = match segment.reader() {
                    | Ok(r) => r,
                    | Err(_) => continue,
                };

                // Scan for keys matching this prefix (any timestamp)
                use std::ops::Bound;
                let mut scan_iter = reader.scan(
                    Bound::Included(key_prefix_lower.as_ref()),
                    Bound::Included(key_prefix_upper.as_ref()),
                );

                // Take the first match (newest version due to timestamp ordering)
                // scan_iter returns (KeyBytes, ValueBytes) already deserialized
                if let Some(Ok((_, val))) = scan_iter.next() {
                    if val.is_tombstone() {
                        return Ok(None);
                    }
                    return Ok(Some(val));
                }
            }

            // Check L1-L7 in parallel using warm thread pool
            // Keys don't overlap within a level in leveled compaction, so parallel search
            // is safe
            let key_prefix_lower_clone = key_prefix_lower.clone();
            let key_prefix_upper_clone = key_prefix_upper.clone();
            let key_for_bloom_clone = key_for_bloom.clone();
            let result = self.read_pool.install(|| {
                version.levels.par_iter().find_map_any(|level| {
                    // Within each level, search segments
                    for segment in &level.segments {
                        if let Ok(reader) = segment.reader() {
                            // Fast bloom filter check - skip segments that definitely don't have
                            // this key
                            if !reader.may_contain(&key_for_bloom_clone) {
                                continue;
                            }

                            use std::ops::Bound;
                            let mut scan_iter = reader.scan(
                                Bound::Included(key_prefix_lower_clone.as_ref()),
                                Bound::Included(key_prefix_upper_clone.as_ref()),
                            );
                            if let Some(Ok((_, val))) = scan_iter.next() {
                                return Some(val);
                            }
                        }
                    }
                    None
                })
            });

            if let Some(val) = result {
                if val.is_tombstone() {
                    return Ok(None);
                }
                return Ok(Some(val));
            }
        }

        // 4. Not found anywhere
        Ok(None)
    }

    fn batch<K: AsRef<[u8]>, V: AsRef<[u8]>>(
        &self,
        ops: &[Batch<K, V>],
    ) -> Result<(), CesiumError> {
        let _batch = ops
            .iter()
            .filter_map(|b| match b {
                | PutNs(ns, k, v, ts) => Some((
                    KeyBytes::new(*ns, Bytes::from(k.as_ref().to_owned()), *ts),
                    ValueBytes::new(*ns, Bytes::from(v.as_ref().to_owned())),
                )),
                | DeleteNs(ns, k, ts) => Some((
                    KeyBytes::new(*ns, Bytes::from(k.as_ref().to_owned()), *ts),
                    ValueBytes::new_tombstone(*ns),
                )),
                | _ => None, // filter out invalid enums
            })
            .collect::<Vec<_>>();

        // Fast path: try to write entire batch to current memtable
        let mtable = {
            let guard = self.state.lock();
            guard.current_memtable()
        };

        match mtable.put_batch(_batch.as_ref()) {
            | Ok(written) if written == _batch.len() => {
                // All written, done!
                Ok(())
            },
            | Ok(written) => {
                // Partial write - need to handle remaining with memtable swaps
                let mut offset = written;
                while offset < _batch.len() {
                    // Swap memtable
                    let new_mtable = {
                        let mut guard = self.state.lock();
                        guard.new_memtable();
                        guard.current_memtable()
                    };

                    // Write remaining to new memtable
                    match new_mtable.put_batch(&_batch[offset..]) {
                        | Ok(w) => {
                            offset += w;
                            if offset >= _batch.len() {
                                return Ok(());
                            }
                        },
                        | Err(e) => {
                            use crate::errs::MemtableError as MtError;
                            // If frozen, retry with current memtable (which was swapped)
                            if matches!(e, MtError::MemtableIsFrozen) {
                                continue; // Retry loop with current memtable
                            }
                            return Err(MemtableError(e));
                        },
                    }
                }
                Ok(())
            },
            | Err(e) => {
                use crate::errs::MemtableError as MtError;
                match e {
                    | MtError::DataExceedsMaximum => {
                        // First entry doesn't fit - swap and retry whole batch
                        let new_mtable = {
                            let mut guard = self.state.lock();
                            guard.new_memtable();
                            guard.current_memtable()
                        };
                        match new_mtable.put_batch(_batch.as_ref()) {
                            | Ok(_) => Ok(()),
                            | Err(e) => Err(MemtableError(e)),
                        }
                    },
                    | MtError::MemtableIsFrozen => {
                        // Memtable was frozen during write - get current and retry
                        // (background flusher swaps memtables asynchronously)
                        let new_mtable = {
                            let guard = self.state.lock();
                            guard.current_memtable()
                        };
                        match new_mtable.put_batch(_batch.as_ref()) {
                            | Ok(_) => Ok(()),
                            | Err(e) => Err(MemtableError(e)),
                        }
                    },
                    | _ => Err(MemtableError(e)),
                }
            },
        }
    }

    fn sync(&self) -> Result<(), CesiumError> {
        self.state.lock().sync()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        Batch::Put,
        Db,
        DbOptions,
    };

    const MAX_KEYS: u64 = 10_000;

    fn db_builder() -> Arc<Db> {
        Db::open(DbOptions::default())
    }

    #[test]
    fn test_db_put() {
        let db = db_builder();

        // initial insert
        let mut keypair_size = 0;
        for i in 0..MAX_KEYS {
            let key = format!("key-{}", i).into_bytes();
            let val = format!("value-{}", i).into_bytes();
            keypair_size += key.len() + val.len();
            assert!(db.put(key.as_ref(), val.as_ref()).is_ok());
        }

        {
            let guard = db.inner.state.lock();
            assert!(
                guard.current_memtable().size() > keypair_size as u64,
                "the memtable must be bigger than the keypair size to ensure the keys are actually stored"
            );
        }

        // re-insert the same keys but with new versions
        for i in 0..MAX_KEYS {
            let key = format!("key-{}", i).into_bytes();
            let val = format!("value-{}", i).into_bytes();
            keypair_size += key.len() + val.len();
            assert!(db.put(key.as_ref(), val.as_ref()).is_ok());
        }

        {
            let guard = db.inner.state.lock();
            assert!(
                guard.current_memtable().size() > (keypair_size * 2) as u64,
                "the memtable must be at least twice as big as before with the new versions"
            );
        }
    }

    #[test]
    fn db_put_batch() {
        let db = db_builder();

        let mut keypair_size = 0;
        for batch_size in [1, 10, 100].iter() {
            let mut batch = Vec::with_capacity(*batch_size);

            for i in 0..(*batch_size * 100) {
                let key = format!("key-{}", i).into_bytes();
                let val = format!("value-{}", i).into_bytes();
                keypair_size += key.len() + val.len();

                let op = Put(key, val.clone(), db.time());
                batch.push(op)
            }

            assert!(db.batch(&batch).is_ok());

            {
                let guard = db.inner.state.lock();
                assert!(
                    guard.current_memtable().size() > keypair_size as u64,
                    "the memtable must be bigger than the keypair size to ensure the keys are actually stored"
                );
            }
        }
    }

    #[test]
    fn test_db_get() {
        let db = db_builder();

        // test get on empty db
        let result = db.get(b"nonexistent");
        assert!(result.is_ok());
        assert!(
            result.unwrap().is_none(),
            "get on empty db should return None"
        );

        // insert and retrieve
        let key = b"test-key";
        let val = b"test-value";
        assert!(db.put(key, val).is_ok());

        let result = db.get(key);
        assert!(result.is_ok());
        let retrieved = result.unwrap();
        assert!(
            retrieved.is_some(),
            "get should return Some for existing key"
        );
        assert_eq!(&retrieved.unwrap()[..], val, "retrieved value should match");

        // test get on different key
        let result = db.get(b"different-key");
        assert!(result.is_ok());
        assert!(
            result.unwrap().is_none(),
            "get on non-existent key should return None"
        );
    }

    #[test]
    fn test_db_get_latest_version() {
        let db = db_builder();

        let key = b"versioned-key";
        let val1 = b"value-1";
        let val2 = b"value-2";
        let val3 = b"value-3";

        // insert multiple versions
        assert!(db.put(key, val1).is_ok());
        assert!(db.put(key, val2).is_ok());
        assert!(db.put(key, val3).is_ok());

        // get should return the latest version
        let result = db.get(key);
        assert!(result.is_ok());
        let retrieved = result.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(
            &retrieved.unwrap()[..],
            val3,
            "get should return the latest value"
        );
    }

    #[test]
    fn test_db_delete() {
        let db = db_builder();

        let key = b"key-to-delete";
        let val = b"value";

        // insert key
        assert!(db.put(key, val).is_ok());

        // verify it exists
        let result = db.get(key);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());

        // delete the key
        assert!(db.delete(key).is_ok());

        // verify the key no longer exists (tombstone filters it out)
        let result = db.get(key);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none(), "deleted key should return None");
    }

    #[test]
    fn test_db_put_ns() {
        let db = db_builder();

        let ns1: u64 = 1;
        let ns2: u64 = 2;
        let key = b"same-key";
        let val1 = b"value-in-ns1";
        let val2 = b"value-in-ns2";

        // put same key in different namespaces
        assert!(db.put_ns(ns1, key, val1).is_ok());
        assert!(db.put_ns(ns2, key, val2).is_ok());

        // retrieve from each namespace
        let result1 = db.get_ns(ns1, key);
        assert!(result1.is_ok());
        let value1 = result1.unwrap();
        assert!(value1.is_some());
        assert_eq!(&value1.unwrap()[..], val1);

        let result2 = db.get_ns(ns2, key);
        assert!(result2.is_ok());
        let value2 = result2.unwrap();
        assert!(value2.is_some());
        assert_eq!(&value2.unwrap()[..], val2);
    }

    #[test]
    fn test_db_get_ns() {
        let db = db_builder();

        let ns: u64 = 42;
        let key = b"namespaced-key";
        let val = b"namespaced-value";

        // get on empty namespace
        let result = db.get_ns(ns, key);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());

        // insert into namespace
        assert!(db.put_ns(ns, key, val).is_ok());

        // retrieve from namespace
        let result = db.get_ns(ns, key);
        assert!(result.is_ok());
        let retrieved = result.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(&retrieved.unwrap()[..], val);

        // verify key doesn't exist in default namespace
        let result = db.get(key);
        assert!(result.is_ok());
        assert!(
            result.unwrap().is_none(),
            "key should not exist in default namespace"
        );
    }

    #[test]
    fn test_db_delete_ns() {
        let db = db_builder();

        let ns: u64 = 10;
        let key = b"key-to-delete";
        let val = b"value";

        // insert into namespace
        assert!(db.put_ns(ns, key, val).is_ok());

        // verify it exists
        let result = db.get_ns(ns, key);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());

        // delete from namespace
        assert!(db.delete_ns(ns, key).is_ok());

        // verify the key no longer exists (tombstone filters it out)
        let result = db.get_ns(ns, key);
        assert!(result.is_ok());
        assert!(
            result.unwrap().is_none(),
            "deleted key in namespace should return None"
        );
    }

    #[test]
    fn test_db_options_default() {
        let opts = DbOptions::default();
        let db = Db::open(opts);

        // basic operation to ensure default options work
        assert!(db.put(b"test", b"value").is_ok());
        let result = db.get(b"test");
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_db_time() {
        let db = db_builder();

        let time1 = db.time();
        let time2 = db.time();

        // time should be monotonically increasing
        assert!(
            time2 >= time1,
            "clock should return monotonically increasing values"
        );
    }

    #[test]
    fn test_db_batch_mixed_operations() {
        use crate::Batch::{
            Delete,
            DeleteNs,
            PutNs,
        };

        let db = db_builder();

        let ns: u64 = 5;
        let batch = vec![
            Put(b"key1".to_vec(), b"val1".to_vec(), db.time()),
            PutNs(ns, b"key2".to_vec(), b"val2".to_vec(), db.time()),
            Put(b"key3".to_vec(), b"val3".to_vec(), db.time()),
        ];

        assert!(db.batch(&batch).is_ok());

        // verify all operations succeeded
        assert!(db.get(b"key1").unwrap().is_some());
        assert!(db.get_ns(ns, b"key2").unwrap().is_some());
        assert!(db.get(b"key3").unwrap().is_some());
    }

    #[test]
    fn test_db_empty_key() {
        let db = db_builder();

        let key = b"";
        let val = b"empty-key-value";

        assert!(db.put(key, val).is_ok());
        let result = db.get(key);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_db_empty_value() {
        let db = db_builder();

        let key = b"key-with-empty-value";
        let val = b"";

        assert!(db.put(key, val).is_ok());
        let result = db.get(key);
        assert!(result.is_ok());
        let retrieved = result.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().len(), 0);
    }

    #[test]
    fn test_db_large_key_value() {
        let db = db_builder();

        let key = vec![b'k'; 1000];
        let val = vec![b'v'; 10000];

        assert!(db.put(&key, &val).is_ok());
        let result = db.get(&key);
        assert!(result.is_ok());
        let retrieved = result.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().len(), val.len());
    }
}
