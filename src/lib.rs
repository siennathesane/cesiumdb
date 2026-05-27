// Copyright (c) Sienna Meridian Satterwhite
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

#![doc = include_str!("../README.md")]
#![feature(sync_unsafe_cell)]
#![cfg_attr(target_arch = "aarch64", feature(integer_atomics))]
#![deny(dead_code)]
#![deny(unused)]
#![deny(unused_mut)]
#![deny(clippy::missing_safety_doc)]
#![deny(clippy::undocumented_unsafe_blocks)]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::unwrap_used))]
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

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[allow(unused)]
use std::sync::Arc;

use bytes::Bytes;

use crate::{
    Batch::{
        Delete,
        DeleteNs,
        Put,
        PutNs,
    },
    errs::CesiumError,
    hlc::HLC,
    keypair::{
        DEFAULT_NS,
        KeyBytes,
    },
};

pub mod autoconfig;

/// Makes a module `pub` when the `benchmarks` feature is enabled,
/// and `pub(crate)` otherwise. This keeps the public API surface minimal
/// while allowing benchmark and fuzz targets to access internals.
macro_rules! bench_pub_mod {
    ($name:ident) => {
        pub mod $name;
    };
}

bench_pub_mod!(block);

mod block_alloc;
mod bloom;

bench_pub_mod!(compact);

bench_pub_mod!(compaction);
pub mod errs;
mod hash;
pub mod hlc;
mod index;
pub(crate) mod io;
bench_pub_mod!(keypair);
bench_pub_mod!(levels);
pub(crate) mod manifest;
pub(crate) mod manifest_reader;
pub(crate) mod manifest_writer;

bench_pub_mod!(map);
bench_pub_mod!(memtable);
bench_pub_mod!(merge);
pub(crate) mod peek;
bench_pub_mod!(raw_entry);
bench_pub_mod!(segment);
bench_pub_mod!(segment_builder);
mod segment_iterator;
pub(crate) mod segment_reader;

bench_pub_mod!(segment_writer);

pub(crate) mod simd;
pub(crate) mod state;
mod stats;
bench_pub_mod!(utils);
bench_pub_mod!(version);

#[cfg(feature = "telemetry")]
pub mod telemetry;

// Internal implementation modules.
mod db_inner;
mod db_options;
mod scan;

// Re-export types that appear in public API signatures so consumers don't
// need to reach into internal modules.
pub use compaction::{
    CompactionStats,
    SchedulerConfig,
};
pub(crate) use db_inner::DbInner;
pub use db_options::{
    Batch,
    DbOptions,
};
pub use scan::OwnedSegmentIterator;
pub use scan::{
    DbScanIterator,
    ReadAmpStats,
};

pub use version::VersionStats;

/// The core CesiumDB API. Use this to create or open a database, and perform
/// point lookups and batch operations.
pub struct Db {
    pub(crate) inner: Arc<DbInner>,
    clock: Arc<dyn HLC>,
}

impl Drop for Db {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

impl Db {
    /// Create or open an existing database.
    pub fn open(opts: DbOptions) -> Result<Arc<Self>, CesiumError> {
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

    /// Scan a range of keys in a specific namespace.
    ///
    /// Returns an iterator over key-value pairs within the specified bounds.
    /// The iterator merges results from memtables and all LSM levels,
    /// automatically handling deduplication and tombstone filtering.
    ///
    /// # Arguments
    ///
    /// * `ns` - The namespace to scan
    /// * `lower` - Lower bound (Unbounded, Included, or Excluded)
    /// * `upper` - Upper bound (Unbounded, Included, or Excluded)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::ops::Bound;
    ///
    /// use cesiumdb::{
    ///     Db,
    ///     DbOptions,
    /// };
    ///
    /// let db = Db::open(DbOptions::default()).unwrap();
    /// let start = b"key-00000".to_vec();
    /// let end = b"key-99999".to_vec();
    ///
    /// for (key, value) in db.scan_ns(0, Bound::Included(&start), Bound::Excluded(&end)) {
    ///     println!("Key: {:?}, Value: {:?}", key, value);
    /// }
    /// ```
    pub fn scan_ns(
        &self,
        ns: u64,
        lower: std::ops::Bound<&[u8]>,
        upper: std::ops::Bound<&[u8]>,
    ) -> Result<DbScanIterator, CesiumError> {
        self.inner.scan(ns, lower, upper)
    }

    /// Scan a range of keys in the default namespace.
    ///
    /// See [`scan_ns`](Self::scan_ns) for more details.
    pub fn scan(
        &self,
        lower: std::ops::Bound<&[u8]>,
        upper: std::ops::Bound<&[u8]>,
    ) -> Result<DbScanIterator, CesiumError> {
        self.scan_ns(DEFAULT_NS, lower, upper)
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
    pub fn compaction_stats(&self) -> Result<CompactionStats, CesiumError> {
        let guard = self.inner.state.lock();
        match guard.compaction_stats() {
            | Some(stats) => Ok(stats),
            | None => Err(CesiumError::CompactionError(
                crate::errs::CompactionError::NotInitialized,
            )),
        }
    }

    /// Returns current version statistics.
    ///
    /// Provides:
    /// - L0 segment count
    /// - Total segment count across all levels
    /// - Total database size in bytes
    /// - Current sequence number
    pub fn version_stats(&self) -> VersionStats {
        self.inner.version_stats()
    }

    /// Returns read amplification statistics.
    pub fn read_amp_stats(&self) -> ReadAmpStats {
        self.inner.read_amp_stats()
    }

    /// Returns the number of frozen memtables waiting to be flushed.
    pub fn frozen_memtable_count(&self) -> usize {
        self.inner.frozen_memtable_count()
    }
}

#[cfg(test)]
mod tests;
