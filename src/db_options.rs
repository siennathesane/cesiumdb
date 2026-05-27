// Copyright (c) Sienna Meridian Satterwhite
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

//! Database configuration and batch operations.
//!
//! Provides [`DbOptions`] for configuring the database, and [`Batch`]
//! for atomic multi-operation writes.

use std::sync::{
    Arc,
    atomic::AtomicU64,
};

use parking_lot::RwLock;

use crate::{
    Db,
    DbInner,
    SchedulerConfig,
    errs::CesiumError,
    hlc::{
        HLC,
        HybridLogicalClock,
    },
    state::DbStorageBuilder,
};

/// Configuration options for Cesium.
#[repr(C)]
pub struct DbOptions {
    pub(crate) engine_opts: DbStorageBuilder,
    pub(crate) clock: Arc<dyn HLC>,
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
    /// to produce a single key for the segments. When compaction happens,
    /// the various levels of segments (and various segments in a specific
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
    /// - Persistent Segment storage
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
        self.engine_opts = self.engine_opts.clone().memtable_size(size);
        self
    }

    /// Sets the target Segment size in bytes.
    ///
    /// Larger segments reduce file count and compaction overhead,
    /// but increase memory usage during compaction.
    pub fn target_segment_size(&mut self, size: u64) -> &mut Self {
        self.engine_opts = self.engine_opts.clone().target_segment_size(size);
        self
    }

    /// Sets the multiplier for target file size per level.
    ///
    /// Level N target size = `target_segment_size * multiplier^(N-1)`.
    /// Default is 1 (same size for all levels).
    pub fn target_file_size_multiplier(&mut self, multiplier: u64) -> &mut Self {
        let mut scheduler = self.engine_opts.scheduler_config.clone();
        scheduler.target_file_size_multiplier = multiplier;
        self.engine_opts = self.engine_opts.clone().scheduler_config(scheduler);
        self
    }

    /// Sets the maximum number of memtables before blocking writes.
    ///
    /// This is the num_memtable_limit parameter.
    pub fn max_memtables(&mut self, count: u64) -> &mut Self {
        self.engine_opts = self.engine_opts.clone().num_memtable_limit(count);
        self
    }

    /// Sets the compaction scheduler configuration.
    pub fn scheduler_config(&mut self, config: SchedulerConfig) -> &mut Self {
        self.engine_opts = self.engine_opts.clone().scheduler_config(config);
        self
    }

    pub fn build(&self) -> Result<Arc<Db>, CesiumError> {
        let mut builder = DbStorageBuilder::new()
            .block_size(self.engine_opts.block_size)
            .target_segment_size(self.engine_opts.target_segment_size)
            .num_memtable_limit(self.engine_opts.num_memtable_limit)
            .memtable_size(self.engine_opts.memtable_size)
            .scheduler_config(self.engine_opts.scheduler_config.clone());

        if let Some(ref path) = self.engine_opts.base_path {
            builder = builder.base_path(path.clone());
        }

        let state = match builder.build() {
            | Ok(s) => s,
            | Err(e) => return Err(e),
        };

        // Create warm thread pool for parallel LSM reads
        // Use half the available cores for reads to leave room for writes
        let num_read_threads = std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(2))
            .unwrap_or(4);

        let read_pool = match rayon::ThreadPoolBuilder::new()
            .num_threads(num_read_threads)
            .thread_name(|i| format!("lsm-reader-{}", i))
            .build()
        {
            | Ok(pool) => pool,
            | Err(e) => {
                return Err(CesiumError::InitializationError(format!(
                    "failed to create read thread pool: {}",
                    e
                )));
            },
        };

        let (curr_memtable, version_manager, frozen_memtables) = {
            let guard = state.lock();
            (
                guard.current_memtable(),
                Arc::clone(&guard.version_manager),
                guard.frozen_memtables_arc(),
            )
        };
        let inner = DbInner {
            state,
            curr_memtable: RwLock::new(curr_memtable),
            frozen_memtables,
            version_manager,
            read_pool,
            total_gets: AtomicU64::new(0),
            l0_reads: AtomicU64::new(0),
            ln_reads: AtomicU64::new(0),
        };

        Ok(Arc::new(Db {
            inner: Arc::new(inner),
            clock: self.clock.clone(),
        }))
    }
}

impl Default for DbOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// A batch of database operations applied atomically.
#[repr(C)]
pub enum Batch<K: AsRef<[u8]>, V: AsRef<[u8]>> {
    /// Put a key-value pair with an explicit timestamp.
    Put(K, V, u128),
    /// Delete a key with an explicit timestamp.
    Delete(K, u128),
    /// Put a key-value pair into a namespace with an explicit timestamp.
    PutNs(u64, K, V, u128),
    /// Delete a key from a namespace with an explicit timestamp.
    DeleteNs(u64, K, u128),
}
