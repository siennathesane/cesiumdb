// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

#![feature(integer_atomics)]
#![allow(dead_code)] // TODO(@siennathesane): remove before release
#![allow(unused)] // TODO(@siennathesane): remove before release
#[cfg(not(unix))]
compile_error!("this crate won't work on non-unix platforms, sorry");
#[cfg(not(target_pointer_width = "64"))]
compile_warn!("this crate is not tested on 32-bit systems");

#[allow(unused)]
use std::sync::Arc;

use bytes::Bytes;
use mimalloc::MiMalloc;
use parking_lot::Mutex;

use crate::{
    errs::CesiumError,
    hlc::{
        HybridLogicalClock,
        HLC,
    },
    keypair::{
        KeyBytes,
        ValueBytes,
        DEFAULT_NS,
    },
    state::{
        DbStorageBuilder,
        DbStorageState,
    },
    Batch::{
        Delete,
        DeleteNs,
        Put,
        PutNs,
    },
};

#[cfg(not(miri))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub mod errs;
pub mod hlc;
pub mod keypair;
pub mod memtable;
pub mod merge;
pub mod peek;
pub(crate) mod state;
mod stats;
mod sbtable;
mod utils;
mod block;
mod segment_writer;
pub mod fs;

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
        self.put_ns(DEFAULT_NS, key, key)
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

    /// Close the database. This drops all associated resources and the handle
    /// will no longer be valid.
    pub fn close(&self) -> Result<(), CesiumError> {
        todo!()
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

    pub fn build(&self) -> Arc<Db> {
        let state = DbStorageBuilder::new()
            .block_size(self.engine_opts.block_size)
            .target_sst_size(self.engine_opts.target_sst_size)
            .num_memtable_limit(self.engine_opts.num_memtable_limit)
            .build();

        let inner = DbInner { state };

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
}

impl DbInner {
    fn get(&self, key: KeyBytes) -> Result<Option<ValueBytes>, CesiumError> {
        // check the current memtable
        {
            let guard = self.state.lock();
            let val = guard.current_memtable().get(key);
            if let Some(val) = val {
                return Ok(Some(val));
            }
        }
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
                    ValueBytes::new(*ns, Bytes::new()),
                )),
                | _ => None, // filter out invalid enums
            })
            .collect::<Vec<_>>();
        {
            let guard = self.state.lock();
            let mtable = guard.current_memtable();
            mtable.put_batch(_batch.as_ref())
        }
    }

    fn sync(&self) -> Result<(), CesiumError> {
        todo!()
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

    const MAX_KEYS: u64 = 100_000;

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
            assert!(guard.current_memtable().size() > keypair_size as u64, "the memtable must be bigger than the keypair size to ensure the keys are actually stored");
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
        for batch_size in [1, 10, 100, 1000].iter() {
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
                assert!(guard.current_memtable().size() > keypair_size as u64, "the memtable must be bigger than the keypair size to ensure the keys are actually stored");
            }

            println!(
                "batch size: {}, keypair_size: {}",
                batch.len(),
                keypair_size
            );
        }
    }
}
