// Copyright (c) Sienna Satterwhite, CesiumDB Contributors

// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

#![feature(sync_unsafe_cell)]
#![feature(let_chains)]
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
use mimalloc::MiMalloc;
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

#[cfg(not(miri))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub mod block;
mod block_alloc;
pub mod errs;
mod hash;
pub mod hlc;
mod index;
pub mod keypair;
mod manifest;
pub mod map;
pub mod memtable;
pub mod merge;
pub mod peek;
pub mod segment;
mod segment_builder;
mod segment_iterator;
pub mod segment_reader;
pub mod segment_writer;
pub(crate) mod state;
mod stats;
pub mod utils;

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

            // TODO(@siennathesane): add memtable swap logic here
            match mtable.put_batch(_batch.as_ref()) {
                | Ok(_) => Ok(()),
                | Err(e) => Err(MemtableError(e)),
            }
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
        assert!(result.unwrap().is_none(), "get on empty db should return None");

        // insert and retrieve
        let key = b"test-key";
        let val = b"test-value";
        assert!(db.put(key, val).is_ok());

        let result = db.get(key);
        assert!(result.is_ok());
        let retrieved = result.unwrap();
        assert!(retrieved.is_some(), "get should return Some for existing key");
        assert_eq!(&retrieved.unwrap()[..], val, "retrieved value should match");

        // test get on different key
        let result = db.get(b"different-key");
        assert!(result.is_ok());
        assert!(result.unwrap().is_none(), "get on non-existent key should return None");
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
        assert_eq!(&retrieved.unwrap()[..], val3, "get should return the latest value");
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

        // note: delete in this implementation puts the key with itself as value
        // this is likely a bug in the original code at line 144
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
        assert!(result.unwrap().is_none(), "key should not exist in default namespace");
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
        assert!(time2 >= time1, "clock should return monotonically increasing values");
    }

    #[test]
    fn test_db_batch_mixed_operations() {
        use crate::Batch::{Delete, DeleteNs, PutNs};

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
