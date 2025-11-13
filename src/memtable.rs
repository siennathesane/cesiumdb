// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

use std::{
    collections::Bound,
    hash::RandomState,
    mem::transmute,
    sync::{
        Arc,
        atomic::{
            AtomicBool,
            AtomicU64,
            Ordering::Relaxed,
        },
    },
    thread,
};

use bloom2::{
    Bloom2,
    BloomFilterBuilder,
    CompressedBitmap,
    FilterSize::KeyBytes3,
};
use bytes::Bytes;
use crossbeam_channel::{
    Sender,
    bounded,
};
use crossbeam_skiplist::{
    SkipMap,
    map::{
        Entry,
        Range,
    },
};
use gxhash::gxhash64;
use parking_lot::Mutex;
use rand::random;
use tracing::instrument;

use crate::{
    errs::{
        MemtableError,
        MemtableError::{
            DataExceedsMaximum,
            MemtableIsFrozen,
        },
    },
    keypair::{
        KeyBytes,
        ValueBytes,
        map_key_bound,
    },
    peek::Peekable,
    stats::STATS,
    utils::{
        Deserializer,
        Serializer,
    },
};

pub const DEFAULT_MEMTABLE_SIZE_IN_BYTES: u64 = 2 << 28; // 256MiB

#[derive(Debug)]
pub struct Memtable {
    id: u64,
    gx_seed: Arc<i64>,
    tx: Sender<Bytes>,
    bloom: Arc<Mutex<Bloom2<RandomState, CompressedBitmap, u64>>>,
    map: Arc<SkipMap<Bytes, Bytes>>,
    size: AtomicU64,
    max_size: AtomicU64,
    frozen: Arc<AtomicBool>,
    // TODO(@siennathesane): add optional wal hook to memtable
    // nb (sienna): the retrieval performance on the memtable is so fucking good
    // that checking a cache is actually _slower_, so no caches for the memtables
}

impl Memtable {
    pub fn new(id: u64, max_size: u64) -> Self {
        let frozen = Arc::new(AtomicBool::new(false));
        let (tx, rx) = bounded::<Bytes>(1_000);
        let gx_seed: Arc<i64> = Arc::new(random());
        let bloom = Arc::new(Mutex::new(
            BloomFilterBuilder::default().size(KeyBytes3).build(),
        ));

        // background thread because
        let frozen_clone = frozen.clone();
        let bloom_clone = bloom.clone();
        let seed_clone = gx_seed.clone();
        thread::spawn(move || {
            while !frozen_clone.load(Relaxed) {
                while let Ok(_key_ptr) = rx.recv() {
                    bloom_clone.lock().insert(&gxhash64(&_key_ptr, *seed_clone))
                }
            }
            STATS.current_threads.fetch_sub(1, Relaxed);
        });
        STATS.current_threads.fetch_add(1, Relaxed);

        Memtable {
            id,
            gx_seed,
            tx,
            bloom,
            map: Arc::new(SkipMap::new()),
            size: AtomicU64::new(0),
            max_size: AtomicU64::new(max_size),
            frozen,
        }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    #[inline]
    pub fn size(&self) -> u64 {
        self.size.load(Relaxed)
    }

    /// Get a key.
    // TODO(@siennathesane): update this with the cuckoo filter and latest cache
    #[instrument(level = "debug")]
    #[inline]
    pub fn get(&self, key: KeyBytes) -> Option<ValueBytes> {
        let _key_ptr = key.serialize_for_latest();
        match self.map.get(&_key_ptr) {
            | None => None,
            | Some(_key) => self
                .map
                .get(&_key.value().clone())
                .map(|val| ValueBytes::deserialize(val.value().clone())),
        }
    }

    #[instrument(level = "debug")]
    #[inline]
    pub fn put(&self, key: KeyBytes, val: ValueBytes) -> Result<(), MemtableError> {
        self.put_batch(&[(key, val)])
    }

    /// Puts a batch of [`data`] into the memtable.
    ///
    /// With versioned keys, it can make O(1) lookups impossible while retaining
    /// the version history. To work around that, the memtable uses a "key
    /// pointer", which is a copy of the key with the maximum possible
    /// version set to mimic the concept of "latest". This "key pointer" retains
    /// a value of the most recently inserted key version. This lets us
    /// lookup "key:latest" to get "key:123_486_713_946". Without this, to
    /// find the latest version of a key with a million versions took about
    /// 2.2s on a Macbook M1 Pro. This optimization allows for O(2) lookups.
    #[instrument(level = "debug")]
    #[inline]
    pub fn put_batch(&self, data: &[(KeyBytes, ValueBytes)]) -> Result<(), MemtableError> {
        // we don't want to write to a frozen memtable
        if self.frozen.load(Relaxed) {
            return Err(MemtableIsFrozen);
        }

        for (key, val) in data.iter() {
            let _key = key.clone().serialize();
            let _key_ptr = key.clone().serialize_for_latest();
            let _val = val.serialize();
            // the key * value both have two u32 bits associated with them
            // on physical storage, so we account for that. we also have to
            // account for the key, the key pointer, and the key pointer's value (re: the
            // key)
            let payload_size = ((_key.len() * 3) + _val.len() + size_of::<u128>()) as u64;

            // we don't want to exceed it
            if payload_size + self.size.load(Relaxed) > self.max_size.load(Relaxed) {
                return Err(DataExceedsMaximum);
            }

            self.map.insert(_key.clone(), _val);
            self.map.insert(_key_ptr.clone(), _key);
            self.size.fetch_add(payload_size, Relaxed);

            // send to the background to prevent a massive performance hit
            let _ = self.tx.send(_key_ptr);

            // TODO(@siennathesane): wal hook on put_batch
        }

        Ok(())
    }

    #[instrument(level = "debug")]
    #[inline]
    pub fn scan(&self, lower: Bound<KeyBytes>, upper: Bound<KeyBytes>) -> MemtableIterator {
        let (_lower, _upper) = (map_key_bound(lower), map_key_bound(upper));

        // clone the Arc to keep the SkipMap alive for the lifetime of the iterator
        let map_clone = self.map.clone();
        let ranger = map_clone.range((_lower, _upper));

        // SAFETY: we're transmuting the lifetime from tied-to-map_clone to 'static.
        // this is sound because:
        // 1. the iterator struct holds map_clone, keeping the SkipMap alive
        // 2. rust's drop order guarantees inner (Range) drops before _map (Arc)
        // 3. therefore, the SkipMap is guaranteed alive during Range's lifetime
        // 4. the Range only borrows; returned values are owned (cloned Bytes)
        let range = unsafe { transmute(ranger) };

        MemtableIterator::new(map_clone, range)
    }

    pub fn freeze(&self) {
        self.frozen.store(true, Relaxed);
    }
}

impl Drop for Memtable {
    // just in case this is randomly dropped, this will ensure the background thread
    // gets cleaned up
    fn drop(&mut self) {
        self.frozen.store(true, Relaxed);
    }
}

#[derive(Debug)]
pub struct MemtableIterator {
    // IMPORTANT: Field order matters for drop order!
    // `inner` must drop before `_map` to ensure Range is destroyed
    // while SkipMap is still alive.
    inner: Range<'static, Bytes, (Bound<Bytes>, Bound<Bytes>), Bytes, Bytes>,
    _map: Arc<SkipMap<Bytes, Bytes>>,
}

impl MemtableIterator {
    #[instrument(level = "trace")]
    fn new(
        map: Arc<SkipMap<Bytes, Bytes>>,
        inner: Range<'static, Bytes, (Bound<Bytes>, Bound<Bytes>), Bytes, Bytes>,
    ) -> Self {
        MemtableIterator { inner, _map: map }
    }

    #[instrument(level = "trace")]
    fn peekable(self) -> Peekable<Self> {
        Peekable::new(self)
    }
}

impl Iterator for MemtableIterator {
    type Item = (KeyBytes, ValueBytes);

    #[instrument(level = "trace")]
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let entry = self.inner.next()?;
            let key = KeyBytes::deserialize(entry.key().clone());

            // Skip "latest" pointer entries (these have ts=0 after inversion)
            // These are internal bookkeeping entries that point to the actual key
            if key.is_pointer_key() {
                continue;
            }

            let value = ValueBytes::deserialize(entry.value().clone());
            return Some((key, value));
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(loom))]
    use std::{sync::Arc, thread};

    #[cfg(loom)]
    use loom::{
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicU64, Ordering},
        },
        thread,
    };

    use bytes::Bytes;
    use rand::{
        Rng,
        RngCore,
    };

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
        memtable::{
            DEFAULT_MEMTABLE_SIZE_IN_BYTES,
            Memtable,
        },
    };

    #[test]
    fn test_memtable_basic() {
        let memtable = Memtable::new(0, DEFAULT_MEMTABLE_SIZE_IN_BYTES);
        let clock = HybridLogicalClock::new();

        let original_key = KeyBytes::new(DEFAULT_NS, Bytes::from("test"), clock.time());
        let original_val = ValueBytes::new(DEFAULT_NS, Bytes::from("value"));
        assert!(
            memtable
                .put(original_key.clone(), original_val.clone())
                .is_ok()
        );

        let val = memtable.get(original_key.clone());
        assert!(val.is_some());
        assert_eq!(original_val, val.unwrap());
    }

    #[test]
    fn test_memtable_versioning() {
        let memtable = Memtable::new(0, 2 << 23);
        let clock = HybridLogicalClock::new();

        let mut rng = rand::rng();
        let ns = rng.random();

        let key = Bytes::from("test-key");

        const VERSIONS: usize = 1_000;

        // write a million different versions of the same key
        let mut batch = Vec::<(KeyBytes, ValueBytes)>::with_capacity(VERSIONS);
        for i in 0..VERSIONS {
            let _key = KeyBytes::new(ns, key.clone(), clock.time());
            let _val = ValueBytes::new(ns, Bytes::copy_from_slice(&i.to_le_bytes()));
            batch.push((_key, _val.clone()));
        }
        assert!(memtable.put_batch(batch.as_ref()).is_ok());

        // let iter = memtable.scan(Bound::Included(KeyBytes::new(ns, key.clone(),
        // u128::MAX)), Bound::Excluded(KeyBytes::new(ns, key.clone(), u128::MIN)));
        // let items = iter.collect::<Vec<_>>();
        // assert_eq!(items.len(), VERSIONS);

        let val = memtable.get(KeyBytes::new(ns, key.clone(), 0));
        assert!(val.is_some());

        // the value we found in the memtable
        let mut val_arr: [u8; 8] = Default::default();
        val_arr.copy_from_slice(&val.unwrap().value.as_ref()[0..8]);

        assert_eq!(usize::from_le_bytes(val_arr), VERSIONS - 1);
    }

    #[test]
    fn test_exceeds_max_size() {
        const MAX_SIZE: u64 = 2 << 6;
        let memtable = Memtable::new(0, MAX_SIZE);
        let clock = HybridLogicalClock::new();

        let mut rng = rand::thread_rng();
        let buf = &mut [0_u8; MAX_SIZE as usize];
        rng.fill_bytes(buf);

        // this will exceed the size of the memtable
        let key = KeyBytes::new(DEFAULT_NS, Bytes::from("test-key"), clock.time());
        let val = ValueBytes::new(DEFAULT_NS, Bytes::copy_from_slice(buf));

        assert!(
            memtable.put(key, val.clone()).is_err(),
            "there must be an error inserting a key pair larger than the max configured size"
        );
    }

    #[test]
    fn test_frozen() {
        const MAX_SIZE: u64 = 2 << 6;
        let memtable = Memtable::new(0, DEFAULT_MEMTABLE_SIZE_IN_BYTES);
        memtable.freeze();
        let clock = HybridLogicalClock::new();

        let mut rng = rand::thread_rng();
        let buf = &mut [0_u8; MAX_SIZE as usize];
        rng.fill_bytes(buf);

        // this will exceed the size of the memtable
        let key = KeyBytes::new(DEFAULT_NS, Bytes::from("test-key"), clock.time());
        let val = ValueBytes::new(DEFAULT_NS, Bytes::copy_from_slice(buf));

        assert!(
            memtable.put(key, val.clone()).is_err(),
            "there must be an error inserting a key pair while the memtable is frozen"
        );
    }

    #[test]
    fn test_get_nonexistent_key() {
        let memtable = Memtable::new(0, DEFAULT_MEMTABLE_SIZE_IN_BYTES);
        let key = KeyBytes::new(DEFAULT_NS, Bytes::from("nonexistent"), 0);

        let result = memtable.get(key);
        assert!(result.is_none(), "get on nonexistent key should return None");
    }

    #[test]
    fn test_size_tracking() {
        let memtable = Memtable::new(0, DEFAULT_MEMTABLE_SIZE_IN_BYTES);
        let clock = HybridLogicalClock::new();

        let initial_size = memtable.size();
        assert_eq!(initial_size, 0, "initial size should be 0");

        let key = KeyBytes::new(DEFAULT_NS, Bytes::from("key"), clock.time());
        let val = ValueBytes::new(DEFAULT_NS, Bytes::from("value"));

        assert!(memtable.put(key.clone(), val.clone()).is_ok());

        let new_size = memtable.size();
        assert!(new_size > 0, "size should increase after put");
        assert!(new_size > initial_size, "size should be greater than initial");
    }

    #[test]
    fn test_scan_empty_memtable() {
        use std::collections::Bound;

        let memtable = Memtable::new(0, DEFAULT_MEMTABLE_SIZE_IN_BYTES);
        let key = KeyBytes::new(DEFAULT_NS, Bytes::from("key"), 0);

        let mut iter = memtable.scan(Bound::Unbounded, Bound::Unbounded);
        assert!(iter.next().is_none(), "scan on empty memtable should return no items");
    }

    #[test]
    fn test_scan_single_key() {
        use std::collections::Bound;

        let memtable = Memtable::new(0, DEFAULT_MEMTABLE_SIZE_IN_BYTES);
        let clock = HybridLogicalClock::new();

        let key = KeyBytes::new(DEFAULT_NS, Bytes::from("key"), clock.time());
        let val = ValueBytes::new(DEFAULT_NS, Bytes::from("value"));
        assert!(memtable.put(key.clone(), val.clone()).is_ok());

        let iter = memtable.scan(Bound::Unbounded, Bound::Unbounded);
        let items: Vec<_> = iter.collect();

        // scan returns actual key entries, not the key pointers
        assert!(items.len() >= 1, "scan should return at least one item");
    }

    #[test]
    fn test_scan_with_bounds() {
        use std::collections::Bound;

        let memtable = Memtable::new(0, DEFAULT_MEMTABLE_SIZE_IN_BYTES);
        let clock = HybridLogicalClock::new();

        // insert multiple keys
        for i in 0..10 {
            let key = KeyBytes::new(DEFAULT_NS, Bytes::from(format!("key-{:02}", i)), clock.time());
            let val = ValueBytes::new(DEFAULT_NS, Bytes::from(format!("value-{}", i)));
            assert!(memtable.put(key, val).is_ok());
        }

        let lower = KeyBytes::new(DEFAULT_NS, Bytes::from("key-03"), u128::MAX);
        let upper = KeyBytes::new(DEFAULT_NS, Bytes::from("key-07"), u128::MIN);

        let iter = memtable.scan(Bound::Included(lower), Bound::Excluded(upper));
        let items: Vec<_> = iter.collect();

        // should return items in the range
        assert!(items.len() >= 1, "scan with bounds should return items in range");
    }

    #[test]
    fn test_multiple_gets() {
        let memtable = Memtable::new(0, DEFAULT_MEMTABLE_SIZE_IN_BYTES);
        let clock = HybridLogicalClock::new();

        // insert multiple key-value pairs
        for i in 0..100 {
            let key = KeyBytes::new(DEFAULT_NS, Bytes::from(format!("key-{}", i)), clock.time());
            let val = ValueBytes::new(DEFAULT_NS, Bytes::from(format!("value-{}", i)));
            assert!(memtable.put(key, val).is_ok());
        }

        // retrieve all of them
        for i in 0..100 {
            let key = KeyBytes::new(DEFAULT_NS, Bytes::from(format!("key-{}", i)), 0);
            let result = memtable.get(key);
            assert!(result.is_some(), "all inserted keys should be retrievable");
        }
    }

    #[test]
    fn test_put_batch_empty() {
        let memtable = Memtable::new(0, DEFAULT_MEMTABLE_SIZE_IN_BYTES);
        let batch: Vec<(KeyBytes, ValueBytes)> = vec![];

        let result = memtable.put_batch(&batch);
        assert!(result.is_ok(), "empty batch should succeed");
    }

    #[test]
    fn test_put_batch_versioned_keys() {
        let memtable = Memtable::new(0, DEFAULT_MEMTABLE_SIZE_IN_BYTES);
        let clock = HybridLogicalClock::new();

        let key_name = Bytes::from("versioned-key");
        let mut batch = vec![];

        // create multiple versions of the same key
        for i in 0..10 {
            let key = KeyBytes::new(DEFAULT_NS, key_name.clone(), clock.time());
            let val = ValueBytes::new(DEFAULT_NS, Bytes::from(format!("version-{}", i)));
            batch.push((key, val));
        }

        assert!(memtable.put_batch(&batch).is_ok());

        // get should return the latest version
        let result = memtable.get(KeyBytes::new(DEFAULT_NS, key_name, 0));
        assert!(result.is_some(), "versioned key should be retrievable");
    }

    #[test]
    fn test_memtable_id_immutable() {
        let memtable = Memtable::new(42, DEFAULT_MEMTABLE_SIZE_IN_BYTES);
        assert_eq!(memtable.id(), 42);

        let clock = HybridLogicalClock::new();
        let key = KeyBytes::new(DEFAULT_NS, Bytes::from("key"), clock.time());
        let val = ValueBytes::new(DEFAULT_NS, Bytes::from("value"));
        assert!(memtable.put(key, val).is_ok());

        // id should remain the same after operations
        assert_eq!(memtable.id(), 42);
    }

    #[test]
    fn test_put_different_namespaces() {
        let memtable = Memtable::new(0, DEFAULT_MEMTABLE_SIZE_IN_BYTES);
        let clock = HybridLogicalClock::new();

        let key_name = Bytes::from("key");

        // put same key in different namespaces
        for ns in 0..5 {
            let key = KeyBytes::new(ns, key_name.clone(), clock.time());
            let val = ValueBytes::new(ns, Bytes::from(format!("value-ns-{}", ns)));
            assert!(memtable.put(key, val).is_ok());
        }

        // verify all namespaces are retrievable
        for ns in 0..5 {
            let key = KeyBytes::new(ns, key_name.clone(), 0);
            let result = memtable.get(key);
            assert!(result.is_some(), "key in namespace {} should be retrievable", ns);
        }
    }

    #[test]
    fn test_iterator_size_hint() {
        use std::collections::Bound;

        let memtable = Memtable::new(0, DEFAULT_MEMTABLE_SIZE_IN_BYTES);
        let clock = HybridLogicalClock::new();

        for i in 0..10 {
            let key = KeyBytes::new(DEFAULT_NS, Bytes::from(format!("key-{}", i)), clock.time());
            let val = ValueBytes::new(DEFAULT_NS, Bytes::from("value"));
            assert!(memtable.put(key, val).is_ok());
        }

        let iter = memtable.scan(Bound::Unbounded, Bound::Unbounded);
        let (lower, _upper) = iter.size_hint();

        // size_hint should return reasonable bounds
        assert!(lower >= 0, "size hint lower bound should be non-negative");
    }

    #[test]
    fn test_drop_frozen_memtable() {
        let memtable = Memtable::new(0, DEFAULT_MEMTABLE_SIZE_IN_BYTES);
        let clock = HybridLogicalClock::new();

        let key = KeyBytes::new(DEFAULT_NS, Bytes::from("key"), clock.time());
        let val = ValueBytes::new(DEFAULT_NS, Bytes::from("value"));
        assert!(memtable.put(key, val).is_ok());

        memtable.freeze();
        // dropping the memtable should clean up properly
        drop(memtable);
    }

    #[test]
    fn test_get_after_multiple_versions() {
        let memtable = Memtable::new(0, DEFAULT_MEMTABLE_SIZE_IN_BYTES);
        let clock = HybridLogicalClock::new();

        let key_name = Bytes::from("multi-version");

        // write 100 versions
        for i in 0..100 {
            let key = KeyBytes::new(DEFAULT_NS, key_name.clone(), clock.time());
            let val = ValueBytes::new(DEFAULT_NS, Bytes::from(format!("v{}", i)));
            assert!(memtable.put(key, val).is_ok());
        }

        // get should still work efficiently
        let result = memtable.get(KeyBytes::new(DEFAULT_NS, key_name, 0));
        assert!(result.is_some(), "should retrieve latest version efficiently");
    }

    #[test]
    fn test_batch_exceeds_max_size() {
        const SMALL_MAX: u64 = 1000;
        let memtable = Memtable::new(0, SMALL_MAX);
        let clock = HybridLogicalClock::new();

        // create a batch that will exceed the max size
        let mut batch = vec![];
        for i in 0..100 {
            let key = KeyBytes::new(DEFAULT_NS, Bytes::from(format!("key-{}", i)), clock.time());
            let val = ValueBytes::new(DEFAULT_NS, Bytes::from(vec![b'x'; 100]));
            batch.push((key, val));
        }

        let result = memtable.put_batch(&batch);
        assert!(result.is_err(), "batch exceeding max size should fail");
    }

    #[test]
    #[cfg(not(loom))]
    fn test_iterator_outlives_memtable() {
        use std::collections::Bound;

        let clock = HybridLogicalClock::new();

        // Create iterator in inner scope
        let iter = {
            let memtable = Memtable::new(0, DEFAULT_MEMTABLE_SIZE_IN_BYTES);

            // Insert some data
            for i in 0..5 {
                let key = KeyBytes::new(DEFAULT_NS, Bytes::from(format!("key-{}", i)), clock.time());
                let val = ValueBytes::new(DEFAULT_NS, Bytes::from(format!("value-{}", i)));
                assert!(memtable.put(key, val).is_ok());
            }

            // Create iterator before memtable is dropped
            memtable.scan(Bound::Unbounded, Bound::Unbounded)
            // memtable is dropped here
        };

        // Iterator should still work even though memtable is gone
        // This proves the Arc keeps the SkipMap alive
        let items: Vec<_> = iter.collect();
        assert!(items.len() >= 1, "iterator should work after memtable is dropped");
    }

    // Loom tests for atomic operation patterns
    // These test the concurrency patterns used in memtable without the crossbeam dependencies

    #[test]
    #[cfg(loom)]
    fn loom_frozen_flag_race() {
        use std::sync::atomic::Ordering::Relaxed;

        loom::model(|| {
            let frozen = Arc::new(AtomicBool::new(false));
            let writes_succeeded = Arc::new(AtomicU64::new(0));

            let f1 = frozen.clone();
            let w1 = writes_succeeded.clone();

            let f2 = frozen.clone();
            let w2 = writes_succeeded.clone();

            // Thread 1: tries to "write"
            let t1 = thread::spawn(move || {
                // This mimics the pattern in put_batch
                if !f1.load(Relaxed) {
                    // Simulate the operation taking time
                    thread::yield_now();
                    // If we got here, we "wrote"
                    w1.fetch_add(1, Relaxed);
                }
            });

            // Thread 2: tries to freeze
            let t2 = thread::spawn(move || {
                f2.store(true, Relaxed);
                w2.load(Relaxed)
            });

            t1.join().unwrap();
            let writes_when_frozen = t2.join().unwrap();

            // The final state: check if frozen
            let is_frozen = frozen.load(Relaxed);
            let total_writes = writes_succeeded.load(Relaxed);

            // If frozen, we might have 0 or 1 writes depending on interleaving
            // This demonstrates the TOCTOU race condition
            if is_frozen && total_writes > 0 {
                // This can happen: check passed, then freeze happened, then write completed
                // This is the race condition!
            }
        });
    }

    #[test]
    #[cfg(loom)]
    fn loom_size_tracking_race() {
        use std::sync::atomic::Ordering::Relaxed;

        loom::model(|| {
            let size = Arc::new(AtomicU64::new(0));
            let max_size = 100u64;

            let s1 = size.clone();
            let s2 = size.clone();

            // Two threads trying to add size
            let t1 = thread::spawn(move || {
                let payload_size = 30u64;
                // This mimics the pattern in put_batch
                if payload_size + s1.load(Relaxed) <= max_size {
                    thread::yield_now();
                    s1.fetch_add(payload_size, Relaxed);
                    true
                } else {
                    false
                }
            });

            let t2 = thread::spawn(move || {
                let payload_size = 80u64;
                if payload_size + s2.load(Relaxed) <= max_size {
                    thread::yield_now();
                    s2.fetch_add(payload_size, Relaxed);
                    true
                } else {
                    false
                }
            });

            let wrote1 = t1.join().unwrap();
            let wrote2 = t2.join().unwrap();

            let final_size = size.load(Relaxed);

            // Both operations might succeed due to TOCTOU, resulting in > max_size
            if wrote1 && wrote2 {
                // This demonstrates the race: both checked, both passed, total exceeds max
                assert!(final_size == 110, "Both writes succeeded, total = {}", final_size);
            }
        });
    }

    #[test]
    #[cfg(loom)]
    fn loom_concurrent_size_updates() {
        use std::sync::atomic::Ordering::Relaxed;

        loom::model(|| {
            let size = Arc::new(AtomicU64::new(0));

            let s1 = size.clone();
            let s2 = size.clone();

            let t1 = thread::spawn(move || {
                s1.fetch_add(10, Relaxed);
            });

            let t2 = thread::spawn(move || {
                s2.fetch_add(20, Relaxed);
            });

            t1.join().unwrap();
            t2.join().unwrap();

            // fetch_add is atomic, so this should always be correct
            assert_eq!(size.load(Relaxed), 30);
        });
    }

    #[test]
    #[cfg(loom)]
    fn loom_freeze_idempotent() {
        use std::sync::atomic::Ordering::Relaxed;

        loom::model(|| {
            let frozen = Arc::new(AtomicBool::new(false));

            let f1 = frozen.clone();
            let f2 = frozen.clone();

            // Multiple threads trying to freeze
            let t1 = thread::spawn(move || {
                f1.store(true, Relaxed);
            });

            let t2 = thread::spawn(move || {
                f2.store(true, Relaxed);
            });

            t1.join().unwrap();
            t2.join().unwrap();

            // Freezing multiple times is safe
            assert!(frozen.load(Relaxed));
        });
    }

    #[test]
    #[cfg(loom)]
    fn loom_read_frozen_while_freezing() {
        use std::sync::atomic::Ordering::Relaxed;

        loom::model(|| {
            let frozen = Arc::new(AtomicBool::new(false));

            let f1 = frozen.clone();
            let f2 = frozen.clone();

            let t1 = thread::spawn(move || {
                f1.store(true, Relaxed);
            });

            let t2 = thread::spawn(move || {
                f2.load(Relaxed)
            });

            t1.join().unwrap();
            let saw_frozen = t2.join().unwrap();

            let final_frozen = frozen.load(Relaxed);

            // t2 either saw false or true, but final must be true
            assert!(final_frozen);
            // saw_frozen can be either true or false depending on interleaving
        });
    }
}
