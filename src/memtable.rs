// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

use std::{
    collections::Bound,
    hash::RandomState,
    mem::transmute,
    sync::{
        atomic::{
            AtomicBool,
            AtomicU64,
            Ordering::Relaxed,
        },
        Arc,
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
    bounded,
    Sender,
};
use crossbeam_skiplist::{
    map::{
        Entry,
        Range,
    },
    SkipMap,
};
use gxhash::gxhash64;
use parking_lot::Mutex;
use rand::random;
use tracing::instrument;

use crate::{
    errs::CesiumError::{
        self,
        DataExceedsMaximum,
        MemtableIsFrozen,
    },
    keypair::{
        map_key_bound,
        KeyBytes,
        ValueBytes,
    },
    peek::Peekable,
    stats::STATS,
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
                .map(|val| ValueBytes::deserialize_from_memory(val.value().clone())),
        }
    }

    #[instrument(level = "debug")]
    #[inline]
    pub fn put(&self, key: KeyBytes, val: ValueBytes) -> Result<(), CesiumError> {
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
    pub fn put_batch(&self, data: &[(KeyBytes, ValueBytes)]) -> Result<(), CesiumError> {
        // we don't want to write to a frozen memtable
        if self.frozen.load(Relaxed) {
            return Err(MemtableIsFrozen);
        }

        for (key, val) in data.iter() {
            let _key = key.clone().serialize_for_memory();
            let _key_ptr = key.clone().serialize_for_latest();
            let _val = val.serialize_for_memory();
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
        let ranger = self.map.range((_lower, _upper));

        // we need to extend the lifetime of `range` to 'static
        // so the user can hold onto it. as self.map is Arc'd,
        // this won't be deallocated while the iterator exists
        let range = unsafe { transmute(ranger) };

        MemtableIterator::new(range)
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
    inner: Range<'static, Bytes, (Bound<Bytes>, Bound<Bytes>), Bytes, Bytes>,
}

impl MemtableIterator {
    #[instrument(level = "trace")]
    fn new(inner: Range<'static, Bytes, (Bound<Bytes>, Bound<Bytes>), Bytes, Bytes>) -> Self {
        MemtableIterator { inner }
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
        self.inner.next().map(|entry: Entry<'_, Bytes, Bytes>| {
            (
                KeyBytes::deserialize_from_memory(entry.key().clone()),
                ValueBytes::deserialize_from_memory(entry.value().clone()),
            )
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use rand::{
        Rng,
        RngCore,
    };

    use crate::{
        hlc::{
            HybridLogicalClock,
            HLC,
        },
        keypair::{
            KeyBytes,
            ValueBytes,
            DEFAULT_NS,
        },
        memtable::{
            Memtable,
            DEFAULT_MEMTABLE_SIZE_IN_BYTES,
        },
    };

    #[test]
    fn test_memtable_basic() {
        let memtable = Memtable::new(0, DEFAULT_MEMTABLE_SIZE_IN_BYTES);
        let clock = HybridLogicalClock::new();

        let original_key = KeyBytes::new(DEFAULT_NS, Bytes::from("test"), clock.time());
        let original_val = ValueBytes::new(DEFAULT_NS, Bytes::from("value"));
        assert!(memtable
            .put(original_key.clone(), original_val.clone())
            .is_ok());

        let val = memtable.get(original_key.clone());
        assert!(val.is_some());
        assert_eq!(original_val, val.unwrap());
    }

    #[test]
    fn test_memtable_versioning() {
        let memtable = Memtable::new(0, 2 << 23);
        let clock = HybridLogicalClock::new();

        let mut rng = rand::thread_rng();
        let ns = rng.gen();

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
}
