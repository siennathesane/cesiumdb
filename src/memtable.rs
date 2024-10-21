use std::{
    collections::Bound,
    hash::RandomState,
    mem::transmute,
    sync::{
        atomic::{
            AtomicBool,
            AtomicUsize,
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
    errs::Error,
    keypair::{
        map_key_bound,
        KeyBytes,
        ValueBytes,
    },
    stats::STATS,
};

#[derive(Debug)]
pub struct Memtable {
    id: usize,
    gx_seed: Arc<i64>,
    tx: Sender<Bytes>,
    bloom: Arc<Mutex<Bloom2<RandomState, CompressedBitmap, u64>>>,
    map: Arc<SkipMap<Bytes, Bytes>>,
    approx_size: AtomicUsize,
    frozen: Arc<AtomicBool>,
    // TODO(@siennathesane): add optional wal hook to memtable
    // TODO(@siennathesane): add cache and performance test. test if
}

impl Memtable {
    pub fn new(id: usize) -> Self {
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
        });
        STATS.current_threads.fetch_add(1, Relaxed);

        Memtable {
            id,
            gx_seed,
            tx,
            bloom,
            map: Arc::new(SkipMap::new()),
            approx_size: AtomicUsize::new(0),
            frozen,
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn size(&self) -> usize {
        self.approx_size.load(Relaxed)
    }

    /// Get a key.
    // TODO(@siennathesane): update this with the cuckoo filter and latest cache
    #[instrument(level = "debug")]
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
    pub fn put(&self, key: KeyBytes, val: ValueBytes) -> Result<(), Error> {
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
    pub fn put_batch(&self, data: &[(KeyBytes, ValueBytes)]) -> Result<(), Error> {
        for (key, val) in data.iter() {
            let _key = key.clone().serialize_for_memory();
            let _key_ptr = key.clone().serialize_for_latest();
            let _val = val.serialize_for_memory();
            // the key * value both have two u32 bits associated with them
            // on physical storage, so we account for that. we also have to
            // account for the key, the key pointer, and the key pointer's value (re: the
            // key)
            let estimated_size = (_key.len() * 3) + _val.len() + size_of::<u128>();

            self.map.insert(_key.clone(), _val);
            self.map.insert(_key_ptr.clone(), _key);
            self.approx_size.fetch_add(estimated_size, Relaxed);

            // send to the background to prevent a massive performance hit
            let _ = self.tx.send(_key_ptr);

            // TODO(@siennathesane): wal hook on put_batch
        }

        Ok(())
    }

    #[instrument(level = "debug")]
    pub fn scan(&self, lower: Bound<KeyBytes>, upper: Bound<KeyBytes>) -> MemtableIterator {
        let (_lower, _upper) = (map_key_bound(lower), map_key_bound(upper));
        let ranger = self.map.range((_lower, _upper));

        // we need to extend the lifetime of `range` to 'static
        // so the user can hold onto it. as self.map is Arc'd,
        // this won't be deallocated while the iterator exists
        let range = unsafe { transmute(ranger) };

        MemtableIterator::new(range)
    }
}

impl Drop for Memtable {
    fn drop(&mut self) {
        self.frozen.store(false, Relaxed);
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
}

impl Iterator for MemtableIterator {
    type Item = (KeyBytes, ValueBytes);

    #[instrument(level = "trace")]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|entry: Entry<'_, Bytes, Bytes>| {
            (
                KeyBytes::deserialize_from_memory(entry.key().clone()),
                ValueBytes::deserialize_from_memory(entry.value().clone()),
            )
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use rand::Rng;

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
        memtable::Memtable,
    };

    #[test]
    fn test_memtable_basic() {
        let memtable = Memtable::new(0);
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
        let memtable = Memtable::new(0);
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
}
