// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

use std::{
    cmp::{
        Ordering,
        Reverse,
    },
    collections::Bound,
    fmt::Debug,
};

use bytes::{
    BufMut,
    Bytes,
    BytesMut,
};
use crc32fast::Hasher;
use tracing::instrument;

use crate::utils::{
    Deserializer,
    Serializer,
};

pub const DEFAULT_NS: u64 = 0;

#[instrument]
pub fn map_key_bound(bound: Bound<KeyBytes>) -> Bound<Bytes> {
    match bound {
        | Bound::Included(x) => {
            let mem_key = x.serialize();
            Bound::Included(mem_key)
        },
        | Bound::Excluded(x) => {
            let mem_key = x.serialize();
            Bound::Excluded(mem_key)
        },
        | Bound::Unbounded => Bound::Unbounded,
    }
}

#[derive(Debug, Eq, Clone, Copy, PartialEq)]
pub struct Key<T: AsRef<[u8]>> {
    /// namespace
    ns: u64,
    /// the actual key
    key: T,
    /// timestamp
    ts: u128,
}

pub type KeyBytes = Key<Bytes>;

impl Key<Bytes> {
    pub fn new(ns: u64, key: Bytes, ts: u128) -> Self {
        Key { ns, key, ts }
    }

    pub fn set_key(&mut self, val: Bytes) {
        self.key = val;
    }

    pub fn key(&self) -> &Bytes {
        &self.key
    }

    pub fn key_len(&self) -> usize {
        self.key.as_ref().len()
    }

    pub fn raw_len(&self) -> usize {
        self.key.as_ref().len() + size_of::<u64>() + size_of::<u128>()
    }

    pub fn is_empty(&self) -> bool {
        self.key.as_ref().is_empty()
    }

    pub fn set_ts(&mut self, ts: u128) {
        self.ts = ts;
    }

    pub fn ts(&self) -> u128 {
        self.ts
    }

    /// Returns true if this is a "latest" pointer key (timestamp inverts to 0)
    pub fn is_pointer_key(&self) -> bool {
        self.ts == 0
    }

    pub fn set_ns(&mut self, ns: u64) {
        self.ns = ns;
    }

    pub fn ns(&self) -> u64 {
        self.ns
    }

    pub fn as_bytes(&self) -> Bytes {
        self.key.clone()
    }

    #[instrument(level = "trace")]
    #[inline]
    pub fn serialize_for_latest(&self) -> Bytes {
        let original = self.serialize();
        let mut bytes = BytesMut::from(original.as_ref());
        for idx in 0..size_of::<u128>() {
            bytes[original.len() - idx - 1] = 0xff;
        }
        bytes.freeze()
    }
}

impl Serializer for Key<Bytes> {
    #[instrument(level = "trace")]
    #[inline]
    fn serialize(&self) -> Bytes {
        let mut bytes =
            BytesMut::with_capacity(size_of::<u64>() + self.key.as_ref().len() + size_of::<u128>());

        // this is the serialized key
        bytes.put_u64_le(self.ns);
        bytes.put_slice(self.key.as_ref());
        bytes.put_u128_le(u128::MAX - self.ts);

        bytes.freeze()
    }
}

impl Deserializer for Key<Bytes> {
    #[instrument(level = "trace")]
    #[inline]
    fn deserialize(slice: Bytes) -> Self {
        let mut ns_arr = [0u8; 8];
        ns_arr.copy_from_slice(&slice[0..8]);

        let mut ts_arr = [0u8; 16];
        ts_arr.copy_from_slice(&slice[slice.len() - 16..]);

        KeyBytes {
            ns: u64::from_le_bytes(ns_arr),
            key: Bytes::copy_from_slice(&slice[8..slice.len() - 16]),
            ts: u128::MAX - u128::from_le_bytes(ts_arr),
        }
    }
}

impl<T: AsRef<[u8]>> AsRef<[u8]> for Key<T> {
    fn as_ref(&self) -> &[u8] {
        self.key.as_ref()
    }
}

impl Default for Key<Bytes> {
    fn default() -> Self {
        KeyBytes {
            ns: 0,
            key: Bytes::default(),
            ts: 0,
        }
    }
}

impl<T: AsRef<[u8]> + PartialOrd> PartialOrd for Key<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (self.ns, self.key.as_ref(), Reverse(self.ts)).partial_cmp(&(
            other.ns,
            other.key.as_ref(),
            Reverse(other.ts),
        ))
    }
}

impl<T: AsRef<[u8]> + Ord> Ord for Key<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.ns, self.key.as_ref(), Reverse(self.ts)).cmp(&(
            other.ns,
            other.key.as_ref(),
            Reverse(other.ts),
        ))
    }
}

impl From<Bytes> for Key<Bytes> {
    fn from(val: Bytes) -> Self {
        let mut ns_arr = [0u8; 8];
        ns_arr.copy_from_slice(&val[0..8]);

        let mut ts_arr = [0u8; 16];
        ts_arr.copy_from_slice(&val[val.len() - 16..]);

        Key {
            ns: u64::from_le_bytes(ns_arr),
            key: Bytes::copy_from_slice(&val[16..val.len() - 8]),
            ts: u128::MAX - u128::from_le_bytes(ts_arr),
        }
    }
}

#[derive(Debug, Eq, Clone, Copy, PartialEq)]
pub struct Value<T: AsRef<[u8]>> {
    pub ns: u64,
    pub value: T,
    pub tombstone: bool,
}

pub type ValueBytes = Value<Bytes>;

impl ValueBytes {
    pub fn new(ns: u64, val: Bytes) -> Self {
        ValueBytes {
            ns,
            value: val,
            tombstone: false,
        }
    }

    pub fn new_tombstone(ns: u64) -> Self {
        ValueBytes {
            ns,
            value: Bytes::new(),
            tombstone: true,
        }
    }

    pub fn is_tombstone(&self) -> bool {
        self.tombstone
    }

    pub fn set_ns(&mut self, ns: u64) {
        self.ns = ns
    }

    pub fn ns(&self) -> u64 {
        self.ns
    }

    pub fn from_slice(ns: u64, slice: &[u8]) -> Self {
        ValueBytes {
            ns,
            value: Bytes::copy_from_slice(slice),
            tombstone: false,
        }
    }

    #[instrument(level = "trace")]
    pub fn deserialize(bytes: Bytes) -> Self {
        let mut ns_arr = [0u8; 8];
        ns_arr.copy_from_slice(&bytes[0..8]);

        let tombstone = bytes[8] != 0;

        ValueBytes {
            ns: u64::from_le_bytes(ns_arr),
            tombstone,
            value: Bytes::copy_from_slice(&bytes[9..]),
        }
    }


    #[instrument(level = "trace")]
    #[inline]
    pub fn serialize(&self) -> Bytes {
        // namespace + tombstone flag + payload
        let len = size_of::<u64>() + size_of::<u8>() + self.value.as_ref().len();

        let mut buf = BytesMut::with_capacity(len);
        buf.put_u64_le(self.ns);
        buf.put_u8(if self.tombstone { 1 } else { 0 });
        buf.put_slice(self.value.as_ref());

        buf.freeze()
    }

    pub fn as_bytes(&self) -> Bytes {
        self.value.clone()
    }
}

impl AsRef<[u8]> for ValueBytes {
    fn as_ref(&self) -> &[u8] {
        self.value.as_ref()
    }
}

impl Default for ValueBytes {
    fn default() -> Self {
        ValueBytes {
            ns: 0,
            value: Bytes::default(),
            tombstone: false,
        }
    }
}

impl<T: AsRef<[u8]> + PartialOrd> PartialOrd for Value<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (self.ns, self.value.as_ref()).partial_cmp(&(other.ns, other.value.as_ref()))
    }
}

impl<T: AsRef<[u8]> + Ord> Ord for Value<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.ns, self.value.as_ref()).cmp(&(other.ns, other.value.as_ref()))
    }
}

impl From<Bytes> for Value<Bytes> {
    fn from(val: Bytes) -> Self {
        let mut ns_arr = [0u8; 8];
        ns_arr.copy_from_slice(&val[0..8]);

        let tombstone = val[8] != 0;

        Value {
            ns: u64::from_le_bytes(ns_arr),
            tombstone,
            value: Bytes::copy_from_slice(&val[9..]),
        }
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use crate::{
        keypair::{
            KeyBytes,
            ValueBytes,
        },
        utils::{
            Deserializer,
            Serializer,
        },
    };

    #[test]
    fn test_key_serialization() {
        let key = KeyBytes {
            ns: 0,
            key: Bytes::from("test"),
            ts: 0,
        };
        let memory_serialized = key.serialize();

        // 8 + 4 + 16
        // ns + payload + ts
        assert_eq!(memory_serialized.clone().len(), 28);

        let de_key = KeyBytes::deserialize(memory_serialized.clone());
        assert_eq!(key, de_key);

        let mut latest_key = de_key.clone();
        latest_key.set_ts(1234456773567);
        let latest_serialized = latest_key.serialize_for_latest();

        assert_eq!(
            memory_serialized.clone(),
            latest_serialized.clone(),
            "latest & original key must be the same"
        );
    }

    #[test]
    fn test_value_serialization() {
        let val = ValueBytes::new(0, Bytes::from("test-value"));
        let serialized = val.serialize();
        // namespace (8) + tombstone flag (1) + value (10) = 19
        assert_eq!(serialized.len(), 19);

        let de_val = ValueBytes::deserialize(serialized);
        assert_eq!(val, de_val);
    }

    #[test]
    fn test_key_ordering_by_namespace() {
        let key1 = KeyBytes::new(0, Bytes::from("key"), 100);
        let key2 = KeyBytes::new(1, Bytes::from("key"), 100);

        assert!(key1 < key2, "keys should be ordered by namespace first");
    }

    #[test]
    fn test_key_ordering_by_key_bytes() {
        let key1 = KeyBytes::new(0, Bytes::from("aaa"), 100);
        let key2 = KeyBytes::new(0, Bytes::from("bbb"), 100);

        assert!(key1 < key2, "keys in same namespace should be ordered by key bytes");
    }

    #[test]
    fn test_key_ordering_by_timestamp() {
        let key1 = KeyBytes::new(0, Bytes::from("key"), 200);
        let key2 = KeyBytes::new(0, Bytes::from("key"), 100);

        // with same ns and key, newer timestamp (200) should come first (Reverse ordering)
        assert!(key1 < key2, "keys with same ns and key should be ordered by timestamp in reverse");
    }

    #[test]
    fn test_key_ordering_complex() {
        // test full ordering: ns, then key, then reverse timestamp
        let mut keys = vec![
            KeyBytes::new(1, Bytes::from("zzz"), 100),
            KeyBytes::new(0, Bytes::from("bbb"), 100),
            KeyBytes::new(0, Bytes::from("aaa"), 200),
            KeyBytes::new(0, Bytes::from("aaa"), 100),
            KeyBytes::new(1, Bytes::from("aaa"), 100),
        ];

        keys.sort();

        // expected order:
        // (0, aaa, 200) - ns=0, key=aaa, newest timestamp
        // (0, aaa, 100) - ns=0, key=aaa, older timestamp
        // (0, bbb, 100) - ns=0, key=bbb
        // (1, aaa, 100) - ns=1, key=aaa
        // (1, zzz, 100) - ns=1, key=zzz

        assert_eq!(keys[0].ns(), 0);
        assert_eq!(keys[0].as_bytes(), Bytes::from("aaa"));
        assert_eq!(keys[0].ts(), 200);

        assert_eq!(keys[1].ns(), 0);
        assert_eq!(keys[1].as_bytes(), Bytes::from("aaa"));
        assert_eq!(keys[1].ts(), 100);

        assert_eq!(keys[2].ns(), 0);
        assert_eq!(keys[2].as_bytes(), Bytes::from("bbb"));

        assert_eq!(keys[3].ns(), 1);
        assert_eq!(keys[3].as_bytes(), Bytes::from("aaa"));

        assert_eq!(keys[4].ns(), 1);
        assert_eq!(keys[4].as_bytes(), Bytes::from("zzz"));
    }

    #[test]
    fn test_key_empty() {
        let key = KeyBytes::new(0, Bytes::new(), 0);
        assert!(key.is_empty());
        assert_eq!(key.key_len(), 0);

        let serialized = key.serialize();
        let deserialized = KeyBytes::deserialize(serialized);
        assert!(deserialized.is_empty());
    }

    #[test]
    fn test_key_large() {
        let large_key = vec![b'k'; 10000];
        let key = KeyBytes::new(0, Bytes::from(large_key.clone()), 999999);

        assert_eq!(key.key_len(), 10000);
        assert_eq!(key.raw_len(), 10000 + size_of::<u64>() + size_of::<u128>());

        let serialized = key.serialize();
        let deserialized = KeyBytes::deserialize(serialized);

        assert_eq!(deserialized.key_len(), 10000);
        assert_eq!(deserialized.as_bytes().len(), 10000);
    }

    #[test]
    fn test_value_empty() {
        let val = ValueBytes::new(0, Bytes::new());
        assert_eq!(val.as_bytes().len(), 0);

        let serialized = val.serialize();
        let deserialized = ValueBytes::deserialize(serialized);
        assert_eq!(deserialized.as_bytes().len(), 0);
    }

    #[test]
    fn test_value_large() {
        let large_value = vec![b'v'; 100000];
        let val = ValueBytes::new(42, Bytes::from(large_value.clone()));

        assert_eq!(val.as_bytes().len(), 100000);

        let serialized = val.serialize();
        let deserialized = ValueBytes::deserialize(serialized);

        assert_eq!(deserialized.ns(), 42);
        assert_eq!(deserialized.as_bytes().len(), 100000);
    }

    #[test]
    fn test_key_default() {
        let key = KeyBytes::default();
        assert_eq!(key.ns(), 0);
        assert_eq!(key.ts(), 0);
        assert!(key.is_empty());
    }

    #[test]
    fn test_value_default() {
        let val = ValueBytes::default();
        assert_eq!(val.ns(), 0);
        assert_eq!(val.as_bytes().len(), 0);
    }

    #[test]
    fn test_key_setters() {
        let mut key = KeyBytes::new(0, Bytes::from("original"), 100);

        key.set_ns(5);
        assert_eq!(key.ns(), 5);

        key.set_ts(200);
        assert_eq!(key.ts(), 200);

        key.set_key(Bytes::from("modified"));
        assert_eq!(key.as_bytes(), Bytes::from("modified"));
    }

    #[test]
    fn test_value_setters() {
        let mut val = ValueBytes::new(0, Bytes::from("original"));

        val.set_ns(10);
        assert_eq!(val.ns(), 10);
    }

    #[test]
    fn test_value_from_slice() {
        let data = b"test data";
        let val = ValueBytes::from_slice(3, data);

        assert_eq!(val.ns(), 3);
        assert_eq!(val.as_bytes(), Bytes::from(&data[..]));
    }

    #[test]
    fn test_value_ordering() {
        let val1 = ValueBytes::new(0, Bytes::from("aaa"));
        let val2 = ValueBytes::new(0, Bytes::from("bbb"));
        let val3 = ValueBytes::new(1, Bytes::from("aaa"));

        assert!(val1 < val2, "values should be ordered by value bytes in same namespace");
        assert!(val1 < val3, "values should be ordered by namespace first");
        assert!(val2 < val3, "namespace ordering should take precedence");
    }

    #[test]
    fn test_map_key_bound() {
        use std::collections::Bound;

        use crate::keypair::map_key_bound;

        let key = KeyBytes::new(0, Bytes::from("test"), 100);
        let bound = Bound::Included(key.clone());
        let mapped = map_key_bound(bound);

        match mapped {
            | Bound::Included(bytes) => {
                let deserialized = KeyBytes::deserialize(bytes);
                assert_eq!(deserialized, key);
            },
            | _ => panic!("expected Included bound"),
        }

        let bound = Bound::Excluded(key.clone());
        let mapped = map_key_bound(bound);
        match mapped {
            | Bound::Excluded(_) => {},
            | _ => panic!("expected Excluded bound"),
        }

        let bound: Bound<KeyBytes> = Bound::Unbounded;
        let mapped = map_key_bound(bound);
        match mapped {
            | Bound::Unbounded => {},
            | _ => panic!("expected Unbounded bound"),
        }
    }

    #[test]
    fn test_timestamp_boundary_values() {
        // test with u128::MAX timestamp
        let key_max = KeyBytes::new(0, Bytes::from("key"), u128::MAX);
        let serialized = key_max.serialize();
        let deserialized = KeyBytes::deserialize(serialized);
        assert_eq!(deserialized.ts(), u128::MAX);

        // test with 0 timestamp
        let key_zero = KeyBytes::new(0, Bytes::from("key"), 0);
        let serialized = key_zero.serialize();
        let deserialized = KeyBytes::deserialize(serialized);
        assert_eq!(deserialized.ts(), 0);
    }

    #[test]
    fn test_namespace_boundary_values() {
        // test with u64::MAX namespace
        let key = KeyBytes::new(u64::MAX, Bytes::from("key"), 100);
        let serialized = key.serialize();
        let deserialized = KeyBytes::deserialize(serialized);
        assert_eq!(deserialized.ns(), u64::MAX);

        let val = ValueBytes::new(u64::MAX, Bytes::from("value"));
        let serialized = val.serialize();
        let deserialized = ValueBytes::deserialize(serialized);
        assert_eq!(deserialized.ns(), u64::MAX);
    }

    #[test]
    fn test_tombstone_creation() {
        let tombstone = ValueBytes::new_tombstone(42);
        assert!(tombstone.is_tombstone());
        assert_eq!(tombstone.ns(), 42);
        assert_eq!(tombstone.as_bytes().len(), 0);
    }

    #[test]
    fn test_tombstone_serialization() {
        let tombstone = ValueBytes::new_tombstone(5);

        // Test memory serialization
        let serialized = tombstone.serialize();
        let deserialized = ValueBytes::deserialize(serialized);
        assert!(deserialized.is_tombstone());
        assert_eq!(deserialized.ns(), 5);

        // Test storage serialization
        let serialized = tombstone.serialize();
        let deserialized = ValueBytes::deserialize(serialized);
        assert!(deserialized.is_tombstone());
        assert_eq!(deserialized.ns(), 5);
    }

    #[test]
    fn test_non_tombstone_values() {
        let value = ValueBytes::new(0, Bytes::from("data"));
        assert!(!value.is_tombstone());

        let value = ValueBytes::from_slice(1, b"more data");
        assert!(!value.is_tombstone());

        let value = ValueBytes::default();
        assert!(!value.is_tombstone());
    }
}
