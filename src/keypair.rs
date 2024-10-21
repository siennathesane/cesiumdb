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

pub(crate) const DEFAULT_NS: u64 = 0;

#[instrument]
pub fn map_key_bound(bound: Bound<KeyBytes>) -> Bound<Bytes> {
    match bound {
        | Bound::Included(x) => {
            let mem_key = x.serialize_for_memory();
            Bound::Included(mem_key)
        },
        | Bound::Excluded(x) => {
            let mem_key = x.serialize_for_memory();
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

    pub fn set_ns(&mut self, ns: u64) {
        self.ns = ns;
    }

    pub fn ns(&self) -> u64 {
        self.ns
    }

    #[instrument(level = "trace")]
    #[inline]
    pub fn deserialize_from_memory(slice: Bytes) -> Self {
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

    #[instrument(level = "trace")]
    #[inline]
    pub fn deserialize_from_storage(slice: Bytes) -> Self {
        #[cfg(feature = "secure")]
        {
            let mut hasher = Hasher::new();
            hasher.update(&slice[16..slice.len() - 16]);
            let checksum = hasher.finalize();

            let mut ecc_arr = [0_u8; 4];
            ecc_arr.copy_from_slice(&slice[0..4]);
            let existing_checksum = u32::from_le_bytes(ecc_arr);

            assert_eq!(
                existing_checksum, checksum,
                "key record has wrong checksum. found: {} computed: {}",
                existing_checksum, checksum
            );
        }

        let mut ns_arr = [0u8; 8];
        ns_arr.copy_from_slice(&slice[8..16]);

        let mut ts_arr = [0u8; 16];
        ts_arr.copy_from_slice(&slice[slice.len() - 16..]);

        KeyBytes {
            ns: u64::from_le_bytes(ns_arr),
            key: Bytes::copy_from_slice(&slice[16..slice.len() - 8]),
            ts: u128::MAX - u128::from_le_bytes(ts_arr),
        }
    }

    pub fn as_bytes(&self) -> Bytes {
        self.key.clone()
    }

    #[instrument(level = "trace")]
    #[inline]
    pub fn serialize_for_storage(&self) -> Bytes {
        let mut hasher = Hasher::new();
        hasher.update(self.key.as_ref());
        let checksum = hasher.finalize();

        // namespace + key + timestamp
        let len = size_of::<u64>() + self.key.as_ref().len() + size_of::<u64>();

        let mut bytes = BytesMut::with_capacity(size_of::<u32>() + size_of::<u32>() + len);

        // this is the serialized key
        bytes.put_u32_le(checksum);
        bytes.put_u32_le(len as u32);
        bytes.put_u64_le(self.ns);
        bytes.put_slice(self.key.as_ref());
        bytes.put_u128_le(u128::MAX - self.ts);

        bytes.freeze()
    }

    #[instrument(level = "trace")]
    #[inline]
    pub fn serialize_for_memory(&self) -> Bytes {
        let mut bytes =
            BytesMut::with_capacity(size_of::<u64>() + self.key.as_ref().len() + size_of::<u128>());

        // this is the serialized key
        bytes.put_u64_le(self.ns);
        bytes.put_slice(self.key.as_ref());
        bytes.put_u128_le(u128::MAX - self.ts);

        bytes.freeze()
    }

    #[instrument(level = "trace")]
    #[inline]
    pub fn serialize_for_latest(&self) -> Bytes {
        let original = self.serialize_for_memory();
        let mut bytes = BytesMut::from(original.as_ref());
        for idx in 0..size_of::<u128>() {
            bytes[original.len() - idx - 1] = 0xff;
        }
        bytes.freeze()
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

#[derive(Debug, Eq, Clone, Copy, PartialEq)]
pub struct Value<T: AsRef<[u8]>> {
    pub ns: u64,
    pub value: T,
}

pub type ValueBytes = Value<Bytes>;

impl ValueBytes {
    pub fn new(ns: u64, val: Bytes) -> Self {
        ValueBytes { ns, value: val }
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
        }
    }

    #[instrument(level = "trace")]
    pub fn deserialize_from_memory(bytes: Bytes) -> Self {
        let mut ns_arr = [0u8; 8];
        ns_arr.copy_from_slice(&bytes[0..8]);

        ValueBytes {
            ns: u64::from_le_bytes(ns_arr),
            value: Bytes::copy_from_slice(&bytes[8..]),
        }
    }

    #[instrument(level = "trace")]
    #[inline]
    pub fn deserialize_from_disk(slice: Bytes) -> Self {
        #[cfg(feature = "secure")]
        {
            let mut hasher = Hasher::new();
            hasher.update(&slice[16..slice.len() - 8]);
            let checksum = hasher.finalize();

            let mut ecc_arr = [0_u8; 4];
            ecc_arr.copy_from_slice(&slice[0..4]);
            let existing_checksum = u32::from_le_bytes(ecc_arr);

            assert_eq!(
                existing_checksum, checksum,
                "value record has wrong checksum. found: {} computed: {}",
                existing_checksum, checksum
            );
        }

        let mut ns_arr = [0u8; 8];
        ns_arr.copy_from_slice(&slice[8..16]);

        ValueBytes {
            ns: u64::from_le_bytes(ns_arr),
            value: Bytes::copy_from_slice(&slice[16..]),
        }
    }

    #[instrument(level = "trace")]
    #[inline]
    pub fn serialize_for_storage(&self) -> Bytes {
        let mut hasher = Hasher::new();
        hasher.update(self.value.as_ref());
        let checksum = hasher.finalize();

        // namespace + payload
        let len = size_of::<u64>() + self.value.as_ref().len();

        let mut buf = BytesMut::with_capacity(size_of::<u64>() + len);
        buf.put_u32_le(checksum);
        buf.put_u32_le(len as u32);
        buf.put_u64_le(self.ns);
        buf.put_slice(self.value.as_ref());

        buf.freeze()
    }

    #[instrument(level = "trace")]
    #[inline]
    pub fn serialize_for_memory(&self) -> Bytes {
        // namespace + payload
        let len = size_of::<u64>() + self.value.as_ref().len();

        let mut buf = BytesMut::with_capacity(len);
        buf.put_u64_le(self.ns);
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

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use crate::keypair::{
        KeyBytes,
        ValueBytes,
    };

    #[test]
    fn test_key_serialization() {
        let key = KeyBytes {
            ns: 0,
            key: Bytes::from("test"),
            ts: 0,
        };
        let memory_serialized = key.serialize_for_memory();

        // 8 + 4 + 16
        // ns + payload + ts
        assert_eq!(memory_serialized.clone().len(), 28);

        let de_key = KeyBytes::deserialize_from_memory(memory_serialized.clone());
        assert_eq!(key, de_key);

        // 4 + 4 + 8 + 4 + 16
        let storage_serialized = key.serialize_for_storage();
        assert_eq!(storage_serialized.len(), 36);
        let de_key =
            KeyBytes::deserialize_from_memory(Bytes::copy_from_slice(&storage_serialized[8..]));
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
        let val = ValueBytes {
            ns: 0,
            value: Bytes::from("test-value"),
        };
        let serialized = val.serialize_for_memory();
        assert_eq!(serialized.len(), 18);

        let de_val = ValueBytes::deserialize_from_memory(serialized);
        assert_eq!(val, de_val);

        let serialized = val.serialize_for_storage();
        assert_eq!(serialized.len(), 26);
        let de_val = ValueBytes::deserialize_from_disk(serialized);
        assert_eq!(val, de_val);
    }
}
