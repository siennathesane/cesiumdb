// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

//! Zero-copy entry wrapper for compaction.
//!
//! `RawEntry` wraps already-serialized key and value bytes, providing
//! lazy accessors via pointer arithmetic instead of deserialization.
//! This eliminates the deserialize/re-serialize round-trip during compaction.

use std::cmp::Ordering;

use bytes::Bytes;

/// A lightweight wrapper around raw serialized key/value bytes.
///
/// Key format: `[ns:8][user_key:var][inverted_ts:16]`
/// Value format: `[ns:8][tombstone:1][value_data:var]`
///
/// All accessors are zero-copy — they slice into the existing buffers.
pub(crate) struct RawEntry {
    /// Serialized key: [ns:8][user_key:var][inverted_ts:16]
    key_data: Bytes,
    /// Serialized value: [ns:8][tombstone:1][value_data:var]
    val_data: Bytes,
}

impl RawEntry {
    /// Creates a new RawEntry from pre-serialized key and value bytes.
    #[inline]
    pub fn new(key_data: Bytes, val_data: Bytes) -> Self {
        Self { key_data, val_data }
    }

    /// Reads the namespace from the first 8 bytes of key_data.
    #[inline]
    pub fn ns(&self) -> u64 {
        u64::from_le_bytes(self.key_data[0..8].try_into().unwrap())
    }

    /// Returns the user key slice: bytes 8..len-16 of key_data.
    #[inline]
    pub fn user_key(&self) -> &[u8] {
        &self.key_data[8..self.key_data.len() - 16]
    }

    /// Reads the timestamp from the last 16 bytes (inverted).
    #[inline]
    pub fn ts(&self) -> u128 {
        // IMPORTANT: Timestamps are now stored as big-endian
        let inverted = u128::from_be_bytes(
            self.key_data[self.key_data.len() - 16..]
                .try_into()
                .unwrap(),
        );
        u128::MAX - inverted
    }

    /// Checks if this entry is a tombstone by reading byte 8 of val_data.
    #[inline]
    pub fn is_tombstone(&self) -> bool {
        self.val_data[8] != 0
    }

    /// Returns the dedup key: `[ns:8][user_key]` (everything except the
    /// timestamp). Two entries with the same dedup key are versions of the
    /// same logical key.
    #[inline]
    pub fn dedup_key(&self) -> &[u8] {
        &self.key_data[..self.key_data.len() - 16]
    }

    /// Compares two raw entries using the same ordering as
    /// `KeyBytes::simd_cmp()`: namespace ascending, then user key bytes
    /// ascending, then timestamp descending.
    ///
    /// Since keys are serialized as `[ns:8_le][user_key][inverted_ts:16_le]`,
    /// and the inverted timestamp already provides descending order via byte
    /// comparison, we can compare the raw serialized bytes directly for the
    /// key portion. However, little-endian namespace comparison requires
    /// field-level comparison.
    #[inline]
    pub fn cmp_key(&self, other: &Self) -> Ordering {
        // Compare namespace (u64 little-endian — must compare as integers, not bytes)
        let self_ns = self.ns();
        let other_ns = other.ns();
        self_ns
            .cmp(&other_ns)
            .then_with(|| {
                // Compare user key bytes
                crate::simd::simd_compare_keys(self.user_key(), other.user_key())
            })
            .then_with(|| {
                // Compare inverted timestamps as bytes (already in correct order for descending
                // ts)
                let self_ts = &self.key_data[self.key_data.len() - 16..];
                let other_ts = &other.key_data[other.key_data.len() - 16..];
                self_ts.cmp(other_ts)
            })
    }

    /// Direct access to the raw serialized key bytes.
    #[inline]
    pub fn raw_key(&self) -> &Bytes {
        &self.key_data
    }

    /// Direct access to the raw serialized value bytes.
    #[inline]
    pub fn raw_val(&self) -> &Bytes {
        &self.val_data
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::*;
    use crate::{
        keypair::{
            KeyBytes,
            ValueBytes,
        },
        utils::Serializer,
    };

    fn make_raw_entry(ns: u64, key: &[u8], ts: u128, value: &[u8], tombstone: bool) -> RawEntry {
        let k = KeyBytes::new(ns, Bytes::copy_from_slice(key), ts);
        let v = if tombstone {
            ValueBytes::new_tombstone(ns)
        } else {
            ValueBytes::new(ns, Bytes::copy_from_slice(value))
        };
        RawEntry::new(k.serialize(), v.serialize())
    }

    #[test]
    fn test_ns() {
        let entry = make_raw_entry(42, b"hello", 100, b"world", false);
        assert_eq!(entry.ns(), 42);
    }

    #[test]
    fn test_user_key() {
        let entry = make_raw_entry(0, b"hello", 100, b"world", false);
        assert_eq!(entry.user_key(), b"hello");
    }

    #[test]
    fn test_ts() {
        let entry = make_raw_entry(0, b"hello", 12345, b"world", false);
        assert_eq!(entry.ts(), 12345);
    }

    #[test]
    fn test_is_tombstone() {
        let live = make_raw_entry(0, b"hello", 1, b"world", false);
        assert!(!live.is_tombstone());

        let dead = make_raw_entry(0, b"hello", 1, b"", true);
        assert!(dead.is_tombstone());
    }

    #[test]
    fn test_dedup_key() {
        let e1 = make_raw_entry(0, b"hello", 1, b"v1", false);
        let e2 = make_raw_entry(0, b"hello", 2, b"v2", false);
        let e3 = make_raw_entry(0, b"world", 1, b"v3", false);

        // Same logical key, different timestamps → same dedup_key
        assert_eq!(e1.dedup_key(), e2.dedup_key());
        // Different logical keys → different dedup_key
        assert_ne!(e1.dedup_key(), e3.dedup_key());
    }

    #[test]
    fn test_cmp_key_matches_keybytes_ordering() {
        // Same ordering as KeyBytes::simd_cmp:
        // ns ascending, user_key ascending, ts descending
        let cases = vec![
            // (ns1, key1, ts1, ns2, key2, ts2)
            (
                0u64,
                b"a".as_ref(),
                1u128,
                0u64,
                b"b".as_ref(),
                1u128,
                Ordering::Less,
            ),
            (0, b"b", 1, 0, b"a", 1, Ordering::Greater),
            (0, b"a", 2, 0, b"a", 1, Ordering::Less), // higher ts → less (descending)
            (0, b"a", 1, 0, b"a", 2, Ordering::Greater),
            (1, b"a", 1, 0, b"a", 1, Ordering::Greater), // ns=1 > ns=0
            (0, b"a", 1, 0, b"a", 1, Ordering::Equal),
        ];

        for (ns1, k1, ts1, ns2, k2, ts2, expected) in cases {
            let e1 = make_raw_entry(ns1, k1, ts1, b"", false);
            let e2 = make_raw_entry(ns2, k2, ts2, b"", false);

            let raw_cmp = e1.cmp_key(&e2);
            assert_eq!(
                raw_cmp, expected,
                "cmp_key mismatch: ns=({},{}), key=({:?},{:?}), ts=({},{}): got {:?}, expected {:?}",
                ns1, ns2, k1, k2, ts1, ts2, raw_cmp, expected
            );

            // Cross-validate with KeyBytes::simd_cmp
            let kb1 = KeyBytes::new(ns1, Bytes::copy_from_slice(k1), ts1);
            let kb2 = KeyBytes::new(ns2, Bytes::copy_from_slice(k2), ts2);
            let simd_cmp = kb1.simd_cmp(&kb2);
            assert_eq!(
                raw_cmp, simd_cmp,
                "RawEntry::cmp_key disagrees with KeyBytes::simd_cmp"
            );
        }
    }

    #[test]
    fn test_raw_key_val_roundtrip() {
        let k = KeyBytes::new(7, Bytes::from("testkey"), 999);
        let v = ValueBytes::new(7, Bytes::from("testval"));
        let key_bytes = k.serialize();
        let val_bytes = v.serialize();

        let entry = RawEntry::new(key_bytes.clone(), val_bytes.clone());
        assert_eq!(entry.raw_key(), &key_bytes);
        assert_eq!(entry.raw_val(), &val_bytes);
    }
}
