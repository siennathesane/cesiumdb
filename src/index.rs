use std::{
    fmt::Debug,
    ptr,
};

use bytes::{
    BufMut,
    Bytes,
    BytesMut,
};
use gxhash::gxhash64;
use tracing::instrument;

/// Minimum safe input length for gxhash64 SIMD loads.
const MIN_HASH_LEN: usize = 16;

/// Safely hash a byte slice with gxhash64, padding short inputs to avoid
/// out-of-bounds SIMD reads on small heap allocations.
#[inline]
fn safe_gxhash64(key: &[u8], seed: i64) -> u64 {
    if key.len() >= MIN_HASH_LEN {
        gxhash64(key, seed)
    } else {
        let mut padded = [0u8; MIN_HASH_LEN];
        padded[..key.len()].copy_from_slice(key);
        gxhash64(&padded, seed)
    }
}

use crate::{
    bloom::{
        Bloom2,
        BloomFilterBuilder,
        BytesBitmap,
        FilterSize::KeyBytes3,
    },
    hash::SeedableHasher,
};

// Index header constants
/// Number of u64 fields in the index header
const INDEX_HEADER_FIELDS: usize = 6;
/// Size of the index header: 6 * u64 = 48 bytes
/// Fields: id, bloom_filter_seed, bloom_filter_size, ns_offset_size,
/// block_offset_size, num_blocks
const INDEX_HEADER_SIZE: usize = INDEX_HEADER_FIELDS * size_of::<u64>();
/// Size of each offset entry (namespace or block): 2 * u64 = 16 bytes
const OFFSET_ENTRY_SIZE: usize = 2 * size_of::<u64>();
/// Minimum valid index size (header + some data)
pub(crate) const MIN_INDEX_SIZE: usize = 56;

/// Integrated index that combines bloom filtering with block-level lookup. The
/// workflow is exposed for internal flexibility. Lookups are O(log n) for any
/// item in the index, regardless of the size of the index.
pub struct Index {
    // Header fields
    /// Id of the segment this index belongs to
    id: u64,
    /// Seed for the bloom filter
    bloom_filter_seed: i64,
    /// Number of blocks in the index
    num_blocks: u64,

    // Data fields
    /// Indexed namespace offsets. The memory layout of
    /// this item:
    ///
    ///   [123, 0, 345, 1]
    ///
    /// Where each offset is stored as a contiguous set of 16-byte pairs. For
    /// each pair, the first 8 bytes are the namespace offset and the second
    /// 8 bytes are the block offset. The data is ordered by hash and there is
    /// no delimiter.
    ns_offset_entries: Vec<(u64, u64)>,

    // in-memory only fields
    /// The bloom filter.
    bloom_filter: Bloom2<SeedableHasher, BytesBitmap, u64>,

    /// Indexed block offsets. The memory layout of this item:
    ///
    ///  [hash, block_offset, hash, block_offset]
    ///
    /// Where each offset is stored as a contiguous set of 8-byte pairs. For
    /// each pair, the first 8 bytes are the hash and the second
    /// 8 bytes are the block offset. The data is ordered by hash and there is
    /// no delimiter.
    block_offset_entries: Vec<(u64, u64)>, // (hash, block_offset) pairs
}

impl Index {
    /// create a new index with the specified id and bloom filter seed
    #[instrument(level = "trace")]
    pub fn new(id: u64, seed: i64) -> Self {
        let hasher = SeedableHasher::new(seed);
        Self {
            id,
            bloom_filter_seed: seed,
            num_blocks: 0,
            ns_offset_entries: Vec::new(),
            bloom_filter: BloomFilterBuilder::hasher(hasher)
                .with_bitmap()
                .size(KeyBytes3)
                .build(),
            block_offset_entries: Vec::new(),
        }
    }

    /// Returns the number of blocks indexed
    pub fn num_blocks(&self) -> u64 {
        self.num_blocks
    }

    /// Sets the number of blocks (used when loading from metadata)
    pub(crate) fn set_num_blocks(&mut self, count: u64) {
        self.num_blocks = count;
    }

    /// Insert an item into the index.
    #[instrument(level = "trace")]
    pub fn insert_item(&mut self, key: &[u8]) {
        let hash = safe_gxhash64(key, self.bloom_filter_seed);

        self.bloom_filter.insert(&hash);
        match self
            .block_offset_entries
            .binary_search_by_key(&hash, |(h, _)| *h)
        {
            | Ok(_) => {},
            | Err(idx) => self
                .block_offset_entries
                .insert(idx, (hash, self.num_blocks)),
        }
    }

    /// Batch rebuild bloom filter from a collection of (key, block_idx) pairs.
    /// Much faster than calling insert_item() repeatedly.
    ///
    /// # Arguments
    /// * `key_block_pairs` - Iterator of (key_bytes, block_index) tuples
    pub fn rebuild_bloom_from_keys<'a, I>(&mut self, key_block_pairs: I)
    where
        I: Iterator<Item = (&'a [u8], u64)>, {
        // Build new bloom filter
        let hasher = SeedableHasher::new(self.bloom_filter_seed);
        let mut new_bloom = BloomFilterBuilder::hasher(hasher)
            .with_bitmap()
            .size(KeyBytes3)
            .build();

        // Collect hashes with block indices
        let mut hashes = Vec::with_capacity(1000); // Preallocate for common case

        for (key, block_idx) in key_block_pairs {
            let hash = safe_gxhash64(key, self.bloom_filter_seed);
            new_bloom.insert(&hash);
            hashes.push((hash, block_idx));
        }

        // Sort and dedup by hash (keep first occurrence for block_offset_entries)
        hashes.sort_unstable_by_key(|(h, _)| *h);
        hashes.dedup_by_key(|(h, _)| *h);

        // Replace structures
        self.block_offset_entries = hashes;
        self.bloom_filter = new_bloom;
    }

    /// Increment the block counter.
    pub fn inc_block_count(&mut self, i: u64) {
        self.num_blocks += i;
    }

    /// Insert a namespace offset to the most recent block.
    #[instrument(level = "trace")]
    pub fn insert_ns_offset(&mut self, ns: u64) {
        match self
            .ns_offset_entries
            .binary_search_by_key(&ns, |(h, _)| *h)
        {
            | Ok(_) => {},
            | Err(idx) => self.ns_offset_entries.insert(idx, (ns, self.num_blocks)),
        }
    }

    /// Check if a key might be present in the index
    pub fn may_contain(&self, key: &[u8]) -> bool {
        let hash = safe_gxhash64(key, self.bloom_filter_seed);
        self.bloom_filter.contains(&hash)
    }

    /// Find the block that contains the start of the namespace
    #[instrument(level = "trace")]
    pub fn get_namespace_block(&self, ns: u64) -> Option<u64> {
        self.ns_offset_entries
            .binary_search_by_key(&ns, |(n, _b)| *n)
            .ok()
            .map(|idx| self.ns_offset_entries[idx].1)
    }

    /// Get a block offset by hash from the in-memory block index
    #[instrument(level = "trace")]
    pub fn get_block(&self, key: &[u8]) -> Option<u64> {
        let hash = safe_gxhash64(key, self.bloom_filter_seed);
        self.block_offset_entries
            .binary_search_by_key(&hash, |(h, _b)| *h)
            .ok()
            .map(|idx| self.block_offset_entries[idx].1)
    }

    /// Combined bloom-filter + block-index lookup.
    /// Returns `None` if the bloom filter says the key is definitely absent,
    /// otherwise returns `Some(block_offset)` (which may be `None` if the
    /// block index doesn't have an entry).
    #[inline]
    pub fn may_contain_and_get_block(&self, key: &[u8]) -> Option<Option<u64>> {
        let hash = safe_gxhash64(key, self.bloom_filter_seed);
        if !self.bloom_filter.contains(&hash) {
            return None;
        }
        let block = self
            .block_offset_entries
            .binary_search_by_key(&hash, |(h, _b)| *h)
            .ok()
            .map(|idx| self.block_offset_entries[idx].1);
        Some(block)
    }

    /// Get the id of this index
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get the total number of blocks in the index; used to determine segment
    /// ownership if needed.
    pub fn block_count(&self) -> u64 {
        self.num_blocks
    }

    /// Get the total number of namespace offsets in the index
    pub fn ns_offset_count(&self) -> u64 {
        self.ns_offset_entries.len() as u64
    }

    /// Returns the total size in bytes this index will occupy when serialized.
    /// It does not contain the number of items in the index.
    #[instrument(level = "trace")]
    pub fn size(&self) -> usize {
        // the overall size of the headers.
        //   id
        //   bloom_filter_seed
        //   bloom_filter_size
        //   ns_offset_size
        //   block_offset_size
        //   num_blocks
        let header_size = INDEX_HEADER_SIZE;

        // the size of the bloom filter data
        let bloom_size = self.bloom_filter.bitmap().clone().freeze().len();

        // the size of the block offsets
        let block_offset_size = self.block_offset_entries.len() * OFFSET_ENTRY_SIZE;

        // the size of the namespace offsets
        let ns_offset_size = self.ns_offset_entries.len() * OFFSET_ENTRY_SIZE;

        header_size + bloom_size + block_offset_size + ns_offset_size
    }

    /// Finalizes the Index by writing it directly to a memory location.
    ///
    /// # Safety
    ///
    /// - `dst` must be valid for at least `self.size()` bytes
    /// - `dst` must be properly aligned for u64 writes (8-byte alignment)
    /// - `dst` must not overlap with any source data
    /// - Caller must ensure exclusive access to the dst memory region
    #[instrument(level = "trace", skip(dst))]
    pub(crate) unsafe fn finalize(&self, dst: *mut u8) {
        // SAFETY: Verify alignment invariants in debug builds
        debug_assert!(!dst.is_null(), "Destination pointer must not be null");
        debug_assert!(
            (dst as usize).is_multiple_of(std::mem::align_of::<u64>()),
            "Destination pointer must be 8-byte aligned for u64 writes"
        );

        // write header fields
        let mut offset = 0;

        // prep the block offset data
        let mut block_offset_entries =
            BytesMut::with_capacity(self.block_offset_entries.len() * 16);
        self.block_offset_entries.iter().for_each(|(h, b)| {
            block_offset_entries.put_u64_le(*h);
            block_offset_entries.put_u64_le(*b);
        });
        let block_offset_entries = block_offset_entries.freeze();

        // prep the namespace offset data
        let mut ns_offset_entries = BytesMut::with_capacity(self.ns_offset_entries.len() * 16);
        self.ns_offset_entries.iter().for_each(|(n, b)| {
            ns_offset_entries.put_u64_le(*n);
            ns_offset_entries.put_u64_le(*b);
        });
        let ns_offset_entries = ns_offset_entries.freeze();

        // get bloom filter data
        let bloom_data = self.bloom_filter.bitmap().clone().freeze();
        let bloom_filter_size = bloom_data.len() as u64;

        // SAFETY: All writes stay within the allocated buffer size (verified by
        // caller). Each write advances the offset to ensure non-overlapping
        // writes.
        unsafe {
            // write id
            ptr::copy_nonoverlapping(
                self.id.to_le_bytes().as_ptr(),
                dst.add(offset),
                size_of::<u64>(),
            );
            offset += size_of::<u64>();

            // write bloom_filter_seed
            ptr::copy_nonoverlapping(
                self.bloom_filter_seed.to_le_bytes().as_ptr(),
                dst.add(offset),
                size_of::<i64>(),
            );
            offset += size_of::<i64>();

            // write bloom_size
            ptr::copy_nonoverlapping(
                bloom_filter_size.to_le_bytes().as_ptr(),
                dst.add(offset),
                size_of::<u64>(),
            );
            offset += size_of::<u64>();

            // write ns_offset_entries length
            ptr::copy_nonoverlapping(
                ns_offset_entries.len().to_le_bytes().as_ptr(),
                dst.add(offset),
                size_of::<u64>(),
            );
            offset += size_of::<u64>();

            // write block_offset_size
            ptr::copy_nonoverlapping(
                block_offset_entries.len().to_le_bytes().as_ptr(),
                dst.add(offset),
                size_of::<u64>(),
            );
            offset += size_of::<u64>();

            // write num_blocks
            ptr::copy_nonoverlapping(
                self.num_blocks.to_le_bytes().as_ptr(),
                dst.add(offset),
                size_of::<u64>(),
            );
            offset += size_of::<u64>();

            // write data sections

            // write block_offsets
            ptr::copy_nonoverlapping(
                block_offset_entries.as_ptr(),
                dst.add(offset),
                block_offset_entries.len(),
            );
            offset += block_offset_entries.len();

            // write ns_offsets
            ptr::copy_nonoverlapping(
                ns_offset_entries.as_ptr(),
                dst.add(offset),
                ns_offset_entries.len(),
            );
            offset += ns_offset_entries.len();

            // write bloom_filter data
            ptr::copy_nonoverlapping(bloom_data.as_ptr(), dst.add(offset), bloom_data.len());
        }
    }
}

impl From<Index> for Bytes {
    #[instrument(level = "trace", skip(value))]
    fn from(value: Index) -> Bytes {
        let size = value.size();
        let mut buffer = BytesMut::with_capacity(size);
        buffer.resize(size, 0);

        // SAFETY: we just allocated enough space
        unsafe {
            value.finalize(buffer.as_mut_ptr());
        }

        buffer.freeze()
    }
}

impl From<&Index> for Bytes {
    #[instrument(level = "trace", skip(value))]
    fn from(value: &Index) -> Bytes {
        let size = value.size();
        let mut buffer = BytesMut::with_capacity(size);
        buffer.resize(size, 0);

        // SAFETY: we just allocated enough space
        unsafe {
            value.finalize(buffer.as_mut_ptr());
        }

        buffer.freeze()
    }
}

impl From<Bytes> for Index {
    #[instrument(level = "trace", skip(value))]
    fn from(value: Bytes) -> Self {
        // the index header metadata fields are 8 bytes each
        debug_assert!(value.len() > 48, "index metadata too small");

        // use offsets so we can track where we are in the buffer
        let mut offset = 0;

        let id = u64::from_le_bytes(value[offset..size_of::<u64>()].try_into().unwrap());
        offset += size_of::<u64>();

        let bloom_filter_seed =
            i64::from_le_bytes(value[offset..offset + size_of::<i64>()].try_into().unwrap());
        offset += size_of::<i64>();

        let bloom_filter_size =
            u64::from_le_bytes(value[offset..offset + size_of::<u64>()].try_into().unwrap());
        offset += size_of::<u64>();

        let ns_offset_size =
            u64::from_le_bytes(value[offset..offset + size_of::<u64>()].try_into().unwrap());
        offset += size_of::<u64>();

        let block_offset_size =
            u64::from_le_bytes(value[offset..offset + size_of::<u64>()].try_into().unwrap());
        offset += size_of::<u64>();

        let num_blocks =
            u64::from_le_bytes(value[offset..offset + size_of::<u64>()].try_into().unwrap());
        offset += size_of::<u64>();

        let block_offsets = BytesMut::from(&value[offset..offset + block_offset_size as usize]);
        offset += block_offsets.len();

        // build the block entries from the block offsets
        let mut block_offset_entries: Vec<(u64, u64)> =
            Vec::with_capacity(block_offsets.len() / 16);

        block_offsets.chunks_exact(16).for_each(|chunk| {
            let hash = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
            let block_offset = u64::from_le_bytes(chunk[8..16].try_into().unwrap());

            match block_offset_entries.binary_search_by_key(&hash, |(h, _)| *h) {
                | Ok(_) => {},
                | Err(idx) => block_offset_entries.insert(idx, (hash, block_offset)),
            }
        });

        let ns_offset_entries_bin =
            BytesMut::from(&value[offset..offset + ns_offset_size as usize]);
        offset += ns_offset_entries_bin.len();

        let mut ns_offset_entries: Vec<(u64, u64)> =
            Vec::with_capacity(ns_offset_entries_bin.len() / 16);

        ns_offset_entries_bin.chunks_exact(16).for_each(|chunk| {
            let mut ns_buf = [0u8; 8];
            ns_buf.copy_from_slice(chunk[0..8].as_ref());
            let ns = u64::from_le_bytes(ns_buf);

            let mut block_buf = [0u8; 8];
            block_buf.copy_from_slice(chunk[8..16].as_ref());
            let block_offset = u64::from_le_bytes(block_buf);

            match ns_offset_entries.binary_search_by_key(&ns, |(n, _b)| *n) {
                | Ok(_) => {},
                | Err(idx) => ns_offset_entries.insert(idx, (ns, block_offset)),
            }
        });

        let bloom_filter_data = BytesMut::from(&value[offset..offset + bloom_filter_size as usize]);

        // recreate the bloom filter
        let hasher = SeedableHasher::new(bloom_filter_seed);
        let bitmap = BytesBitmap::from_bytes(bloom_filter_data.clone());

        // Ensure we're using the correct filter size and configuration
        let bloom_filter = BloomFilterBuilder::hasher(hasher)
            .with_bitmap()
            .with_bitmap_data(bitmap, KeyBytes3)
            .build();

        Self {
            id,
            bloom_filter_seed,
            block_offset_entries,
            ns_offset_entries,
            bloom_filter,
            num_blocks,
        }
    }
}

impl Debug for Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Index")
            .field("id", &self.id)
            .field("bloom_filter_seed", &self.bloom_filter_seed)
            .field("num_blocks", &self.num_blocks)
            .field("ns_offset_entries", &self.ns_offset_entries)
            .field("block_offset_entries", &self.block_offset_entries)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    

    use bytes::Bytes;

    use super::*;
    use crate::keypair::{
        DEFAULT_NS,
        KeyBytes,
    };

    // helper function to create test data
    fn create_test_key(id: u32) -> Bytes {
        KeyBytes::new(
            DEFAULT_NS,
            Bytes::copy_from_slice(format!("test_key_{id}").as_bytes()),
            u128::MAX,
        )
        .as_bytes()
    }

    fn create_test_key_ns(id: u32, ns: u64) -> Bytes {
        KeyBytes::new(
            ns,
            Bytes::copy_from_slice(format!("test_key_{id}").as_bytes()),
            u128::MAX,
        )
        .as_bytes()
    }

    #[test]
    fn test_debug() {
        let index = Index::new(42, 123);
        let debug_str = format!("{:?}", index);
        assert!(debug_str.contains("Index"));
        assert!(debug_str.contains("id: 42"));
        assert!(debug_str.contains("bloom_filter_seed: 123"));
    }

    #[test]
    fn test_new_index() {
        let index = Index::new(42, 123);
        assert_eq!(index.id(), 42);
        assert_eq!(index.block_count(), 0);
        assert_eq!(index.ns_offset_count(), 0);
        assert_eq!(index.block_offset_entries.len(), 0);
    }

    #[test]
    fn test_add_item() {
        let mut index = Index::new(1, 100);

        // add some items
        let item1 = create_test_key(1);
        let item2 = create_test_key(2);

        index.insert_item(&item1);
        index.insert_item(&item2);

        // bloom filter should now contain these items
        assert!(index.may_contain(&item1));
        assert!(index.may_contain(&item2));

        // but not some random key
        let non_existent = create_test_key(999);
        assert!(!index.may_contain(&non_existent));
    }

    #[test]
    fn test_add_block() {
        let seed = 100;
        let mut index = Index::new(1, seed);

        index.inc_block_count(1);
        index.inc_block_count(1);
        index.inc_block_count(1);

        assert_eq!(3, index.block_count());
    }

    #[test]
    fn test_add_ns_offset() {
        let seed = 100;
        let test_ns = 150;
        let mut index = Index::new(1, seed);

        index.insert_ns_offset(test_ns);

        assert_eq!((test_ns, 0), index.ns_offset_entries[0]);
    }

    #[test]
    fn test_serialization() {
        let seed = 123;
        let mut index = Index::new(42, seed);

        // add items
        let mut keys = Vec::new();
        for i in 0..1000 {
            let key = create_test_key(i);
            index.insert_item(&key);

            if i % 20 == 0 {
                index.insert_ns_offset(i as u64);
            }

            if i % 100 == 0 {
                index.inc_block_count(1);
            }

            keys.push(key);
        }

        // serialize
        let serialized = Bytes::from(index);

        // check that we got some data
        assert!(!serialized.is_empty());

        // verify header values are preserved
        assert_eq!(u64::from_le_bytes(serialized[0..8].try_into().unwrap()), 42); // id
        assert_eq!(
            i64::from_le_bytes(serialized[8..16].try_into().unwrap()),
            seed
        ); // bloom_filter_seed
        assert_ne!(
            u64::from_le_bytes(serialized[16..24].try_into().unwrap()),
            0
        ); // bloom_filter_size
        assert_eq!(
            u64::from_le_bytes(serialized[24..32].try_into().unwrap()),
            800
        ); // ns_offset_size
        assert_eq!(
            u64::from_le_bytes(serialized[32..40].try_into().unwrap()),
            16000
        ); // block_offset_size
        assert_eq!(
            u64::from_le_bytes(serialized[40..48].try_into().unwrap()),
            10
        ); // num_blocks
    }

    #[test]
    fn test_full_serialization_roundtrip() {
        let seed = 123;
        let id = 42;
        let mut index = Index::new(id, seed);

        // add items
        let mut keys = Vec::new();
        let mut _ns_offsets = 0;
        let mut _blocks = 0;
        for i in 0..1000 {
            let key = create_test_key(i);
            index.insert_item(&key);

            if i % 20 == 0 {
                index.insert_ns_offset(i as u64);
                _ns_offsets += 1;
            }

            if i % 100 == 0 {
                index.inc_block_count(1);
                _blocks += 1;
            }

            keys.push(key);
        }

        // serialize
        let serialized = Bytes::from(index);

        // deserialize into a new index
        let deserialized = Index::from(serialized);

        // verify the indexes match
        assert_eq!(id, deserialized.id);
        assert_eq!(seed, deserialized.bloom_filter_seed);
        assert_eq!(_ns_offsets, deserialized.ns_offset_entries.len());
        assert_eq!(_blocks, deserialized.block_count());

        // verify all added keys are found in the deserialized index
        for key in &keys {
            assert!(
                deserialized.may_contain(key),
                "Failed to find key in deserialized bloom filter"
            );
        }
    }

    #[test]
    fn test_empty_index_serialization() {
        // test serializing an empty index
        let empty_index = Index::new(1, 100);
        let serialized = Bytes::from(empty_index);

        // deserialize and check it's still empty
        let deserialized = Index::from(serialized);

        assert_eq!(deserialized.id(), 1);
        assert_eq!(deserialized.block_count(), 0);
        assert_eq!(deserialized.ns_offset_count(), 0);
        assert_eq!(deserialized.block_offset_entries.len(), 0);
    }

    #[test]
    fn test_serialization_byte_order() {
        // ensure that byte order is consistently little-endian
        let index = Index::new(0x0102030405060708, 0x0102030405060708);

        // serialize
        let bytes = Bytes::from(index);

        // check id byte order (little-endian)
        assert_eq!(bytes[0], 0x08);
        assert_eq!(bytes[1], 0x07);
        assert_eq!(bytes[2], 0x06);
        assert_eq!(bytes[3], 0x05);
        assert_eq!(bytes[4], 0x04);
        assert_eq!(bytes[5], 0x03);
        assert_eq!(bytes[6], 0x02);
        assert_eq!(bytes[7], 0x01);

        // check seed byte order (little-endian)
        assert_eq!(bytes[8], 0x08);
        assert_eq!(bytes[9], 0x07);
        assert_eq!(bytes[10], 0x06);
        assert_eq!(bytes[11], 0x05);
        assert_eq!(bytes[12], 0x04);
        assert_eq!(bytes[13], 0x03);
        assert_eq!(bytes[14], 0x02);
        assert_eq!(bytes[15], 0x01);
    }

    #[test]
    fn test_false_positives() {
        // bloom filters have false positives, so let's test that property
        let mut index = Index::new(1, 100);

        // add a moderate number of items
        for i in 0..1000 {
            index.insert_item(&create_test_key(i));
        }

        // check for items we know we didn't add
        // note: this is probabilistic, so there's a small chance of real false
        // positives
        let mut false_positives = 0;
        for i in 2000..3000 {
            if index.may_contain(&create_test_key(i)) {
                false_positives += 1;
            }
        }

        // we expect some false positives but not too many
        // bloom filter false positive rates depend on size and item count
        // typically less than 1% for a reasonable configuration
        assert!(
            false_positives < 50,
            "Too many false positives: {}",
            false_positives
        );
    }

    #[test]
    fn test_find_block() {
        let seed = 42;
        let mut index = Index::new(1, seed);

        // Insert items in different blocks
        let key1 = create_test_key(101);
        let key2 = create_test_key(202);
        let key3 = create_test_key(303);

        // First block (0-indexed)
        index.insert_item(&key1);
        index.inc_block_count(1);

        // Second block
        index.insert_item(&key2);
        index.inc_block_count(1);

        // Third block
        index.insert_item(&key3);

        // Verify we can find each key in its respective block (0-indexed)
        assert_eq!(index.get_block(&key1), Some(0), "Key1 should be in block 0");
        assert_eq!(index.get_block(&key2), Some(1), "Key2 should be in block 1");
        assert_eq!(index.get_block(&key3), Some(2), "Key3 should be in block 2");

        // Test that a non-existent key returns None
        let nonexistent_key = create_test_key(999);
        assert_eq!(
            index.get_block(&nonexistent_key),
            None,
            "Non-existent key should return None"
        );

        // Test after serialization/deserialization
        let serialized = Bytes::from(&index);
        let deserialized = Index::from(serialized);

        assert_eq!(
            deserialized.get_block(&key1),
            Some(0),
            "After serialization, key1 should be in block 0"
        );
        assert_eq!(
            deserialized.get_block(&key2),
            Some(1),
            "After serialization, key2 should be in block 1"
        );
        assert_eq!(
            deserialized.get_block(&key3),
            Some(2),
            "After serialization, key3 should be in block 2"
        );
        assert_eq!(
            deserialized.get_block(&nonexistent_key),
            None,
            "After serialization, non-existent key should return None"
        );
    }

    // Regression tests for get_namespace_block() bug fix
    // Bug: The function was searching by block offset (*b) instead of namespace
    // (*n) and returning from block_offset_entries instead of ns_offset_entries

    #[test]
    fn test_get_namespace_block_basic() {
        let seed = 100;
        let mut index = Index::new(1, seed);

        // Insert namespace offsets for different namespaces
        // Each namespace is mapped to a specific block
        let ns1 = 100u64;
        let ns2 = 200u64;
        let ns3 = 300u64;

        // Namespace 100 starts at block 0 (0-indexed)
        index.insert_ns_offset(ns1);
        index.inc_block_count(1);

        // Namespace 200 starts at block 1
        index.insert_ns_offset(ns2);
        index.inc_block_count(1);

        // Namespace 300 starts at block 2
        index.insert_ns_offset(ns3);
        index.inc_block_count(1);

        // Verify we can find each namespace's starting block (0-indexed)
        assert_eq!(
            index.get_namespace_block(ns1),
            Some(0),
            "Namespace 100 should start at block 0"
        );
        assert_eq!(
            index.get_namespace_block(ns2),
            Some(1),
            "Namespace 200 should start at block 1"
        );
        assert_eq!(
            index.get_namespace_block(ns3),
            Some(2),
            "Namespace 300 should start at block 2"
        );
    }

    #[test]
    fn test_get_namespace_block_not_found() {
        let seed = 100;
        let mut index = Index::new(1, seed);

        // Insert a single namespace
        let ns = 100u64;
        index.insert_ns_offset(ns);

        // Look for a namespace that doesn't exist
        assert_eq!(
            index.get_namespace_block(999),
            None,
            "Non-existent namespace should return None"
        );
    }

    #[test]
    fn test_get_namespace_block_empty_index() {
        let seed = 100;
        let index = Index::new(1, seed);

        // Try to find namespace in empty index
        assert_eq!(
            index.get_namespace_block(100),
            None,
            "Empty index should return None for any namespace"
        );
    }

    #[test]
    fn test_get_namespace_block_searches_by_namespace_not_block() {
        // This test specifically validates that the function searches by namespace
        // and not by block offset, which was the bug.
        let seed = 100;
        let mut index = Index::new(1, seed);

        // Create a scenario where namespace values and block offsets differ
        // significantly This ensures the binary search is using the right field

        // Namespace 50 maps to block 0
        index.insert_ns_offset(50);

        // Namespace 150 maps to block 1 (increment block count first)
        index.inc_block_count(1);
        index.insert_ns_offset(150);

        // Namespace 300 maps to block 2
        index.inc_block_count(1);
        index.insert_ns_offset(300);

        // If the bug existed, searching for namespace 50 would fail
        // because it would try to search by block offset instead of namespace
        assert_eq!(
            index.get_namespace_block(50),
            Some(0),
            "Should find namespace 50 at block 0"
        );
        assert_eq!(
            index.get_namespace_block(150),
            Some(1),
            "Should find namespace 150 at block 1"
        );
        assert_eq!(
            index.get_namespace_block(300),
            Some(2),
            "Should find namespace 300 at block 2"
        );

        // These should not be found (proving we're searching by namespace, not block
        // offset)
        assert_eq!(
            index.get_namespace_block(1),
            None,
            "Block offset 0 should not be found as a namespace"
        );
        assert_eq!(
            index.get_namespace_block(2),
            None,
            "Block offset 1 should not be found as a namespace"
        );
        assert_eq!(
            index.get_namespace_block(3),
            None,
            "Block offset 2 should not be found as a namespace"
        );

        // More specifically, test that we're NOT finding by block values
        // If the bug existed and we searched by the second field (block offset),
        // searching for "1" might incorrectly find the entry for namespace 50
        assert_eq!(
            index.get_namespace_block(49),
            None,
            "Namespace 49 should not exist"
        );
        assert_eq!(
            index.get_namespace_block(51),
            None,
            "Namespace 51 should not exist"
        );
    }

    #[test]
    fn test_get_namespace_block_returns_from_correct_array() {
        // This test validates that the function returns values from ns_offset_entries
        // and not from block_offset_entries, which was part of the bug
        let seed = 100;
        let mut index = Index::new(1, seed);

        // Add namespace offsets
        let ns1 = 1000u64;
        let ns2 = 2000u64;

        index.insert_ns_offset(ns1);
        index.inc_block_count(1);

        index.insert_ns_offset(ns2);
        index.inc_block_count(1);

        // Also add many more block offset entries (for keys) to create a clear
        // difference
        for i in 0..50 {
            let key = create_test_key(i);
            index.insert_item(&key);
        }

        // Verify that namespace lookups return the correct block offsets
        // from ns_offset_entries, not from block_offset_entries
        let result1 = index.get_namespace_block(ns1);
        let result2 = index.get_namespace_block(ns2);

        assert_eq!(
            result1,
            Some(0),
            "Namespace 1000 should map to block 0 from ns_offset_entries"
        );
        assert_eq!(
            result2,
            Some(1),
            "Namespace 2000 should map to block 1 from ns_offset_entries"
        );

        // Verify the arrays are actually different in size
        assert_eq!(
            index.ns_offset_entries.len(),
            2,
            "Should have exactly 2 namespace entries"
        );
        assert!(
            index.block_offset_entries.len() >= 50,
            "Should have at least 50 block offset entries, demonstrating they're distinct arrays"
        );
    }

    #[test]
    fn test_get_namespace_block_multiple_namespaces() {
        let seed = 100;
        let mut index = Index::new(1, seed);

        // Insert many namespaces to test binary search correctness
        let namespaces = vec![10u64, 20, 30, 40, 50, 60, 70, 80, 90, 100];

        for (_i, ns) in namespaces.iter().enumerate() {
            index.insert_ns_offset(*ns);
            index.inc_block_count(1);
        }

        // Verify all namespaces can be found correctly
        for (i, ns) in namespaces.iter().enumerate() {
            let expected_block = i as u64;
            assert_eq!(
                index.get_namespace_block(*ns),
                Some(expected_block),
                "Namespace {} should map to block {}",
                ns,
                expected_block
            );
        }

        // Verify non-existent namespaces return None
        assert_eq!(index.get_namespace_block(5), None);
        assert_eq!(index.get_namespace_block(15), None);
        assert_eq!(index.get_namespace_block(105), None);
    }

    #[test]
    fn test_get_namespace_block_boundary_values() {
        let seed = 100;
        let mut index = Index::new(1, seed);

        // Test with boundary values
        let ns_min = 0u64;
        let ns_max = u64::MAX;
        let ns_mid = u64::MAX / 2;

        index.insert_ns_offset(ns_min);
        index.inc_block_count(1);

        index.insert_ns_offset(ns_mid);
        index.inc_block_count(1);

        index.insert_ns_offset(ns_max);
        index.inc_block_count(1);

        assert_eq!(
            index.get_namespace_block(ns_min),
            Some(0),
            "Should handle minimum u64 value"
        );
        assert_eq!(
            index.get_namespace_block(ns_mid),
            Some(1),
            "Should handle mid-range u64 value"
        );
        assert_eq!(
            index.get_namespace_block(ns_max),
            Some(2),
            "Should handle maximum u64 value"
        );
    }

    #[test]
    fn test_get_namespace_block_after_serialization() {
        let seed = 100;
        let mut index = Index::new(1, seed);

        // Setup namespaces with distinct values from block offsets
        let ns1 = 1234u64;
        let ns2 = 5678u64;
        let ns3 = 9999u64;

        index.insert_ns_offset(ns1);
        index.inc_block_count(1);

        index.insert_ns_offset(ns2);
        index.inc_block_count(1);

        index.insert_ns_offset(ns3);
        index.inc_block_count(1);

        // Serialize and deserialize
        let serialized = Bytes::from(index);
        let deserialized = Index::from(serialized);

        // Verify namespace lookups work correctly after deserialization
        assert_eq!(
            deserialized.get_namespace_block(ns1),
            Some(0),
            "After deserialization, namespace 1234 should map to block 0"
        );
        assert_eq!(
            deserialized.get_namespace_block(ns2),
            Some(1),
            "After deserialization, namespace 5678 should map to block 1"
        );
        assert_eq!(
            deserialized.get_namespace_block(ns3),
            Some(2),
            "After deserialization, namespace 9999 should map to block 2"
        );

        // Verify non-existent namespace returns None
        assert_eq!(
            deserialized.get_namespace_block(4321),
            None,
            "After deserialization, non-existent namespace should return None"
        );
    }

    #[test]
    fn test_get_namespace_block_with_mixed_operations() {
        // Test namespace lookups in a realistic scenario with mixed operations
        let seed = 100;
        let mut index = Index::new(1, seed);

        // Simulate a realistic workload with keys and namespaces
        let ns1 = 100u64;
        let ns2 = 200u64;

        // Block 1: namespace 100
        index.insert_ns_offset(ns1);
        index.inc_block_count(1);

        // Add some keys
        for i in 0..10 {
            let key = create_test_key_ns(i, ns1);
            index.insert_item(&key);
        }

        // Block 2: namespace 200
        index.insert_ns_offset(ns2);
        index.inc_block_count(1);

        // Add more keys
        for i in 10..20 {
            let key = create_test_key_ns(i, ns2);
            index.insert_item(&key);
        }

        // Verify namespace lookups work correctly
        assert_eq!(
            index.get_namespace_block(ns1),
            Some(0),
            "Namespace 100 should be at block 0"
        );
        assert_eq!(
            index.get_namespace_block(ns2),
            Some(1),
            "Namespace 200 should be at block 1"
        );

        // Verify we have entries in both arrays
        assert!(
            index.ns_offset_entries.len() >= 2,
            "Should have at least 2 namespace entries"
        );
        assert!(
            index.block_offset_entries.len() >= 20,
            "Should have at least 20 block offset entries"
        );
    }
}
