use std::{
    hash::RandomState,
    sync::Arc,
};

use bloom2::{
    Bloom2,
    BloomFilterBuilder,
    BytesBitmap,
    CompressedBitmap,
    FilterSize::KeyBytes3,
};
use bytes::{
    BufMut,
    Bytes,
    BytesMut,
};
use gxhash::{
    GxBuildHasher,
    GxHasher,
    gxhash64,
};

use crate::{
    hash::SeedableHasher,
    utils::{
        Deserializer,
        Serializer,
    },
};

/// The value at which the bloom filter has a 50% probability of false positives
/// for 3-byte key storage
const BLOOM_OVERRIDE: usize = 10300768;

/// integrated index that combines bloom filtering with block-level lookup
pub struct Index {
    // serialized header fields
    id: u64,
    bloom_filter_seed: i64,
    bloom_filter_size: u64,
    bloom_filter_offset: u64,
    ns_offset_size: u64,
    block_offset_size: u64,
    block_starting_keys_hash_offsets_size: u64,

    // serialized data fields
    block_starting_key_hash_offsets: BytesMut,
    block_offsets: BytesMut,
    ns_offsets: BytesMut,
    bloom_filter: BytesMut,

    // in-memory only fields
    active_bloom: Bloom2<SeedableHasher, BytesBitmap, u64>,
    block_entries: Vec<(u64, u64)>, // (hash, block_offset) pairs
}

impl Index {
    /// create a new index with the specified id and bloom filter seed
    pub fn new(id: u64, seed: i64) -> Self {
        let hasher = SeedableHasher::new(seed);
        Self {
            id,
            bloom_filter_seed: seed,
            bloom_filter_size: 0,
            bloom_filter_offset: 0,
            ns_offset_size: 0,
            block_offset_size: 0,
            block_starting_keys_hash_offsets_size: 0,
            block_starting_key_hash_offsets: BytesMut::new(),
            block_offsets: BytesMut::new(),
            ns_offsets: BytesMut::new(),
            bloom_filter: BytesMut::new(),
            active_bloom: BloomFilterBuilder::hasher(hasher)
                .with_bitmap()
                .size(KeyBytes3)
                .build(),
            block_entries: Vec::new(),
        }
    }

    /// add an item to the bloom filter
    pub fn add_item(&mut self, item: &[u8]) {
        let hash = gxhash64(item, self.bloom_filter_seed);
        self.active_bloom.insert(&hash);
    }

    /// add a new block with the given starting key
    pub fn add_block(&mut self, starting_key: &[u8]) {
        let hash = gxhash64(starting_key, self.bloom_filter_seed);

        // add to serialized block offsets
        self.block_offset_size += 1;
        self.block_offsets.extend_from_slice(&hash.to_le_bytes());

        // add to in-memory block index
        let block_offset = (self.block_offset_size - 1);
        self.insert_block_entry(hash, block_offset);
    }

    /// add a namespace offset to the most recent block
    // Ensure no underflow occurs in the implementation
    pub fn add_ns_offset(&mut self, ns: u64) {
        self.ns_offset_size += 1;
        // Ensure there's at least one block before accessing block_offsets
        if self.block_offsets.is_empty() {
            // Handle the case where there are no blocks yet
            return;
        }
        let cur_block_offset = self.block_offsets[self.block_offsets.len() - 8..].as_ref();
        self.ns_offsets.extend_from_slice(cur_block_offset);
    }

    /// check if a key might be present in the index
    pub fn may_contain(&self, key: &[u8]) -> bool {
        let hash = gxhash64(key, self.bloom_filter_seed);
        self.active_bloom.contains(&hash)
    }

    /// find the block that would contain this key
    pub fn find_block(&self, key: &[u8]) -> Option<u64> {
        let hash = gxhash64(key, self.bloom_filter_seed);
        self.get_block(hash)
    }

    /// get the id of this index
    pub fn id(&self) -> u64 {
        self.id
    }

    /// get the total number of blocks in the index
    pub fn block_count(&self) -> u64 {
        self.block_offset_size
    }

    /// get the total number of namespace offsets in the index
    pub fn ns_offset_count(&self) -> u64 {
        self.ns_offset_size
    }

    // internal methods that were previously in BlockIndex

    /// insert a block entry into the in-memory block index
    fn insert_block_entry(&mut self, hash: u64, block_offset: u64) {
        match self.block_entries.binary_search_by_key(&hash, |(h, _)| *h) {
            | Ok(idx) => self.block_entries[idx].1 = block_offset,
            | Err(idx) => self.block_entries.insert(idx, (hash, block_offset)),
        }
    }

    /// get a block offset by hash from the in-memory block index
    fn get_block(&self, hash: u64) -> Option<u64> {
        self.block_entries
            .binary_search_by_key(&hash, |(h, _)| *h)
            .ok()
            .map(|idx| self.block_entries[idx].1)
    }

    /// get number of block entries in the in-memory index
    fn block_entries_len(&self) -> usize {
        self.block_entries.len()
    }
}

impl Serializer for Index {
    fn serialize_for_memory(&self) -> Bytes {
        // unimplemented per source specification
        unimplemented!()
    }

    fn serialize(&self) -> Bytes {
        // serialize bloom filter - get a complete copy of the bitmap data
        let bloom_data = self.active_bloom.bitmap().clone().freeze();
        let bloom_size = bloom_data.len() as u64;

        // calculate offsets for the serialized layout
        let header_size = 56; // 7 u64/i64 fields at 8 bytes each
        let block_starting_keys_offset = header_size;
        let block_offsets_offset =
            block_starting_keys_offset + self.block_starting_keys_hash_offsets_size;
        let ns_offsets_offset = block_offsets_offset + (self.block_offset_size * 8);
        let bloom_filter_offset = ns_offsets_offset + (self.ns_offset_size * 8);

        let total_size = bloom_filter_offset + bloom_size;
        let mut buf = BytesMut::with_capacity(total_size as usize);

        // write header
        buf.put_u64_le(self.id);
        buf.put_i64_le(self.bloom_filter_seed);
        buf.put_u64_le(bloom_size);
        buf.put_u64_le(bloom_filter_offset);
        buf.put_u64_le(self.ns_offset_size);
        buf.put_u64_le(self.block_offset_size);
        buf.put_u64_le(self.block_starting_keys_hash_offsets_size);

        // write data sections
        buf.put(self.block_starting_key_hash_offsets.as_ref());
        buf.put(self.block_offsets.as_ref());
        buf.put(self.ns_offsets.as_ref());
        buf.put(bloom_data.as_ref());

        buf.freeze()
    }
}

impl Deserializer for Index {
    fn deserialize_from_memory(payload: Bytes) -> Self {
        // unimplemented per source specification
        unimplemented!()
    }

    fn deserialize(payload: Bytes) -> Self {
        // extract header fields
        let id = u64::from_le_bytes(payload[0..8].try_into().unwrap());
        let bloom_filter_seed = i64::from_le_bytes(payload[8..16].try_into().unwrap());
        let bloom_filter_size = u64::from_le_bytes(payload[16..24].try_into().unwrap());
        let bloom_filter_offset = u64::from_le_bytes(payload[24..32].try_into().unwrap());
        let ns_offset_size = u64::from_le_bytes(payload[32..40].try_into().unwrap());
        let block_offset_size = u64::from_le_bytes(payload[40..48].try_into().unwrap());
        let block_starting_keys_hash_offsets_size =
            u64::from_le_bytes(payload[48..56].try_into().unwrap());

        // extract data sections using computed offsets
        let header_size = 56;
        let block_starting_key_hash_offsets = BytesMut::from(
            &payload[header_size..header_size + block_starting_keys_hash_offsets_size as usize],
        );

        let block_offsets_start = header_size + block_starting_keys_hash_offsets_size as usize;
        let block_offsets_end = block_offsets_start + block_offset_size as usize * 8;
        let block_offsets = BytesMut::from(&payload[block_offsets_start..block_offsets_end]);

        let ns_offsets = BytesMut::from(
            &payload[block_offsets_end..block_offsets_end + ns_offset_size as usize * 8],
        );

        // extract bloom filter data
        let bloom_data =
            if bloom_filter_offset as usize + bloom_filter_size as usize <= payload.len() {
                Bytes::copy_from_slice(
                    &payload[bloom_filter_offset as usize..
                        bloom_filter_offset as usize + bloom_filter_size as usize],
                )
            } else {
                Bytes::new()
            };

        // recreate the bloom filter
        let hasher = SeedableHasher::new(bloom_filter_seed);
        // In the deserialize method, update the bloom filter creation:
        let active_bloom = if !bloom_data.is_empty() {
            // Create bitmap from the serialized bytes
            let bitmap = BytesBitmap::from_bytes(bloom_data.clone());

            // Ensure we're using the correct filter size and configuration
            BloomFilterBuilder::hasher(hasher)
                .with_bitmap()
                .with_bitmap_data(bitmap, KeyBytes3)
                .build()
        } else {
            BloomFilterBuilder::hasher(hasher)
                .with_bitmap()
                .size(KeyBytes3)
                .build()
        };

        // build the block entries from the block offsets
        let mut block_entries = Vec::new();
        for i in 0..block_offset_size as usize {
            let offset = i * 8;
            if offset + 8 <= block_offsets.len() {
                let hash_bytes = &block_offsets[offset..offset + 8];
                let hash = u64::from_le_bytes(hash_bytes.try_into().unwrap());
                let block_offset = i as u64;

                // insert maintaining sorted order
                match block_entries.binary_search_by_key(&hash, |(h, _)| *h) {
                    | Ok(idx) => block_entries[idx].1 = block_offset,
                    | Err(idx) => block_entries.insert(idx, (hash, block_offset)),
                }
            }
        }

        Self {
            id,
            bloom_filter_seed,
            bloom_filter_size,
            bloom_filter_offset,
            ns_offset_size,
            block_offset_size,
            block_starting_keys_hash_offsets_size,
            block_starting_key_hash_offsets,
            block_offsets,
            ns_offsets,
            bloom_filter: BytesMut::new(), // not needed after deserialization
            active_bloom,
            block_entries,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use bytes::Bytes;

    use super::*;

    // helper function to create test data
    fn create_test_key(id: u32) -> Vec<u8> {
        format!("test_key_{}", id).into_bytes()
    }

    #[test]
    fn test_new_index() {
        let index = Index::new(42, 123);
        assert_eq!(index.id(), 42);
        assert_eq!(index.block_count(), 0);
        assert_eq!(index.ns_offset_count(), 0);
        assert_eq!(index.block_entries_len(), 0);
    }

    #[test]
    fn test_add_item() {
        let mut index = Index::new(1, 100);

        // add some items
        let item1 = create_test_key(1);
        let item2 = create_test_key(2);

        index.add_item(&item1);
        index.add_item(&item2);

        // bloom filter should now contain these items
        assert!(index.may_contain(&item1));
        assert!(index.may_contain(&item2));

        // but not some random key
        let non_existent = create_test_key(999);
        assert!(!index.may_contain(&non_existent));
    }

    #[test]
    fn test_add_block() {
        let mut index = Index::new(1, 100);

        // add some blocks with starting keys
        let key1 = create_test_key(10);
        let key2 = create_test_key(20);
        let key3 = create_test_key(30);

        index.add_block(&key1);
        index.add_block(&key2);
        index.add_block(&key3);

        // check block count
        assert_eq!(index.block_count(), 3);

        // verify hashes are in the block entries
        let hash1 = gxhash64(&key1, index.bloom_filter_seed);
        let hash2 = gxhash64(&key2, index.bloom_filter_seed);
        let hash3 = gxhash64(&key3, index.bloom_filter_seed);

        assert_eq!(index.get_block(hash1), Some(0));
        assert_eq!(index.get_block(hash2), Some(1));
        assert_eq!(index.get_block(hash3), Some(2));

        // non-existent hash should return None
        assert_eq!(index.get_block(12345), None);
    }

    #[test]
    fn test_find_block() {
        let mut index = Index::new(1, 100);

        // add some blocks with starting keys
        let key1 = create_test_key(10);
        let key2 = create_test_key(20);
        let key3 = create_test_key(30);

        index.add_block(&key1);
        index.add_block(&key2);
        index.add_block(&key3);

        // find block using the same keys
        assert_eq!(index.find_block(&key1), Some(0));
        assert_eq!(index.find_block(&key2), Some(1));
        assert_eq!(index.find_block(&key3), Some(2));

        // non-existent key should return None
        let non_existent = create_test_key(999);
        assert_eq!(index.find_block(&non_existent), None);
    }

    #[test]
    fn test_add_ns_offset() {
        let mut index = Index::new(1, 100);

        // add a block first
        let key1 = create_test_key(10);
        index.add_block(&key1);

        // add namespace offsets
        index.add_ns_offset(100);
        index.add_ns_offset(200);

        // check ns offset count
        assert_eq!(index.ns_offset_count(), 2);

        // add another block
        let key2 = create_test_key(20);
        index.add_block(&key2);

        // add namespace offsets to the second block
        index.add_ns_offset(300);

        // check updated count
        assert_eq!(index.ns_offset_count(), 3);
    }

    #[test]
    fn test_serialization() {
        let mut index = Index::new(42, 123);

        // add items
        for i in 0..10 {
            let key = create_test_key(i);
            index.add_item(&key);
        }

        // add blocks
        index.add_block(&create_test_key(100));
        index.add_block(&create_test_key(200));

        // add namespace offsets
        index.add_ns_offset(1000);
        index.add_ns_offset(2000);

        // serialize
        let serialized = index.serialize();

        // check that we got some data
        assert!(!serialized.is_empty());

        // verify header values are preserved
        assert_eq!(u64::from_le_bytes(serialized[0..8].try_into().unwrap()), 42); // id
        assert_eq!(
            i64::from_le_bytes(serialized[8..16].try_into().unwrap()),
            123
        ); // bloom_filter_seed
        assert_eq!(
            u64::from_le_bytes(serialized[32..40].try_into().unwrap()),
            2
        ); // ns_offset_size
        assert_eq!(
            u64::from_le_bytes(serialized[40..48].try_into().unwrap()),
            2
        ); // block_offset_size
    }

    #[test]
    fn test_full_serialization_roundtrip() {
        let mut original = Index::new(42, 123);

        // add a bunch of items
        let mut added_keys = Vec::new();
        for i in 0..100 {
            let key = create_test_key(i);
            original.add_item(&key);
            added_keys.push(key);
        }

        // add blocks at specific positions
        let block_keys = vec![
            create_test_key(10),
            create_test_key(30),
            create_test_key(70),
        ];

        for key in &block_keys {
            original.add_block(key);
            // Make sure items in block_keys are also in the bloom filter
            original.add_item(key);
        }

        // add namespace offsets
        original.add_ns_offset(100);
        original.add_ns_offset(200);
        original.add_ns_offset(300);

        // serialize
        let serialized = original.serialize();

        // deserialize into a new index
        let deserialized = Index::deserialize(serialized);

        // verify the indexes match
        assert_eq!(deserialized.id(), original.id());
        assert_eq!(deserialized.block_count(), original.block_count());
        assert_eq!(deserialized.ns_offset_count(), original.ns_offset_count());

        // verify all added keys are found in the deserialized index
        for key in &added_keys {
            assert!(
                deserialized.may_contain(key),
                "Failed to find key in deserialized bloom filter"
            );
        }

        // verify block lookup works after deserialization
        for key in &block_keys {
            assert_eq!(deserialized.find_block(key), original.find_block(key));
        }
    }

    #[test]
    fn test_block_collision_handling() {
        // create a mock hasher that always returns the same hash to test collision
        // handling
        let mut index = Index::new(1, 100);

        // add blocks with different keys but potentially same hash
        let key1 = create_test_key(1);
        let key2 = create_test_key(2);

        index.add_block(&key1);

        // manually insert a collision to test the handling
        let hash = gxhash64(&key1, index.bloom_filter_seed);
        let block_offset = 999; // different offset
        index.insert_block_entry(hash, block_offset);

        // check that the collision was handled by updating the existing entry
        assert_eq!(index.get_block(hash), Some(block_offset));
        assert_eq!(index.block_entries_len(), 1); // still only one entry
    }

    #[test]
    fn test_empty_index_serialization() {
        // test serializing an empty index
        let empty_index = Index::new(1, 100);
        let serialized = empty_index.serialize();

        // deserialize and check it's still empty
        let deserialized = Index::deserialize(serialized);

        assert_eq!(deserialized.id(), 1);
        assert_eq!(deserialized.block_count(), 0);
        assert_eq!(deserialized.ns_offset_count(), 0);
        assert_eq!(deserialized.block_entries_len(), 0);
    }

    #[test]
    fn test_large_index() {
        // test with a large number of items to ensure we don't hit any limits
        let mut index = Index::new(1, 100);

        // add many items
        for i in 0..1000 {
            index.add_item(&create_test_key(i));

            // add a block every 100 items
            if i % 100 == 0 {
                index.add_block(&create_test_key(i));
                index.add_ns_offset(i as u64);
            }
        }

        // check counts - we have blocks at 0, 100, 200, ..., 900 (10 blocks total)
        let expected_block_count = 10;
        assert_eq!(index.block_count(), expected_block_count);
        assert_eq!(index.ns_offset_count(), expected_block_count);

        // serialize and deserialize
        let serialized = index.serialize();
        let deserialized = Index::deserialize(serialized);

        // verify everything is preserved
        assert_eq!(deserialized.block_count(), index.block_count());
        assert_eq!(deserialized.ns_offset_count(), index.ns_offset_count());

        // check a few random lookups
        for i in (0..1000).step_by(137) {
            // use a prime step to get good coverage
            let key = create_test_key(i);
            assert!(
                deserialized.may_contain(&key),
                "Failed to find key {} in deserialized bloom filter",
                i
            );
        }
    }

    #[test]
    fn test_block_lookup_performance() {
        let mut index = Index::new(1, 100);

        // add a large number of blocks
        let mut block_keys = Vec::new();
        for i in 0..100 {
            // reduced from 10000 to 100 for test speed
            let key = create_test_key(i);
            block_keys.push(key.clone());
            index.add_block(&key);
        }

        // verify we can find all blocks efficiently
        for (i, key) in block_keys.iter().enumerate() {
            assert_eq!(index.find_block(key), Some(i as u64));
        }
    }

    #[test]
    fn test_update_block_offset() {
        let mut index = Index::new(1, 100);

        // add a block
        let key = create_test_key(1);
        index.add_block(&key);

        let hash = gxhash64(&key, index.bloom_filter_seed);

        // verify initial offset
        assert_eq!(index.get_block(hash), Some(0));

        // update the offset
        index.insert_block_entry(hash, 42);

        // verify updated offset
        assert_eq!(index.get_block(hash), Some(42));
    }

    #[test]
    fn test_serialization_byte_order() {
        // ensure that byte order is consistently little-endian
        let index = Index::new(0x0102030405060708, 0x0102030405060708);

        // serialize
        let bytes = index.serialize();

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
            index.add_item(&create_test_key(i));
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
    fn test_block_with_same_starting_key() {
        let mut index = Index::new(1, 100);

        // add the same block key twice
        let key = create_test_key(1);

        index.add_block(&key);
        index.add_block(&key);

        // we should have two blocks
        assert_eq!(index.block_count(), 2);

        // but the hash lookup should point to the last one
        let hash = gxhash64(&key, index.bloom_filter_seed);
        assert_eq!(index.get_block(hash), Some(1));
    }

    #[test]
    fn test_index_with_no_bloom_data() {
        // test creating an index without bloom data and serializing it
        let index = Index::new(1, 100);
        // don't add any items, so bloom filter will be empty

        // serialize and deserialize
        let serialized = index.serialize();
        let deserialized = Index::deserialize(serialized);

        // check a key that wasn't added
        assert!(!deserialized.may_contain(&create_test_key(1)));
    }

    #[test]
    fn test_bloom_filter_serialization() {
        // Specific test for bloom filter serialization
        let mut original = Index::new(1, 100);

        // Add a single item to the bloom filter
        let test_key = create_test_key(42);
        original.add_item(&test_key);

        // Verify it's in the original
        assert!(original.may_contain(&test_key));

        // Serialize and deserialize
        let serialized = original.serialize();
        let deserialized = Index::deserialize(serialized);

        // Test specific keys
        assert!(
            deserialized.may_contain(&test_key),
            "Key should be found in deserialized bloom filter"
        );

        // Test a key we didn't add
        let missing_key = create_test_key(999);
        assert!(
            !deserialized.may_contain(&missing_key),
            "Key should NOT be found in deserialized bloom filter"
        );
    }
}
