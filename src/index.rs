use std::{
    hash::RandomState,
    sync::Arc,
};

use bloom2::{Bloom2, BloomFilterBuilder, CompressedBitmap};
use bloom2::FilterSize::KeyBytes3;
use bytes::{BufMut, Bytes, BytesMut};
use gxhash::gxhash64;
use crate::utils::{Deserializer, Serializer};

pub(crate) struct SegmentIndex {
    // serialized fields
    id: u64,
    bloom_filter_seed: i64,
    bloom_filter_size: u64,
    bloom_filter_offset: u64,
    ns_offset_size: u64,
    block_offset_size: u64,
    block_starting_keys_hash_offsets_size: u64,
    
    // serialized fields
    block_starting_key_hash_offsets: BytesMut,
    block_offsets: BytesMut,
    ns_offsets: BytesMut,
    bloom_filter: BytesMut,

    // temporary fields
    active_bloom: Bloom2<RandomState, CompressedBitmap, u64>,
}

impl SegmentIndex {
    fn new(idx: u64, seed: i64) -> Arc<Self> {
        Arc::new(Self {
            id: idx,
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
            active_bloom: BloomFilterBuilder::default().size(KeyBytes3).build(),
        })
    }

    /// Add an item to the index.
    pub(crate) fn add_item(&mut self, item: Bytes) {
        self.active_bloom.insert(&gxhash64(&item, self.bloom_filter_seed));
    }

    /// Add a block to the index.
    pub(crate) fn add_block(&mut self, starting_key: Bytes) {
        self.block_offset_size += 1;
        self.block_offsets.extend_from_slice(&gxhash64(&starting_key, self.bloom_filter_seed).to_le_bytes());
    }

    /// Add a namespace offset to the most recently added block.
    pub(crate) fn add_ns_offset(&mut self, ns :u64) {
        self.ns_offset_size += 1;
        let cur_block_offset = self.block_offsets[self.block_offsets.len() - 8..].as_ref();
        self.ns_offsets.extend_from_slice(u64::from_le_bytes(cur_block_offset.try_into().unwrap()).to_le_bytes().as_ref());
    }
}

impl Serializer for SegmentIndex {
    fn serialize_for_memory(&self) -> Bytes {
        unimplemented!()
    }

    fn serialize(&self) -> Bytes {
        let mut buf = BytesMut::new();
        buf.put_u64_le(self.id);
        buf.put_i64_le(self.bloom_filter_seed);
        buf.put_u64_le(self.bloom_filter_size);
        buf.put_u64_le(self.bloom_filter_offset);
        buf.put_u64_le(self.ns_offset_size);
        buf.put_u64_le(self.block_offset_size);
        buf.put_u64_le(self.block_starting_keys_hash_offsets_size);
        buf.put(self.block_starting_key_hash_offsets.as_ref());
        buf.put(self.block_offsets.as_ref());
        buf.put(self.ns_offsets.as_ref());
        buf.put(self.bloom_filter.as_ref());
        buf.freeze()
    }
}

impl Deserializer for SegmentIndex {
    fn deserialize_from_memory(payload: Bytes) -> Self {
        unimplemented!()
    }

    // TODO(@siennathesane): test this
    fn deserialize(payload: Bytes) -> Self {
        let id = u64::from_le_bytes(payload[0..8].try_into().unwrap());
        let bloom_filter_seed = i64::from_le_bytes(payload[8..16].try_into().unwrap());
        let bloom_filter_size = u64::from_le_bytes(payload[16..24].try_into().unwrap());
        let bloom_filter_offset = u64::from_le_bytes(payload[24..32].try_into().unwrap());
        let ns_offset_size = u64::from_le_bytes(payload[32..40].try_into().unwrap());
        let block_offset_size = u64::from_le_bytes(payload[40..48].try_into().unwrap());
        let block_starting_keys_hash_offsets_size = u64::from_le_bytes(payload[48..56].try_into().unwrap());
        let block_starting_key_hash_offsets = BytesMut::from(&payload[56..56 + block_starting_keys_hash_offsets_size as usize]);
        let block_offsets = BytesMut::from(&payload[56 + block_starting_keys_hash_offsets_size as usize..56 + block_starting_keys_hash_offsets_size as usize + block_offset_size as usize * 8]);
        let ns_offsets = BytesMut::from(&payload[56 + block_starting_keys_hash_offsets_size as usize + block_offset_size as usize * 8..56 + block_starting_keys_hash_offsets_size as usize + block_offset_size as usize * 8 + ns_offset_size as usize * 8]);
        let bloom_filter = BytesMut::from(&payload[56 + block_starting_keys_hash_offsets_size as usize + block_offset_size as usize * 8 + ns_offset_size as usize * 8..]);
        
        let active_bloom = BloomFilterBuilder::default().size(KeyBytes3).build();
        
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
            bloom_filter,
            active_bloom,
        }
    }
}