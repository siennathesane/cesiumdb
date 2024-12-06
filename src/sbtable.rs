use std::fs::File;
use bytes::Bytes;

pub(crate) struct SBTable {
    key_fd: File,
    value_fd: File,
    key_metadata: SBKeyTableMetadata,
    value_metadata: SBValueTableMetadata,
}

pub(crate) struct SBKeyTableMetadata {
    bloom_filter_seed: u64,
    bloom_filter_size: u64,
    bloom_filter_offset: u64,
    bloom_filter: Bytes,
}

pub(crate) struct SBValueTableMetadata {
    value_idx_size: u64,
    values: Bytes,
}

pub(crate) struct Entry {
    hash: u64,
    block: u64,
}

impl Entry {
    pub(crate) fn new(hash: u64, block: u64) -> Self {
        Self { hash, block }
    }    
    
    pub(crate) fn hash(&self) -> u64 {
        self.hash
    }
    
    pub(crate) fn set_hash(&mut self, hash: u64) {
        self.hash = hash;
    }
    
    pub(crate) fn block(&self) -> u64 {
        self.block
    }
    
    pub(crate) fn set_block(&mut self, block: u64) {
        self.block = block;
    }
}
