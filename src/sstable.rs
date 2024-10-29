//! SSTables are made of two separate components: a key table and a value table.
//! This is to optimize disk reads. When an SSTable is created, it will spawn
//! three threads: a key reader thread and two value reader threads. Generally,
//! the data flow of a read looks like this:
//!
//! [`State`] -> [`SSTable`] -> [`KeyThreadChannel`] -> [`BloomFilter`] ->
//! [`LruCache`] -> [`ValueThreadChannel`] -> [`LruCache`] -> [`Value`]
//!
//! This allows us to have unmanaged background threads without requiring async
//! and still sharing memory across requests. It also allows us some basic load
//! balancing across multiple threads without requiring locking.
//!
//! The SSTable stores the [`KeyTable`] as `id.0` and the [`ValueTable`] as
//! `id.1` to easily differentiate the difference between the filetypes in a
//! recovery scenario.

use bytes::{
    BufMut,
    Bytes,
    BytesMut,
};

pub const DEFAULT_BLOCK_SIZE: u64 = 4096;
pub const DEFAULT_SUPERBLOCK_SIZE: u64 = 2 << 23; // 8MiB

pub(crate) struct SortedBinaryTable {
    /// File descriptor for the key table
    fd: i64,
    /// File descriptor for the value table
    value_fd: i64,
}

impl SortedBinaryTable {
    pub(crate) fn new(fd: i64, value_fd: i64) -> Self {
        Self { fd, value_fd }
    }
}

#[repr(C)]
pub(crate) struct KeyTable {
    // disk layout
    table_id: f64,
    size: u64,
    filter_size: u64,
    index_size: u64,
    seed: u64,
    filter: Bytes,     // Vec<u8>
    ns_offsets: Bytes, // Vec<u64>
    index: Bytes,      // Vec<KeyMetadata>
}

impl KeyTable {
    pub(crate) fn new(
        table_id: f64,
        size: u64,
        filter_size: u64,
        index_size: u64,
        seed: u64,
        filter: Bytes,
        ns_offsets: Bytes,
        index: Bytes,
    ) -> Self {
        Self {
            table_id,
            size,
            filter_size,
            index_size,
            seed,
            filter,
            ns_offsets,
            index,
        }
    }

    pub(crate) fn serialize(&self) {
        let size = size_of_val(&self);
        let mut buf = BytesMut::with_capacity(size);

        buf.put_f64(self.table_id);
        buf.put_u64(self.size);
        buf.put_u64(self.filter_size);
        buf.put_u64(self.index_size);
        buf.put_u64(self.seed);
        buf.put(self.filter.clone());
        buf.put(self.ns_offsets.clone());
        buf.put(self.index.clone());
    }
}

#[repr(C)]
pub(crate) struct ValueTable {
    table_id: f64,
    size: u64,
    index_size: u64,
    ns_offsets: Bytes, // Vec<u64>
    index: Bytes,      // Vec<ValueMetadata>
}

/// Disk metadata about a key
#[repr(C)]
struct KeyMetadata {
    ns: u64,
    offset: u64,
    size: u64,
    padding: u64,
}

impl KeyMetadata {
    fn new(ns: u64, offset: u64, size: u64) -> Self {
        Self {
            ns,
            offset,
            size,
            padding: 0,
        }
    }
}

/// Disk metadata about a key
#[repr(C)]
struct ValueMetadata {
    ns: u64,
    offset: u64,
    size: u64,
    padding: u64,
}

impl ValueMetadata {
    fn new(ns: u64, offset: u64, size: u64) -> Self {
        Self {
            ns,
            offset,
            size,
            padding: 0,
        }
    }
}

#[repr(C)]
struct NsOffset {
    ns: u64,
    offset: u64,
}

impl NsOffset {
    fn new(ns: u64, offset: u64) -> Self {
        Self { ns, offset }
    }
}
