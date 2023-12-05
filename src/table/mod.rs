use moka::sync::Cache;
use serde::{
    Deserialize,
    Serialize,
};

use crate::{
    types::buffers::DataKey,
    ChecksumVerificationMode,
    CompressionType,
};

/// The SSTable block builder.
///
/// The core data layout for numerics is little endian.
///
/// The core structure of a block is a list of entries, the block metadata, the
/// size of the block metadata, and the checksum of the block. Generally, the
/// file layout looks like this:
///
/// ```markdown
/// Entry1 | Entry2 | ... | EntryN | Block Offsets | 4-byte Offset Size/Count | 8-byte Checksum
/// ```
///
/// The block metadata contains the list of offsets in the block used for binary
/// searching.
pub mod builder;
pub mod table;

/// Options for opening or building a [`Table`].
pub struct Options {
    /// Open tables in read only mode.
    pub read_only: bool,

    /// Maximum size of the table.
    pub table_size: usize,

    /// It's recommended for this to be 0.9x the table size.
    pub table_capacity: usize,

    /// The checksum verification mode for Table.
    pub chk_mode: ChecksumVerificationMode,

    /// The false positive probability parameter for fast key checking
    pub cache_probability: f64,

    /// The size of each block inside SSTable in bytes.
    pub block_size: usize,

    /// The key used to decrypt the encrypted text.
    pub data_key: Option<DataKey>,

    /// Indicates the compression algorithm used for block compression.
    pub compression: CompressionType,

    /// Block cache is used to cache decompressed and decrypted blocks.
    pub block_cache: Cache<Vec<u8>, Vec<u8>>,

    /// Index cache is the internal index cache
    pub index_cache: Cache<Vec<u8>, Vec<u8>>,

    /// The zstd compression level used for compressing blocks.
    pub zstd_compression_level: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            read_only: false,
            table_size: 0,
            table_capacity: 0,
            chk_mode: 0,
            cache_probability: 0.0,
            block_size: 0,
            data_key: None,
            compression: 0,
            block_cache: Cache::builder().build(),
            index_cache: Cache::builder().build(),
            zstd_compression_level: 0,
        }
    }
}
