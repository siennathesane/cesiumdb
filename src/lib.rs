/// The core CesiumDB interface
pub mod db;

/// A wonderful [`skip list`] implementation courtesy of [`JP-Ellis`]
///
/// [`skip list`]: https://en.wikipedia.org/wiki/Skip_list
/// [`JP-Ellis`]: https://github.com/JP-Ellis/rust-skiplist/
pub mod skiplist;
pub mod table;

mod encoding;
mod types;

/// [`ChecksumVerificationMode`] tells when should DB verify checksum for
/// SSTable blocks.
type ChecksumVerificationMode = u8;

/// [`NO_VERIFICATION`] indicates DB should not verify checksum for SSTable
/// blocks.
pub const NO_VERIFICATION: ChecksumVerificationMode = 0;
/// [`ON_TABLE_READ`] indicates checksum should be verified while opening
/// SSTable.
pub const ON_TABLE_READ: ChecksumVerificationMode = 1;
/// [`ON_BLOCK_READ`] indicates checksum should be verified on every SSTable
/// block read.
pub const ON_BLOCK_READ: ChecksumVerificationMode = 2;
/// [`ON_TABLE_AND_BLOCK_READ`] indicates checksum should be verified on SSTable
/// opening and on every block read.
pub const ON_TABLE_AND_BLOCK_READ: ChecksumVerificationMode = 3;

/// [`CompressionType`] specifies how a block should be compressed.
type CompressionType = u8;

/// [`NONE`] mode indicates that a block is not compressed.
pub const NONE: CompressionType = 0;

/// [`ZSTD`] mode indicates that a block is compressed using ZSTD algorithm.
pub const ZSTD: CompressionType = 1;
