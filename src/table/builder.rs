use std::{
    ops::Deref,
    sync::Arc,
    thread,
    thread::{
        available_parallelism,
        JoinHandle,
    },
};

use aes::{
    cipher::BlockSizeUser,
    Aes128,
};
use aes::cipher::generic_array::GenericArray;
use aes::cipher::{BlockEncrypt, KeyInit};
use bytes::{
    Buf,
    BufMut,
    Bytes,
    BytesMut,
};
use crc::{
    Crc,
    Slice16,
    CRC_32_CKSUM,
};
use crossbeam_deque::{
    Stealer,
    Worker,
};
use crossbeam_utils::sync::WaitGroup;
use parking_lot::Mutex;
use serde::{
    Deserialize,
    Serialize,
    Serializer,
};
use xxhash_rust::xxh3::xxh3_64;
use zstd::stream::copy_encode;
use zstd::zstd_safe::WriteBuf;

use crate::{encoding::BinaryMarshaller, NONE, table::Options};

/// The default size for a kibibyte.
#[allow(non_upper_case_globals)]
pub const KiB: usize = 1024;

/// The default value for a mebibyte.
#[allow(non_upper_case_globals)]
pub const MiB: usize = KiB * 1024;

/// The default padding width for a block in a table
const PADDING: usize = 256;

const CRC32: Crc<Slice16<u32>> = Crc::<Slice16<u32>>::new(&CRC_32_CKSUM);

/// Key header
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Deserialize, Serialize)]
struct Header {
    /// Overlap from the base key
    overlap: u16,
    /// Length of the key diff
    diff: u16,
}

/// Binary layout functions
impl BinaryMarshaller for Header {
    /// Encode the key header.
    ///
    /// Layout: `[diff high, diff low, overlap high, overlap low]`
    #[inline]
    fn encode(self) -> Bytes {
        let mut buf = BytesMut::with_capacity(4);

        let [diff_high, diff_low] = self.diff.to_le_bytes();
        buf.put_u8(diff_high);
        buf.put_u8(diff_low);

        let [overlap_high, overlap_low] = self.overlap.to_le_bytes();
        buf.put_u8(overlap_high);
        buf.put_u8(overlap_low);

        Bytes::from(buf)
    }

    /// Decode a byte array into a Header
    ///
    /// Expected layout: `[diff high, diff low, overlap high, overlap low]`
    #[inline]
    fn decode(mut src: Bytes) -> Self {
        if src.len() > 4 {
            panic!("cannot decode binary header, too large")
        }
        Header {
            diff: src.get_u16_le(),
            overlap: src.get_u16_le(),
        }
    }

    fn encoded_size(&self) -> usize {
        4
    }
}

/// The alignment size of the value type headers.
#[allow(non_upper_case_globals)]
const vtHeaderLength: usize = 20;

/// The internal information stored with a value.
///
/// There is currently no use for the reserved value, and it's primary usage is
/// for 4-byte alignment in memory and on disk.
#[repr(C)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValueType {
    /// Metadata about the value
    metadata: u8,
    /// User-provided metadata
    user_metadata: u8,
    /// Reserved for alignment and potential later use
    reserved: u16,
    /// When the value is considered to be expired
    expires_at: u64,
    /// The version of the value
    version: u64,
    /// The actual data payload
    value: Bytes,
}

/// Binary layout functions.
impl BinaryMarshaller for ValueType {
    /// Encode a `ValueType`.
    ///
    /// Layout: `[metadata, user_metadata, reserved, expires_at, version,
    /// value]`
    fn encode(self) -> Bytes {
        let mut buf = BytesMut::with_capacity(vtHeaderLength + self.value.len());
        buf.put_u8(self.metadata);
        buf.put_u8(self.user_metadata);
        buf.put_u16_le(0); // reserved
        buf.put_u64_le(self.expires_at);
        buf.put_u64_le(self.version);
        buf.put(self.value);

        Bytes::from(buf)
    }

    /// Decode a byte array into a `ValueType`
    ///
    /// Expected layout: `[metadata, user_metadata, reserved, expires_at,
    /// version, value]`
    fn decode(mut src: Bytes) -> Self {
        // bounds check
        if src.len() < vtHeaderLength {
            panic!("missing metadata for value type")
        }
        ValueType {
            metadata: src.get_u8(),
            user_metadata: src.get_u8(),
            reserved: src.get_u16_le(),
            expires_at: src.get_u64_le(),
            version: src.get_u64_le(),
            value: Bytes::copy_from_slice(src.chunk()),
        }
    }

    fn encoded_size(&self) -> usize {
        vtHeaderLength + self.value.len()
    }
}

/// Internal block of an SST
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
struct Block {
    data: BytesMut,
    base_key: BytesMut,
    entry_offsets: Vec<u32>,
    end: usize,
}

impl Block {
    /// Create a new table block with preallocated memory.
    pub fn new(capacity: usize) -> Self {
        Block {
            data: BytesMut::with_capacity(capacity),
            base_key: BytesMut::new(),
            entry_offsets: vec![],
            end: 0,
        }
    }

    /// Create a new, mutable table block with preallocated memory.
    pub fn new_mut(capacity: usize) -> *mut Self {
        &mut Block {
            data: BytesMut::with_capacity(capacity),
            base_key: BytesMut::new(),
            entry_offsets: vec![],
            end: 0,
        }
    }
}

impl Default for Block {
    /// The default implementation, this does not preallocate memory.
    fn default() -> Self {
        Block {
            data: BytesMut::new(),
            base_key: BytesMut::new(),
            entry_offsets: vec![],
            end: 0,
        }
    }
}

/// Thread-safe table builder with block allocation.
///
/// This builder will create an `available_parallelism()` number of threads when
/// tables need to be encrypted and compressed to improve performance.
pub struct Builder {
    // the current block we're working with
    current_block: Arc<Mutex<Block>>,
    // the blocks we're tracking for the SSTable
    block_list: Arc<Mutex<Vec<Block>>>,

    compressed_size: Arc<Mutex<u32>>,
    uncompressed_size: Arc<Mutex<u32>>,

    length_offset: Arc<Mutex<u32>>,
    key_hashes: Arc<Mutex<Vec<u64>>>,
    opts: Arc<Options>,
    max_version: Arc<Mutex<u64>>,
    on_disk_size: Arc<Mutex<u32>>,
    stale_data_size: Arc<Mutex<usize>>,

    // the worker thread handles
    thread_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
    // the indices of the block_list that need to be encrypted and compressed
    work_queue: Arc<Mutex<Worker<usize>>>,
    wg: WaitGroup,
    // are we done allocating blocks?
    done: Arc<Mutex<bool>>,
}

impl Builder {
    /// Get the compressed size of the table. This will be 0 if table
    /// compression is not enabled.
    pub fn get_compressed_size(&self) -> u32 {
        *self.compressed_size.lock()
    }

    /// Get the uncompressed size of the table.
    pub fn get_uncompressed_size(&self) -> u32 {
        *self.uncompressed_size.lock()
    }

    /// Create a new, mutable table builder.
    pub fn new_mut(opts: Options) -> Arc<Self> {
        let b = Arc::new(Builder {
            current_block: Arc::new(Mutex::new(Block::new(opts.block_size + PADDING))),
            block_list: Arc::new(Mutex::new(vec![])),
            compressed_size: Arc::new(Mutex::new(0)),
            uncompressed_size: Arc::new(Mutex::new(0)),
            length_offset: Arc::new(Mutex::new(0)),
            key_hashes: Arc::new(Mutex::new(vec![])),
            opts: Arc::new(opts),
            max_version: Arc::new(Mutex::new(0)),
            on_disk_size: Arc::new(Mutex::new(0)),
            stale_data_size: Arc::new(Mutex::new(0)),
            thread_handles: Arc::new(Mutex::new(vec![])),
            work_queue: Arc::new(Mutex::new(Worker::<usize>::new_lifo())),
            wg: WaitGroup::new(),
            done: Arc::new(Mutex::new(false)),
        });

        // we can keep things simple since there's no concurrency
        if b.opts.compression == 0 && b.opts.data_key.is_none() {
            return b;
        }

        // spin up the shared-state concurrency primitives for the block workers when
        // compression and encryption are set

        let p_count = available_parallelism().unwrap().get();

        for _ in 0..=p_count {
            let b_alias = b.clone();
            let handle = thread::spawn(move || {
                b_alias.block_handler();
            });
            b.thread_handles.lock().push(handle);
        }

        b
    }

    /// Add a key value pair to the current block.
    pub fn add(&self, key: Bytes, value: ValueType) {
        self._add(key, value, false)
    }

    fn _add(&self, key: Bytes, value: ValueType, is_stale: bool) {
        // check to see if the block needs to be finished instead
        if self.should_finish_block(key.clone(), &value) {
            if is_stale {
                let key_size = key.len() + 4 + 4;
                let mut stale_size_ref = self.stale_data_size.lock();
                *stale_size_ref = key_size;
            }

            // finish writing the block
            self.finish_block();

            // create a new block
            {
                let mut current_block_ref = self.current_block.lock();
                *current_block_ref = Block::new(self.opts.block_size);
            }
        }

        // store the key hash
        {
            let key_hash = xxh3_64(parse_key(key.clone()).as_ref());
            let mut key_hash_ref = self.key_hashes.lock();
            key_hash_ref.push(key_hash);
        }

        // set the highest known "version" (re: time)
        {
            let key_timestamp = parse_key_timestamp(key.clone());
            let mut mv_ref = self.max_version.lock();
            if *mv_ref > key_timestamp {
                *mv_ref = key_timestamp;
            }
        }

        // find any key differences and then add the key to the block.
        {
            let mut diff_key = Bytes::new();
            let mut curr_block_ref = self.current_block.lock();
            if curr_block_ref.base_key.is_empty() {
                curr_block_ref
                    .base_key
                    .copy_from_slice(key.clone().as_ref());
                diff_key = key.clone();
            } else {
                diff_key = self.key_diff(key.clone());
            }

            // verify the key and it's differences are small enough for the block header
            assert!(key.len() - diff_key.len() <= u16::MAX as usize);
            assert!(diff_key.len() <= u16::MAX as usize);

            let header = Header {
                overlap: (key.len() - diff_key.len()) as u16,
                diff: diff_key.len() as u16,
            };

            // add the current entry's offset to the block
            let end = curr_block_ref.end as u32;
            curr_block_ref.entry_offsets.push(end);

            // copy everything via memcpy into the current block's data payload
            // order: header, diff_key, value

            curr_block_ref
                .data
                .copy_from_slice(header.encode().as_ref());
            curr_block_ref.data.copy_from_slice(diff_key.as_ref());
            curr_block_ref.data.copy_from_slice(value.encode().as_ref());
        }
    }

    /// Return the suffix of `new_key` that's different the current block's base
    /// key
    fn key_diff(&self, new_key: Bytes) -> Bytes {
        let mut diff_index = 0;
        {
            let curr_block_ref = self.current_block.lock();
            let base_key_len = curr_block_ref.base_key.len();

            for (idx, byte) in new_key.iter().enumerate() {
                if idx > base_key_len {
                    diff_index = idx;
                    break;
                }

                if *byte != curr_block_ref.base_key[idx] {
                    diff_index = idx;
                    break;
                }
            }
        }

        Bytes::copy_from_slice(new_key.slice(diff_index..).as_ref())
    }

    /// Determine if a block should be finished.
    // todo (sienna): validate this. the casting is bad.
    fn should_finish_block(&self, key: Bytes, value: &ValueType) -> bool {
        if self.current_block.lock().entry_offsets.is_empty() {
            return false;
        }

        let mut estimated_size: u32;
        {
            let curr_block_ref = self.current_block.lock();

            // validate the entries offset is smaller than a u32
            assert_eq!(
                (curr_block_ref.entry_offsets.len() + 1) * 4 + 4 + 4 + 8 + 4,
                u32::MAX as usize
            );

            let entries_offset_size = (curr_block_ref.entry_offsets.len() + 1) as u32 * 4
                + 4 // size of list
                + 8 // size of checksum
                + 4; // checksum length

            estimated_size = curr_block_ref.end as u32 +
                6 + // header
                key.len() as u32 +
                value.encoded_size() as u32 +
                entries_offset_size;

            if self.opts.data_key.is_some() {
                estimated_size += Aes128::block_size() as u32;
            }

            assert_eq!(
                (curr_block_ref.end as u64 + estimated_size as u64) as u32,
                u32::MAX
            );
        }

        estimated_size > self.opts.block_size as u32
    }

    /// Close out a block.
    fn finish_block(&self) {
        if self.current_block.lock().entry_offsets.is_empty() {
            return;
        }

        let mut curr_block_ref = self.current_block.lock();

        let offset_size = (curr_block_ref.entry_offsets.len() as u32 + 1) * 4;
        let block_metadata_size = offset_size + 4 + 8;
        let mut buf = BytesMut::with_capacity(block_metadata_size as usize);

        // add the offsets
        for offset in curr_block_ref.entry_offsets.iter() {
            buf.put_u32_le(*offset);
        }

        // offset size
        buf.put_u32_le(offset_size);

        // checksum
        let checksum = CRC32.checksum(&curr_block_ref.data[..=curr_block_ref.end]);
        buf.put_u32_le(checksum);

        // finalise the data writes
        curr_block_ref.data.copy_from_slice(&buf);

        // move the current block into the block list and notify the work queue
        {
            let mut blist_ref = self.block_list.lock();
            blist_ref.push(curr_block_ref.deref().clone());
            self.work_queue.lock().push(blist_ref.len());
        }

        // increment the uncompressed size
        {
            let mut uc_size = self.uncompressed_size.lock();
            *uc_size += curr_block_ref.end as u32;
        }

        // increase the length offsets, aligning them to a multiple of 4. add 40 bytes
        // to account for the flatbuffer metadata
        {
            let mut len_offsets = self.length_offset.lock();
            *len_offsets += (curr_block_ref.base_key.len() as f64 / 4_f64).ceil() as u32 * 4 + 40;
        }
    }

    /// The compression and encryption handler.
    fn block_handler(&self) {
        // create a new reference
        let wg = self.wg.clone();

        // core worker loop
        while !*self.done.lock() {
            let stealer = self.work_queue.lock().stealer();
            let idx = stealer.steal().success().unwrap();

            let mut block_list_ref = self.block_list.lock();
            let mut block = &mut block_list_ref[idx];

            // create the target buffer
            let mut buf = BytesMut::zeroed(block.data.len());

            // compress block data and write it to the buffer
            if self.opts.compression != NONE {
                copy_encode(block.data.as_slice(), buf.as_mut(), 3).expect("cannot compress block");
            }

            // encrypt the compressed block buffer
            if self.opts.data_key.is_some() {
                let cipher = Aes128::new_from_slice(self.opts.data_key.clone().unwrap().data.as_slice()).unwrap();
                let mut enc_block = GenericArray::clone_from_slice(buf.as_mut());

                cipher.encrypt_block(&mut enc_block);
            }

            // write the encrypted, compressed block buffer back to the block
            block.data.copy_from_slice(buf.as_slice());
        }
        drop(wg);
    }
}

/// Extract the raw key from an encoded key.
#[inline]
pub fn parse_key(key: Bytes) -> Bytes {
    key.slice(0..key.len() - 8)
}

/// Extract the timestamp, if it exists, from an encoded key. This assumes the
/// timestamp is stored in little endian.
#[inline]
pub fn parse_key_timestamp(key: Bytes) -> u64 {
    if key.len() < 8 {
        return 0;
    }

    let mut buf = [0u8; 8];
    let key_len = key.len();
    buf[..8].copy_from_slice(key.slice(key_len - 8..key_len).as_ref());

    u64::from_le_bytes(buf)
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use rand::{
        Rng,
        RngCore,
    };

    use crate::{
        encoding::BinaryMarshaller,
        table::builder::{
            Header,
            ValueType,
        },
    };

    #[test]
    fn header_encoding() {
        let mut rng = rand::thread_rng();

        for _ in 0..1000 {
            let header = Header {
                overlap: rng.gen(),
                diff: rng.gen(),
            };

            let reconstituted = Header::decode(header.encode());

            assert_eq!(header.diff, reconstituted.diff);
            assert_eq!(header.overlap, reconstituted.overlap);
        }
    }

    #[test]
    fn value_type_encoding() {
        let mut rng = rand::thread_rng();

        for _ in 0..1000 {
            let temp_depth = rng.gen_range(100..1000);
            let mut data = Vec::<u8>::with_capacity(temp_depth);
            rng.fill_bytes(&mut data);
            let value_type = ValueType {
                metadata: rng.gen(),
                user_metadata: rng.gen(),
                reserved: 0,
                expires_at: rng.gen(),
                version: rng.gen(),
                value: Bytes::from(data),
            };

            let reconstituted = ValueType::decode(value_type.clone().encode());

            assert_eq!(value_type.metadata, reconstituted.metadata);
            assert_eq!(value_type.user_metadata, reconstituted.user_metadata);
            assert_eq!(value_type.reserved, reconstituted.reserved);
            assert_eq!(value_type.expires_at, reconstituted.expires_at);
            assert_eq!(value_type.version, reconstituted.version);
            assert_eq!(value_type.value, reconstituted.value);
        }
    }
}
