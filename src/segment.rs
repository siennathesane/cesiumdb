use std::{
    fmt::Display,
    mem,
    ptr,
    sync::{
        Arc,
        atomic::{
            AtomicBool,
            AtomicU64,
            Ordering::Relaxed,
        },
    },
};

use bytes::{
    BufMut,
    Bytes,
    BytesMut,
};
use parking_lot::Mutex;

use crate::{
    block::{
        Block,
        EntryFlag,
        EntryFlag::{
            Complete,
            End,
            Middle,
            Start,
        },
        MAX_ENTRY_SIZE,
    },
    errs::{
        SegmentError,
        SegmentError::{
            CantCreateReader,
            ReadOnly,
        },
    },
    index::Index,
    keypair::DEFAULT_NS,
    map::Map,
    segment::BlockType::{
        Key,
        Value,
    },
    segment_reader::{
        ReadConfig,
        SegmentReader,
    },
    segment_writer::SegmentWriter,
};

// Constants for value location metadata embedded in keys
/// Size of value location metadata: u64 (block_num) + u16 (entry_index) = 10
/// bytes
pub(crate) const VALUE_LOCATION_SIZE: usize = size_of::<u64>() + size_of::<u16>();
/// Offset where value block number is stored in key metadata
const VALUE_BLOCK_OFFSET: usize = 0;
/// Offset where value entry index is stored in key metadata
const VALUE_ENTRY_OFFSET: usize = size_of::<u64>();
/// Offset where actual key data starts (after metadata)
pub(crate) const KEY_DATA_OFFSET: usize = VALUE_LOCATION_SIZE;

// Segment file size constants
/// Default segment size: 64 MiB
pub(crate) const DEFAULT_SEGMENT_SIZE: u64 = 64 * 1024 * 1024;
/// Threshold for detecting pre-allocated empty segments
const PREALLOCATED_FILE_THRESHOLD: u64 = DEFAULT_SEGMENT_SIZE;

// Metadata size constant
/// Size of segment metadata: 4 * u64 = 32 bytes
const METADATA_SIZE: usize = 4 * size_of::<u64>();

#[derive(Debug)]
pub enum BlockType {
    Key,
    Value,
}

impl Display for BlockType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            | Key => write!(f, "key"),
            | Value => write!(f, "value"),
        }
    }
}

#[derive(Debug)]
pub(crate) struct Metadata {
    id: u64,
    block_count: u64,
    index_size: u64,
    index_start: u64,
}

impl Metadata {
    pub(crate) fn new(id: u64, block_count: u64, index_size: u64, index_start: u64) -> Self {
        Self {
            id,
            block_count,
            index_size,
            index_start,
        }
    }

    pub(crate) fn serialized_size(&self) -> usize {
        // 4 fields, each is a u64 (8 bytes)
        4 * size_of::<u64>()
    }

    /// Finalizes the Metadata by writing it directly to a memory location.
    ///
    /// # Safety
    ///
    /// - `dst` must be valid for at least `self.serialized_size()` bytes (32
    ///   bytes)
    /// - `dst` must be properly aligned for u64 values (8-byte alignment)
    /// - `dst` must not overlap with any source data
    /// - Caller must ensure exclusive access to the dst memory region
    pub(crate) unsafe fn finalize(&self, dst: *mut u8) {
        // SAFETY: Verify alignment invariants in debug builds
        debug_assert!(!dst.is_null(), "Destination pointer must not be null");
        debug_assert!(
            (dst as usize).is_multiple_of(std::mem::align_of::<u64>()),
            "Destination pointer must be 8-byte aligned for u64 writes"
        );

        let mut offset = 0;

        // SAFETY: All writes stay within the allocated buffer size (32 bytes).
        // Each write advances the offset to ensure non-overlapping writes.
        unsafe {
            // Write id
            ptr::copy_nonoverlapping(
                self.id.to_le_bytes().as_ptr(),
                dst.add(offset),
                size_of::<u64>(),
            );
            offset += size_of::<u64>();

            // Write block_count
            ptr::copy_nonoverlapping(
                self.block_count.to_le_bytes().as_ptr(),
                dst.add(offset),
                size_of::<u64>(),
            );
            offset += size_of::<u64>();

            // Write index_size
            ptr::copy_nonoverlapping(
                self.index_size.to_le_bytes().as_ptr(),
                dst.add(offset),
                size_of::<u64>(),
            );
            offset += size_of::<u64>();

            // Write index_start
            ptr::copy_nonoverlapping(
                self.index_start.to_le_bytes().as_ptr(),
                dst.add(offset),
                size_of::<u64>(),
            );
        }
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn block_count(&self) -> usize {
        self.block_count as usize
    }

    pub(crate) fn index_size(&self) -> usize {
        self.index_size as usize
    }

    pub(crate) fn index_start(&self) -> usize {
        self.index_start as usize
    }
}

impl From<Metadata> for Bytes {
    fn from(metadata: Metadata) -> Bytes {
        let size = metadata.serialized_size();
        let mut buffer = BytesMut::with_capacity(size);
        buffer.resize(size, 0);

        // SAFETY: we just allocated enough space
        unsafe {
            metadata.finalize(buffer.as_mut_ptr());
        }

        buffer.freeze()
    }
}

impl From<Bytes> for Metadata {
    fn from(bytes: Bytes) -> Self {
        assert!(bytes.len() >= 32, "Metadata requires at least 32 bytes");

        let id = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let block_count = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let index_size = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let index_start = u64::from_le_bytes(bytes[24..32].try_into().unwrap());

        Self {
            id,
            block_count,
            index_size,
            index_start,
        }
    }
}

pub struct Segment {
    // writers are used to write data whereas the map is used to read data

    // keys
    key_writer: Mutex<Option<SegmentWriter>>,
    key_handle: Option<Arc<Map>>,
    key_index: Arc<Mutex<Index>>,
    current_key_block: Mutex<Block>,
    key_id: u64,

    // values
    val_writer: Mutex<Option<SegmentWriter>>,
    val_handle: Option<Arc<Map>>,
    current_val_block: Mutex<Block>,
    val_block_count: AtomicU64,
    val_id: u64,

    // shared
    current_ns: AtomicU64,
}

impl Segment {
    /// Create a new segment. Once the segment has been closed, it cannot be
    /// opened again for writes.
    pub fn new(
        key_id: u64,
        val_id: u64,
        seed: i64,
        key_writer: SegmentWriter,
        val_writer: SegmentWriter,
    ) -> Self {
        let mut key_index = Index::new(key_id, seed);

        // explicitly record the default namespace in the index
        // during init to ensure the namespace is always recorded.
        // this is because the namespace is only recorded when it
        // changes, but if we start with the default namespace, it
        // will never be seen to change. this creates an extra byte
        // of space in the index, but there isn't a way to say "this
        // is the first namespace" without doing this. so even if the
        // first seen namespace isn't the default one, it will still
        // be a pointer to the same record.
        key_index.insert_ns_offset(DEFAULT_NS);

        Self {
            key_writer: Mutex::new(Some(key_writer)),
            key_handle: None,
            val_writer: Mutex::new(Some(val_writer)),
            val_handle: None,
            key_index: Arc::new(Mutex::new(key_index)),
            current_key_block: Mutex::new(Block::new()),
            current_val_block: Mutex::new(Block::new()),
            val_block_count: AtomicU64::new(0),
            current_ns: AtomicU64::new(DEFAULT_NS),
            key_id,
            val_id,
        }
    }

    /// Open a pre-written segment for reading. A pre-written segment cannot be
    /// written to.
    pub fn open(
        key_map: Arc<Map>,
        key_index: Index,
        key_id: u64,
        val_map: Arc<Map>,
        val_id: u64,
    ) -> Result<Arc<Segment>, SegmentError> {
        Ok(Arc::new(Segment {
            key_writer: Mutex::new(None),
            key_handle: Some(key_map),
            val_writer: Mutex::new(None),
            val_handle: Some(val_map),
            key_index: Arc::new(Mutex::new(key_index)),
            current_key_block: Mutex::new(Block::new()),
            current_val_block: Mutex::new(Block::new()),
            val_block_count: AtomicU64::new(0),
            current_ns: AtomicU64::new(DEFAULT_NS),
            key_id,
            val_id,
        }))
    }

    /// Returns true if this segment is read-only (opened from disk).
    pub fn is_read_only(&self) -> bool {
        self.key_writer.lock().is_none()
    }

    pub fn write(&self, key: &[u8], val: &[u8]) -> Result<(), SegmentError> {
        use crate::errs::BlockError;

        if self.key_writer.lock().is_none() {
            return Err(ReadOnly);
        }

        // set the namespace
        let ns = u64::from_le_bytes(key[0..8].as_ref().try_into().unwrap());
        if ns != self.current_ns.load(Relaxed) {
            self.current_ns.store(ns, Relaxed);
            self.key_index.lock().insert_ns_offset(ns);
        }

        // Write value to value block FIRST so we know where it lands
        let (value_block_num, value_entry_index) = match self.add_entry_with_retry(val, &Value) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };

        // Now write key with embedded value location metadata
        // Format: [value_block_num:u64][value_entry_index:u16][key_data]
        let mut key_with_metadata = BytesMut::with_capacity(VALUE_LOCATION_SIZE + key.len());
        key_with_metadata.put_u64_le(value_block_num);
        key_with_metadata.put_u16_le(value_entry_index);
        key_with_metadata.put_slice(key);

        // Write key with embedded value location metadata
        // Note: We ignore the return value since we already have the value location
        match self.add_entry_with_retry(&key_with_metadata, &Key) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };

        self.key_index.lock().insert_item(key);

        Ok(())
    }

    pub fn new_reader(&self) -> Result<SegmentReader, SegmentError> {
        let km: Arc<Map> = match &self.key_handle {
            | Some(handle) => handle.clone(),
            | None => {
                let writer = self.key_writer.lock();
                match writer.as_ref() {
                    | Some(w) => w.map.clone(),
                    | None => return Err(CantCreateReader),
                }
            },
        };

        let vm: Arc<Map> = match &self.val_handle {
            | Some(handle) => handle.clone(),
            | None => {
                let writer = self.val_writer.lock();
                match writer.as_ref() {
                    | Some(w) => w.map.clone(),
                    | None => return Err(CantCreateReader),
                }
            },
        };

        match SegmentReader::new(km, vm, self.key_index.clone()) {
            | Ok(v) => Ok(v),
            | Err(e) => Err(e),
        }
    }

    /// Flush any pending blocks
    /// Helper method to add an entry to a block with retry logic.
    /// Returns (block_num, entry_index) for where the entry was placed.
    ///
    /// This handles the common pattern of:
    /// 1. Try adding to current block
    /// 2. If too large -> split across blocks
    /// 3. If block full -> flush block and retry
    fn add_entry_with_retry(
        &self,
        data: &[u8],
        block_type: &BlockType,
    ) -> Result<(u64, u16), SegmentError> {
        use crate::errs::BlockError;

        let (block_mutex, block_counter) = match block_type {
            | Key => (&self.current_key_block, None),
            | Value => (&self.current_val_block, Some(&self.val_block_count)),
        };

        let mut block = block_mutex.lock();

        match block.add_entry(data, Complete) {
            | Ok(()) => {
                // Entry added successfully to current block
                let block_num = match block_counter {
                    | Some(counter) => counter.load(Relaxed),
                    | None => 0, // For key blocks, we don't use block_num in the same way
                };
                let entry_idx = (block.num_entries() - 1);
                drop(block);
                Ok((block_num, entry_idx))
            },
            | Err(be) => {
                drop(block);

                match be {
                    | BlockError::TooLargeForBlock => {
                        // Entry is too large for a single block, split it
                        match self.split_across_blocks(data, block_type) {
                            | Ok(v) => Ok(v),
                            | Err(e) => Err(e),
                        }
                    },
                    | BlockError::CorruptedBlock => {
                        unreachable!("unexpected corrupted block error during write")
                    },
                    | BlockError::BlockFull => {
                        // Flush current block and retry
                        match self.write_block(block_type) {
                            | Ok(v) => v,
                            | Err(e) => return Err(e),
                        };

                        let mut block = block_mutex.lock();
                        match block.add_entry(data, Complete) {
                            | Ok(_) => {
                                // Entry added successfully to new block
                                let block_num = match block_counter {
                                    | Some(counter) => counter.load(Relaxed),
                                    | None => 0,
                                };
                                let entry_idx = (block.num_entries() - 1);
                                Ok((block_num, entry_idx))
                            },
                            | Err(rbe) => match rbe {
                                | BlockError::TooLargeForBlock => {
                                    drop(block);
                                    match self.split_across_blocks(data, block_type) {
                                        | Ok(v) => Ok(v),
                                        | Err(e) => Err(e),
                                    }
                                },
                                | BlockError::CorruptedBlock | BlockError::BlockFull => {
                                    unreachable!("unexpected block error after flush")
                                },
                            },
                        }
                    },
                }
            },
        }
    }

    /// Helper to write a chunk to a fresh block with the given flag.
    /// This is used during multi-block entry splitting.
    fn write_chunk_to_new_block(
        &self,
        chunk: &[u8],
        flag: EntryFlag,
        block_type: &BlockType,
    ) -> Result<(), SegmentError> {
        let mut block = Block::new();
        match block.add_entry(chunk, flag).map_err(|e| {
            eprintln!(
                "Error adding {:?} chunk to {} block: {:?}, chunk size: {}",
                flag,
                block_type,
                e,
                chunk.len()
            );
            SegmentError::InsufficientSpace
        }) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };

        match block_type {
            | Key => {
                let mut writer_guard = self.key_writer.lock();
                let writer = match writer_guard.as_mut() {
                    | Some(v) => v,
                    | None => return Err(ReadOnly),
                };
                match writer.write_block(block) {
                    | Ok(v) => v,
                    | Err(e) => return Err(e),
                };
                drop(writer_guard);
                self.key_index.lock().inc_block_count(1);
            },
            | Value => {
                let mut writer_guard = self.val_writer.lock();
                let writer = match writer_guard.as_mut() {
                    | Some(v) => v,
                    | None => return Err(ReadOnly),
                };
                match writer.write_block(block) {
                    | Ok(v) => v,
                    | Err(e) => return Err(e),
                };
                drop(writer_guard);
                self.val_block_count.fetch_add(1, Relaxed);
            },
        }

        Ok(())
    }

    pub fn flush(&self) -> Result<(), SegmentError> {
        if self.key_writer.lock().is_none() {
            return Err(ReadOnly);
        }

        // Flush key block if it has entries
        {
            let key_block = self.current_key_block.lock();
            if !key_block.is_empty() {
                drop(key_block);
                match self.write_block(&Key) {
                    | Ok(v) => v,
                    | Err(e) => return Err(e),
                };
            }
        }

        // Flush value block if it has entries
        {
            let val_block = self.current_val_block.lock();
            if !val_block.is_empty() {
                drop(val_block);
                match self.write_block(&Value) {
                    | Ok(v) => v,
                    | Err(e) => return Err(e),
                };
            }
        }

        Ok(())
    }

    /// Split a payload across multiple blocks.
    /// Returns (block_num, entry_index) where the START entry was placed.
    fn split_across_blocks(
        &self,
        data: &[u8],
        r#type: &BlockType,
    ) -> Result<(u64, u16), SegmentError> {
        if data.is_empty() {
            return Err(SegmentError::InsufficientSpace);
        }

        let mut remaining = data;
        let max_chunk_size = MAX_ENTRY_SIZE - 1;
        let start_entry_index: u16 = 0; // Always at index 0 in fresh block

        // Ensure current block is flushed before starting
        {
            let block = match r#type {
                | Key => self.current_key_block.lock(),
                | Value => self.current_val_block.lock(),
            };
            if !block.is_empty() {
                drop(block);
                match self.write_block(r#type) {
                    | Ok(v) => v,
                    | Err(e) => return Err(e),
                };
            }
        }

        // Capture the block number where START will be placed
        let start_block_num = match r#type {
            | Key => self.key_index.lock().block_count(),
            | Value => self.val_block_count.load(Relaxed),
        };

        // Write START chunk
        let chunk_size = std::cmp::min(max_chunk_size, remaining.len());
        let chunk = &remaining[..chunk_size];

        // For keys, also insert into index
        if matches!(r#type, Key) {
            self.key_index.lock().insert_item(data);
        }

        match self.write_chunk_to_new_block(chunk, Start, r#type) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };
        remaining = &remaining[chunk_size..];

        // Process middle chunks (MIDDLE flag)
        while remaining.len() > max_chunk_size {
            let chunk = &remaining[..max_chunk_size];
            match self.write_chunk_to_new_block(chunk, Middle, r#type) {
                | Ok(v) => v,
                | Err(e) => return Err(e),
            };
            remaining = &remaining[max_chunk_size..];
        }

        // Process the last chunk (END flag) if there's anything left
        if !remaining.is_empty() {
            match self.write_chunk_to_new_block(remaining, End, r#type) {
                | Ok(v) => v,
                | Err(e) => return Err(e),
            };
        }

        Ok((start_block_num, start_entry_index))
    }

    fn write_block(&self, r#type: &BlockType) -> Result<(), SegmentError> {
        match r#type {
            | Key => {
                // Check if block is empty
                {
                    let key_block = self.current_key_block.lock();
                    if key_block.is_empty() {
                        return Ok(()); // Nothing to write
                    }
                }

                // Extract the first key if it exists
                let starting_key_data = {
                    let key_block = self.current_key_block.lock();
                    match key_block.get(0) {
                        | Some((_flag, key_data)) => Some(key_data.to_vec()),
                        | None => None,
                    }
                };

                // Swap the block
                let block = {
                    let mut key_block = self.current_key_block.lock();
                    mem::replace(&mut *key_block, Block::new())
                };

                // Add to index BEFORE incrementing block count (for 0-based indexing)
                if let Some(key_data) = starting_key_data {
                    self.key_index.lock().insert_item(&key_data);
                }

                // Write block

                {
                    let mut writer_guard = self.key_writer.lock();
                    match writer_guard.as_mut() {
                        | Some(writer) => {
                            let res = writer.write_block(block);
                            // Sync index block count with writer after write
                            if res.is_ok() {
                                drop(writer_guard);
                                self.key_index.lock().inc_block_count(1);
                            }
                            res
                        },
                        | None => Err(ReadOnly),
                    }
                }
            },
            | Value => {
                // Check if block is empty
                {
                    let val_block = self.current_val_block.lock();
                    if val_block.is_empty() {
                        return Ok(()); // Nothing to write
                    }
                }

                // Swap the block
                let block = {
                    let mut val_block = self.current_val_block.lock();
                    mem::replace(&mut *val_block, Block::new())
                };

                // Write block
                // Note: We don't need a value index since the value location is stored
                // in the key metadata

                {
                    let mut writer_guard = self.val_writer.lock();
                    match writer_guard.as_mut() {
                        | Some(writer) => {
                            let res = writer.write_block(block);
                            // Increment block count after successful write
                            if res.is_ok() {
                                drop(writer_guard);
                                self.val_block_count.fetch_add(1, Relaxed);
                            }
                            res
                        },
                        | None => Err(ReadOnly),
                    }
                }
            },
        }
    }

    pub(crate) fn close(&self) -> Result<(), SegmentError> {
        if self.key_writer.lock().is_none() {
            return Err(ReadOnly);
        }

        match self.flush() {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        }

        {
            let mut writer_guard = self.key_writer.lock();
            if let Some(writer) = writer_guard.as_mut() {
                // Index block count already synced during writes
                let block_count = writer.block_count();

                let key_index = self.key_index.lock();
                let index_size = key_index.size();
                let index_start = match writer.write_index(&key_index) {
                    | Ok(v) => v,
                    | Err(e) => return Err(e),
                };
                drop(key_index);

                match writer.write_metadata(Metadata::new(
                    self.key_id,
                    block_count,
                    index_size as u64,
                    index_start,
                )) {
                    | Ok(v) => v,
                    | Err(e) => return Err(e),
                };
                match writer.close() {
                    | Ok(v) => v,
                    | Err(e) => return Err(e),
                };
            }
            // Set writer to None to mark segment as read-only
            *writer_guard = None;
        }

        {
            let mut writer_guard = self.val_writer.lock();
            if let Some(writer) = writer_guard.as_mut() {
                let block_count = writer.block_count();

                // Value segments don't need an index since value locations are stored in key
                // metadata Mark as closing before writing metadata
                writer.begin_close();

                // Write minimal metadata (no index)
                match writer.write_metadata(Metadata::new(
                    self.val_id,
                    block_count,
                    0, // no index
                    0, // no index
                )) {
                    | Ok(v) => v,
                    | Err(e) => return Err(e),
                };
                match writer.close() {
                    | Ok(v) => v,
                    | Err(e) => return Err(e),
                };
            }
            // Set writer to None to mark segment as read-only
            *writer_guard = None;
        }

        Ok(())
    }

    /// Returns the segment ID (key_id)
    #[inline]
    pub fn id(&self) -> u64 {
        self.key_id
    }

    /// Returns the approximate size of this segment in bytes
    ///
    /// This includes both key and value files plus index overhead.
    /// The actual size may be slightly different due to alignment.
    pub fn size_in_bytes(&self) -> u64 {
        let key_size = if let Some(ref handle) = self.key_handle {
            handle.len() as u64
        } else {
            // Estimate based on writer if still open
            0 // TODO: Track bytes written
        };

        let val_size = if let Some(ref handle) = self.val_handle {
            handle.len() as u64
        } else {
            // Estimate based on writer if still open
            0 // TODO: Track bytes written
        };

        key_size + val_size
    }

    /// Creates a SegmentReader for this segment
    ///
    /// This can only be called on read-only segments (opened from disk).
    pub fn reader(&self) -> Result<crate::segment_reader::SegmentReader, SegmentError> {
        if !self.is_read_only() {
            return Err(SegmentError::ReadOnly);
        }

        let key_handle = match self.key_handle.as_ref() {
            | Some(v) => v.clone(),
            | None => return Err(SegmentError::ReadOnly),
        };
        let val_handle = match self.val_handle.as_ref() {
            | Some(v) => v.clone(),
            | None => return Err(SegmentError::ReadOnly),
        };
        let key_index = self.key_index.clone();

        crate::segment_reader::SegmentReader::new(key_handle, val_handle, key_index)
    }
}

impl Drop for Segment {
    fn drop(&mut self) {
        let res = self.close();
        if let Err(e) = res &&
            !matches!(e, ReadOnly)
        {
            // Log the error instead of panicking during Drop to avoid double-panic
            // situations and unwinding issues. This error indicates that
            // segment metadata may not have been flushed properly, which could
            // lead to data loss on reopening.
            eprintln!(
                "CRITICAL: Failed to close segment {} cleanly during drop: {:?}",
                self.key_id, e
            );
            eprintln!("This may result in data loss or corruption when the segment is reopened.");
        }
    }
}

#[cfg(test)]
#[allow(clippy::question_mark_used)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::undocumented_unsafe_blocks)]
mod tests {
    use std::{
        collections::HashMap,
        sync::Arc,
    };

    use bytes::Bytes;
    use rand::{
        Rng,
        RngCore,
        prelude::SliceRandom,
        rng,
        thread_rng,
    };
    use tempfile::tempdir;

    use super::*;
    use crate::{
        block::Block,
        hlc::{
            HLC,
            HybridLogicalClock,
        },
        keypair::{
            DEFAULT_NS,
            KeyBytes,
            ValueBytes,
        },
        map::Map,
        memtable::Memtable,
        segment_reader::SegmentReader,
        segment_writer::SegmentWriter,
    };

    // helper function to create temporary segment components
    fn create_test_segment() -> (Arc<Segment>, tempfile::TempDir) {
        let dir = tempdir().expect("failed to create temp dir");

        // Add a random component to filenames to ensure uniqueness even if temp dir is
        // reused
        let random_id: u64 = rand::random();

        // Create key map and writer
        let key_path = dir.path().join(format!("test-key-segment-{}", random_id));
        let key_map = Arc::new(Map::new(key_path, 4096 * 10).expect("failed to create key map"));
        let key_writer = SegmentWriter::new(key_map.clone()).expect("failed to create key writer");

        // Create value map and writer
        let val_path = dir.path().join(format!("test-val-segment-{}", random_id));
        let val_map = Arc::new(Map::new(val_path, 4096 * 10).expect("failed to create val map"));
        let val_writer = SegmentWriter::new(val_map.clone()).expect("failed to create val writer");

        // Create segment
        let seed = 42i64; // Fixed seed for reproducibility
        let segment = Arc::new(Segment::new(1, 2, seed, key_writer, val_writer));

        (segment, dir)
    }

    // helper function to create test key-value pair
    fn create_kv(key: &str, value: &str, clock: &HybridLogicalClock) -> (KeyBytes, ValueBytes) {
        (
            KeyBytes::new(DEFAULT_NS, Bytes::from(key.to_string()), clock.time()),
            ValueBytes::new(DEFAULT_NS, Bytes::from(value.to_string())),
        )
    }

    #[test]
    fn test_segment_basic_write() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Simple key-value pair
        let key = [0u8, 0, 0, 0, 0, 0, 0, 0, b'a', b'b', b'c'];
        let val = [0u8, 0, 0, 0, 0, 0, 0, 0, b'1', b'2', b'3'];

        // Write to segment
        let result = segment.write(&key, &val);
        assert!(result.is_ok(), "Failed to write to segment: {:?}", result);
    }

    #[test]
    fn test_segment_multiple_writes() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Write multiple key-value pairs
        for i in 0u32..10 {
            let mut key = vec![0u8; 8]; // namespace
            key.extend_from_slice(&i.to_le_bytes());

            let mut val = vec![0u8; 8]; // namespace
            val.extend_from_slice(&(i * 10).to_le_bytes());

            let result = segment.write(&key, &val);
            assert!(result.is_ok(), "Failed to write entry {}: {:?}", i, result);
        }
    }

    #[test]
    fn test_segment_namespace_handling() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Write entries with different namespaces
        for ns in &[1u64, 2u64, 1u64, 3u64, 2u64] {
            let mut key = ns.to_le_bytes().to_vec();
            key.extend_from_slice(b"testkey");

            let mut val = ns.to_le_bytes().to_vec();
            val.extend_from_slice(b"testvalue");

            let result = segment.write(&key, &val);
            assert!(
                result.is_ok(),
                "Failed to write entry for ns {}: {:?}",
                ns,
                result
            );
        }

        // Check that the namespaces were recorded in the key index
        // Since we wrote 3 different namespaces with some repetition
        // We need to make sure the ns tracking works correctly
        assert!(
            segment.key_index.lock().ns_offset_count() >= 3,
            "Key index should track at least 3 namespace changes"
        );
    }

    #[test]
    fn test_segment_large_entry() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Create a key that fits in one block
        let mut key = vec![0u8; 8]; // namespace
        key.extend_from_slice(b"large_entry_key");

        // Create a large value that spans multiple blocks
        let mut val = vec![0u8; 8]; // namespace
        let large_data_size = 8192; // 2 blocks worth of data
        let mut large_data = vec![0u8; large_data_size];
        thread_rng().fill_bytes(&mut large_data);
        val.extend_from_slice(&large_data);

        let result = segment.write(&key, &val);
        assert!(result.is_ok(), "Failed to write large entry: {:?}", result);
    }

    #[test]
    fn test_segment_mixed_entry_sizes() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        let mut rng = thread_rng();

        // Write entries with varying sizes
        for i in 0u32..20 {
            let mut key = vec![0u8; 8]; // namespace
            key.extend_from_slice(&i.to_le_bytes());

            // Generate random-sized values
            let size = match i % 4 {
                | 0 => 10,   // Small
                | 1 => 1000, // Medium
                | 2 => 4000, // Almost a block
                | 3 => 6000, // Multi-block
                | _ => unreachable!(),
            };

            let mut val = vec![0u8; 8]; // namespace
            let mut data = vec![0u8; size];
            rng.fill_bytes(&mut data);
            val.extend_from_slice(&data);

            let result = segment.write(&key, &val);
            assert!(
                result.is_ok(),
                "Failed to write entry with size {}: {:?}",
                size,
                result
            );
        }
    }

    #[test]
    fn test_segment_reader_creation() {
        let (segment, _dir) = create_test_segment();

        // Get a new reader
        let reader = segment.new_reader();
        assert!(reader.is_ok());
    }

    #[test]
    fn test_segment_with_many_small_entries() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Write many small entries to test block packing
        for i in 0u32..1000 {
            let mut key = vec![0u8; 8]; // namespace
            key.extend_from_slice(&i.to_le_bytes());

            let mut val = vec![0u8; 8]; // namespace
            val.extend_from_slice(format!("Value{}", i).as_bytes());

            let result = segment.write(&key, &val);
            assert!(
                result.is_ok(),
                "Failed to write small entry {}: {:?}",
                i,
                result
            );
        }

        // Check blocks were created by looking at the writers
        assert!(
            segment.key_writer.lock().as_ref().unwrap().block_count() > 0,
            "Should have created some key blocks"
        );
        assert!(
            segment.val_writer.lock().as_ref().unwrap().block_count() > 0,
            "Should have created some value blocks"
        );
    }

    #[test]
    fn test_segment_sequential_keys() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Test with lexicographically ordered keys
        let mut data = vec![];
        for c in 'a'..='z' {
            let key = format!("key_{}", c);
            let value = format!("value_{}", c);

            let mut key_data = DEFAULT_NS.to_le_bytes().to_vec();
            key_data.extend_from_slice(key.as_bytes());

            let mut val_data = DEFAULT_NS.to_le_bytes().to_vec();
            val_data.extend_from_slice(value.as_bytes());

            data.push((key_data, val_data));
        }

        // Write in order
        for (key, val) in data {
            let result = segment.write(&key, &val);
            assert!(
                result.is_ok(),
                "Failed to write sequential key: {:?}",
                result
            );
        }
    }

    #[test]
    fn test_segment_random_order_keys() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Create entries with random ordering
        let mut data = vec![];
        for i in 0..100 {
            let key = format!("random_key_{:03}", i);
            let value = format!("random_value_{:03}", i);

            let mut key_data = DEFAULT_NS.to_le_bytes().to_vec();
            key_data.extend_from_slice(key.as_bytes());

            let mut val_data = DEFAULT_NS.to_le_bytes().to_vec();
            val_data.extend_from_slice(value.as_bytes());

            data.push((key_data, val_data));
        }

        // Shuffle the data
        let mut rng = rand::thread_rng();
        data.shuffle(&mut rng);

        // Write in random order
        for (key, val) in data {
            let result = segment.write(&key, &val);
            assert!(
                result.is_ok(),
                "Failed to write random-order key: {:?}",
                result
            );
        }
    }

    #[test]
    fn test_segment_index_building() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Write entries and check index growth
        let initial_key_blocks = segment.key_index.lock().block_count();

        for i in 0u32..512 {
            let mut key = vec![0u8; 8]; // namespace
            key.extend_from_slice(&i.to_le_bytes());

            let mut val = vec![0u8; 8]; // namespace
            val.extend_from_slice(&(i * 10).to_le_bytes());

            let result = segment.write(&key, &val);
            assert!(result.is_ok(), "Failed to write entry {}: {:?}", i, result);
        }

        segment.flush().expect("failed to flush segment");

        let final_key_blocks = segment.key_index.lock().block_count();

        assert_eq!(
            final_key_blocks, 4,
            "there should be 4 blocks in the key index for 512 entries with 10-byte value location metadata, found: {}",
            final_key_blocks
        );
    }

    #[test]
    fn test_segment_is_read_only_new_segment() {
        let (segment, _dir) = create_test_segment();

        // A newly created segment should NOT be read-only
        assert!(
            !segment.is_read_only(),
            "New segment should not be read-only"
        );
    }

    #[test]
    fn test_segment_close_makes_read_only() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Write some data
        let key = [0u8, 0, 0, 0, 0, 0, 0, 0, b'a', b'b', b'c'];
        let val = [0u8, 0, 0, 0, 0, 0, 0, 0, b'1', b'2', b'3'];
        segment.write(&key, &val).expect("failed to write");

        // Close the segment
        segment.close().expect("failed to close segment");

        // Now segment should be read-only
        assert!(
            segment.is_read_only(),
            "Segment should be read-only after close"
        );
    }

    #[test]
    fn test_segment_write_to_read_only_fails() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Write and close
        let key = [0u8, 0, 0, 0, 0, 0, 0, 0, b'a', b'b', b'c'];
        let val = [0u8, 0, 0, 0, 0, 0, 0, 0, b'1', b'2', b'3'];
        segment.write(&key, &val).expect("failed to write");
        segment.close().expect("failed to close");

        // Attempt to write should fail
        let result = segment.write(&key, &val);
        assert!(result.is_err(), "Writing to read-only segment should fail");
        assert!(
            matches!(result.err().unwrap(), ReadOnly),
            "Expected ReadOnly error"
        );
    }

    #[test]
    fn test_segment_flush_read_only_fails() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Write, close
        let key = [0u8, 0, 0, 0, 0, 0, 0, 0, b'a', b'b', b'c'];
        let val = [0u8, 0, 0, 0, 0, 0, 0, 0, b'1', b'2', b'3'];
        segment.write(&key, &val).expect("failed to write");
        segment.close().expect("failed to close");

        // Flush on read-only should fail
        let result = segment.flush();
        assert!(result.is_err(), "Flush on read-only segment should fail");
        assert!(
            matches!(result.err().unwrap(), ReadOnly),
            "Expected ReadOnly error"
        );
    }

    #[test]
    fn test_segment_close_twice() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Write some data
        let key = [0u8, 0, 0, 0, 0, 0, 0, 0, b'a', b'b', b'c'];
        let val = [0u8, 0, 0, 0, 0, 0, 0, 0, b'1', b'2', b'3'];
        segment.write(&key, &val).expect("failed to write");

        // Close first time
        segment.close().expect("failed to close");

        // Close second time should fail
        let result = segment.close();
        assert!(result.is_err(), "Second close should fail");
        assert!(
            matches!(result.err().unwrap(), ReadOnly),
            "Expected ReadOnly error"
        );
    }

    #[test]
    fn test_segment_id() {
        let (segment, _dir) = create_test_segment();

        // The test helper creates segment with key_id=1
        assert_eq!(segment.id(), 1, "Segment ID should be 1");
    }

    #[test]
    fn test_segment_new_reader_before_close() {
        let (mut segment, _dir) = create_test_segment();
        let segment_ref = Arc::get_mut(&mut segment).unwrap();

        // Write some data
        let key = [0u8, 0, 0, 0, 0, 0, 0, 0, b't', b'e', b's', b't'];
        let val = [0u8, 0, 0, 0, 0, 0, 0, 0, b'v', b'a', b'l'];
        segment_ref.write(&key, &val).expect("failed to write");
        segment_ref.flush().expect("failed to flush");

        // Get reader before closing
        let reader = segment_ref.new_reader();
        assert!(reader.is_ok(), "Should be able to create reader before close");
    }

    #[test]
    fn test_metadata_serialization() {
        let metadata = Metadata::new(12345, 100, 4096, 8192);

        // Convert to bytes
        let bytes: Bytes = metadata.into();
        assert_eq!(bytes.len(), 32, "Metadata should be 32 bytes");

        // Convert back
        let restored = Metadata::from(bytes);
        assert_eq!(restored.id(), 12345);
        assert_eq!(restored.block_count(), 100);
        assert_eq!(restored.index_size(), 4096);
        assert_eq!(restored.index_start(), 8192);
    }

    #[test]
    fn test_metadata_serialized_size() {
        let metadata = Metadata::new(1, 2, 3, 4);
        assert_eq!(
            metadata.serialized_size(),
            32,
            "Metadata should always serialize to 32 bytes"
        );
    }

    #[test]
    fn test_metadata_finalize_unsafe() {
        let metadata = Metadata::new(0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x87654321);

        // Allocate aligned buffer
        let mut buffer = vec![0u8; 32];

        // SAFETY: buffer is properly sized and aligned for this test
        unsafe {
            metadata.finalize(buffer.as_mut_ptr());
        }

        // Verify the data was written correctly
        let id = u64::from_le_bytes(buffer[0..8].try_into().unwrap());
        let block_count = u64::from_le_bytes(buffer[8..16].try_into().unwrap());
        let index_size = u64::from_le_bytes(buffer[16..24].try_into().unwrap());
        let index_start = u64::from_le_bytes(buffer[24..32].try_into().unwrap());

        assert_eq!(id, 0xDEADBEEF);
        assert_eq!(block_count, 0xCAFEBABE);
        assert_eq!(index_size, 0x12345678);
        assert_eq!(index_start, 0x87654321);
    }

    #[test]
    fn test_block_type_display() {
        assert_eq!(format!("{}", Key), "key");
        assert_eq!(format!("{}", Value), "value");
    }

    #[test]
    fn test_segment_empty_write() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Write entry with empty value (but still need namespace)
        let key = [0u8, 0, 0, 0, 0, 0, 0, 0, b'k', b'e', b'y'];
        let val = [0u8, 0, 0, 0, 0, 0, 0, 0]; // just namespace, no actual value

        let result = segment.write(&key, &val);
        assert!(result.is_ok(), "Should be able to write entry with empty value");
    }

    #[test]
    fn test_segment_flush_empty() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Flush without writing anything should succeed
        let result = segment.flush();
        assert!(result.is_ok(), "Flushing empty segment should succeed");
    }

    #[test]
    fn test_segment_multiple_flushes() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Write and flush multiple times
        for i in 0..5 {
            let mut key = vec![0u8; 8];
            key.extend_from_slice(&(i as u32).to_le_bytes());

            let mut val = vec![0u8; 8];
            val.extend_from_slice(b"value");

            segment.write(&key, &val).expect("failed to write");
            segment.flush().expect("failed to flush");
        }

        // Should have written blocks
        assert!(
            segment.key_index.lock().block_count() >= 5,
            "Should have created multiple key blocks"
        );
    }

    #[test]
    fn test_segment_boundary_entry_size() {
        use crate::block::MAX_ENTRY_SIZE;

        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Create entry just at the boundary of max size
        let key = vec![0u8; 8 + 10]; // namespace + small key

        // Create value that's close to but not exceeding MAX_ENTRY_SIZE
        let val_size = MAX_ENTRY_SIZE - 20; // Leave some room for overhead
        let mut val = vec![0u8; 8]; // namespace
        val.extend(vec![b'x'; val_size]);

        let result = segment.write(&key, &val);
        assert!(result.is_ok(), "Should be able to write entry near max size");
    }

    #[test]
    fn test_segment_very_large_entry_multiblock() {
        use crate::block::MAX_ENTRY_SIZE;

        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Create entry that must span multiple blocks
        let key = vec![0u8; 8 + 10];

        // Create value much larger than MAX_ENTRY_SIZE
        let mut val = vec![0u8; 8];
        val.extend(vec![b'y'; MAX_ENTRY_SIZE * 3]);

        let result = segment.write(&key, &val);
        assert!(result.is_ok(), "Should be able to write multi-block entry");

        segment.flush().expect("failed to flush");

        // Value should span multiple blocks
        let val_block_count = segment.val_block_count.load(Relaxed);
        assert!(val_block_count >= 3, "Value should span at least 3 blocks");
    }

    #[test]
    fn test_segment_concurrent_readers() {
        let (segment, _dir) = create_test_segment();

        // Create multiple readers
        let reader1 = segment.new_reader();
        let reader2 = segment.new_reader();
        let reader3 = segment.new_reader();

        assert!(reader1.is_ok(), "First reader should be created");
        assert!(reader2.is_ok(), "Second reader should be created");
        assert!(reader3.is_ok(), "Third reader should be created");
    }

    #[test]
    fn test_segment_write_read_verify() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Write multiple entries
        let entries: Vec<(Vec<u8>, Vec<u8>)> = (0..10)
            .map(|i| {
                let mut key = vec![0u8; 8];
                key.extend(format!("key_{:03}", i).as_bytes());
                let mut val = vec![0u8; 8];
                val.extend(format!("value_{:03}", i).as_bytes());
                (key, val)
            })
            .collect();

        for (key, val) in &entries {
            segment.write(key, val).expect("failed to write");
        }
        segment.flush().expect("failed to flush");

        // Get reader and verify entries
        let reader = segment.new_reader().expect("failed to create reader");

        for (key, expected_val) in &entries {
            // Strip the value location metadata when looking up
            let result = reader.get(key);
            assert!(result.is_ok(), "Get should not error");
            let found = result.unwrap();
            assert!(found.is_some(), "Key should be found: {:?}", key);
            assert_eq!(found.unwrap().as_ref(), expected_val.as_slice());
        }
    }

    #[test]
    fn test_segment_namespace_changes() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Write entries with different namespaces
        let namespaces: Vec<u64> = vec![0, 1, 2, 0, 3, 1, 4, 2];

        for (i, &ns) in namespaces.iter().enumerate() {
            let mut key = ns.to_le_bytes().to_vec();
            key.extend(format!("key_{}", i).as_bytes());

            let mut val = ns.to_le_bytes().to_vec();
            val.extend(format!("val_{}", i).as_bytes());

            segment.write(&key, &val).expect("failed to write");
        }

        // Check that namespace changes were tracked
        let ns_count = segment.key_index.lock().ns_offset_count();
        // Should track unique namespaces: 0, 1, 2, 3, 4 = 5 unique
        // Plus initial default namespace insertion = at least 5
        assert!(ns_count >= 5, "Should track at least 5 namespace offsets, got {}", ns_count);
    }

    #[test]
    fn test_segment_val_block_count_tracking() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        let initial_count = segment.val_block_count.load(Relaxed);
        assert_eq!(initial_count, 0, "Initial val_block_count should be 0");

        // Write enough data to create value blocks
        for i in 0..100 {
            let mut key = vec![0u8; 8];
            key.extend(&(i as u32).to_le_bytes());

            let mut val = vec![0u8; 8];
            val.extend(vec![b'v'; 100]); // 100 bytes of value data

            segment.write(&key, &val).expect("failed to write");
        }

        segment.flush().expect("failed to flush");

        let final_count = segment.val_block_count.load(Relaxed);
        assert!(final_count > 0, "val_block_count should have increased");
    }

    #[test]
    fn test_segment_open_creates_read_only_segment() {
        use crate::index::Index;

        let dir = tempdir().expect("failed to create temp dir");

        // Create key and value maps
        let key_path = dir.path().join("test-key-segment");
        let key_map = Arc::new(Map::new(key_path, 4096 * 10).expect("failed to create key map"));

        let val_path = dir.path().join("test-val-segment");
        let val_map = Arc::new(Map::new(val_path, 4096 * 10).expect("failed to create val map"));

        let seed = 42i64;
        let key_index = Index::new(1, seed);

        // Open segment (creates read-only segment)
        let segment = Segment::open(key_map, key_index, 1, val_map, 2);
        assert!(segment.is_ok(), "Should be able to open segment");

        let segment = segment.unwrap();
        assert!(segment.is_read_only(), "Opened segment should be read-only");
    }

    #[test]
    fn test_segment_open_write_fails() {
        use crate::index::Index;

        let dir = tempdir().expect("failed to create temp dir");

        let key_path = dir.path().join("test-key-segment");
        let key_map = Arc::new(Map::new(key_path, 4096 * 10).expect("failed to create key map"));

        let val_path = dir.path().join("test-val-segment");
        let val_map = Arc::new(Map::new(val_path, 4096 * 10).expect("failed to create val map"));

        let seed = 42i64;
        let key_index = Index::new(1, seed);

        let segment = Segment::open(key_map, key_index, 1, val_map, 2).unwrap();

        // Try to write to opened (read-only) segment
        let key = [0u8, 0, 0, 0, 0, 0, 0, 0, b'k', b'e', b'y'];
        let val = [0u8, 0, 0, 0, 0, 0, 0, 0, b'v', b'a', b'l'];

        // We need Arc::get_mut but segment is already Arc, so we need to unwrap
        // Since Segment::open returns Arc<Segment>, we can use the reference directly
        // But write takes &self, so this should work
        let result = segment.write(&key, &val);
        assert!(result.is_err(), "Writing to opened segment should fail");
        assert!(matches!(result.err().unwrap(), ReadOnly));
    }

    #[test]
    fn test_segment_size_in_bytes_new_segment() {
        let (segment, _dir) = create_test_segment();

        // New segment with writers won't have handles set, so size is 0
        let size = segment.size_in_bytes();
        assert_eq!(size, 0, "New segment should report 0 size (no handles yet)");
    }

    #[test]
    fn test_segment_size_in_bytes_opened_segment() {
        use crate::index::Index;

        let dir = tempdir().expect("failed to create temp dir");

        let key_path = dir.path().join("test-key-segment");
        let key_map = Arc::new(Map::new(key_path, 4096 * 5).expect("failed to create key map"));

        let val_path = dir.path().join("test-val-segment");
        let val_map = Arc::new(Map::new(val_path, 4096 * 3).expect("failed to create val map"));

        let seed = 42i64;
        let key_index = Index::new(1, seed);

        let segment = Segment::open(key_map, key_index, 1, val_map, 2).unwrap();

        // Opened segment should report size from handles
        let size = segment.size_in_bytes();
        assert_eq!(size, 4096 * 5 + 4096 * 3, "Opened segment should report map sizes");
    }

    #[test]
    fn test_segment_reader_method_on_opened_segment() {
        use crate::index::Index;
        use crate::block::Block;

        let dir = tempdir().expect("failed to create temp dir");

        let key_path = dir.path().join("test-key-segment");
        let key_map = Arc::new(Map::new(key_path, 4096 * 5).expect("failed to create key map"));

        let val_path = dir.path().join("test-val-segment");
        let val_map = Arc::new(Map::new(val_path, 4096 * 5).expect("failed to create val map"));

        let seed = 42i64;
        let key_index = Index::new(1, seed);

        let segment = Segment::open(key_map, key_index, 1, val_map, 2).unwrap();

        // reader() method should work on read-only segment
        let reader = segment.reader();
        assert!(reader.is_ok(), "reader() should succeed on read-only segment");
    }

    #[test]
    fn test_segment_reader_method_on_new_segment_fails() {
        let (segment, _dir) = create_test_segment();

        // reader() method should fail on writable segment
        let reader = segment.reader();
        assert!(reader.is_err(), "reader() should fail on writable segment");
    }
}
