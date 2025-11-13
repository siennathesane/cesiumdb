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

use parking_lot::Mutex;

use bytes::{
    BufMut,
    Bytes,
    BytesMut,
};

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
        debug_assert!(
            !dst.is_null(),
            "Destination pointer must not be null"
        );
        debug_assert!(
            dst as usize % std::mem::align_of::<u64>() == 0,
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
        let (value_block_num, value_entry_index) = {
            let mut val_block = self.current_val_block.lock();

            match val_block.add_entry(val, Complete) {
                | Ok(()) => {
                    // Value added successfully to current block
                    let block_num = self.val_block_count.load(Relaxed);
                    let entry_idx = (val_block.num_entries() - 1) as u16;
                    drop(val_block);
                    (block_num, entry_idx)
                },
                | Err(be) => {
                    drop(val_block);

                    match be {
                        | BlockError::TooLargeForBlock => {
                            // Split returns where the START entry was placed
                            match self.split_across_blocks(val, &Value) {
                                | Ok((block_num, entry_idx)) => (block_num, entry_idx),
                                | Err(e) => return Err(e),
                            }
                        },
                        | BlockError::CorruptedBlock => {
                            unreachable!("unexpected corrupted block error during write")
                        },
                        | BlockError::BlockFull => {
                            match self.write_block(&Value) {
                                | Ok(_) => {},
                                | Err(e) => return Err(e),
                            };

                            let mut val_block = self.current_val_block.lock();
                            match val_block.add_entry(val, Complete) {
                                | Ok(_) => {
                                    // Value added successfully to new block
                                    let block_num = self.val_block_count.load(Relaxed);
                                    let entry_idx = (val_block.num_entries() - 1) as u16;
                                    (block_num, entry_idx)
                                },
                                | Err(rbe) => match rbe {
                                    | BlockError::TooLargeForBlock => {
                                        drop(val_block);
                                        // Split returns where the START entry was placed
                                        match self.split_across_blocks(val, &Value) {
                                            | Ok((block_num, entry_idx)) => (block_num, entry_idx),
                                            | Err(e) => return Err(e),
                                        }
                                    },
                                    | BlockError::CorruptedBlock | BlockError::BlockFull => {
                                        unreachable!("unexpected val block error, no idea how we got here")
                                    },
                                },
                            }
                        },
                    }
                },
            }
        };

        // Now write key with embedded value location metadata
        // Format: [value_block_num:u64][value_entry_index:u16][key_data]
        let mut key_with_metadata = BytesMut::with_capacity(10 + key.len());
        key_with_metadata.put_u64_le(value_block_num);
        key_with_metadata.put_u16_le(value_entry_index);
        key_with_metadata.put_slice(key);

        {
            let mut key_block = self.current_key_block.lock();
            match key_block.add_entry(&key_with_metadata, Complete) {
                | Ok(()) => {},
                | Err(be) => {
                    drop(key_block);

                    match be {
                        | BlockError::TooLargeForBlock => {
                            // Key split - we already have value location in metadata, so ignore return value
                            match self.split_across_blocks(&key_with_metadata, &Key) {
                                | Ok(_) => {},
                                | Err(e) => return Err(e),
                            }
                        },
                        | BlockError::CorruptedBlock => {
                            unreachable!("unexpected corrupted block error during key write")
                        },
                        | BlockError::BlockFull => {
                            match self.write_block(&Key) {
                                | Ok(_) => {},
                                | Err(e) => return Err(e),
                            };

                            let mut key_block = self.current_key_block.lock();
                            match key_block.add_entry(&key_with_metadata, Complete) {
                                | Ok(_) => {},
                                | Err(rbe) => match rbe {
                                    | BlockError::TooLargeForBlock => {
                                        drop(key_block);
                                        // Key split - we already have value location in metadata, so ignore return value
                                        match self.split_across_blocks(&key_with_metadata, &Key) {
                                            | Ok(_) => {},
                                            | Err(e) => return Err(e),
                                        }
                                    },
                                    | BlockError::CorruptedBlock | BlockError::BlockFull => {
                                        unreachable!("unexpected key block error, no idea how we got here")
                                    },
                                },
                            }
                        },
                    }
                },
            }
        }

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

        SegmentReader::new(km, vm, self.key_index.clone())
    }

    /// Flush any pending blocks
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
                    | Ok(_) => {},
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
                    | Ok(_) => {},
                    | Err(e) => return Err(e),
                };
            }
        }

        Ok(())
    }

    /// Split a payload across multiple blocks.
    /// Returns (block_num, entry_index) where the START entry was placed.
    fn split_across_blocks(&self, data: &[u8], r#type: &BlockType) -> Result<(u64, u16), SegmentError> {
        let mut remaining = data;

        // First, determine the maximum chunk size that can fit in a block
        // Account for the entry flag byte and other overhead
        let max_chunk_size = MAX_ENTRY_SIZE - 1;

        // Track where the START entry is placed
        let start_block_num: u64;
        let start_entry_index: u16 = 0; // Always at index 0 in fresh block

        // Process the first chunk (START flag)
        if !remaining.is_empty() {
            let chunk_size = std::cmp::min(max_chunk_size, remaining.len());
            let chunk = &remaining[..chunk_size];

            match r#type {
                | Key => {
                    // Always start with a new block
                    {
                        let key_block = self.current_key_block.lock();
                        if !key_block.is_empty() {
                            drop(key_block);
                            match self.write_block(&Key) {
                                | Ok(_) => {},
                                | Err(e) => return Err(e),
                            };
                        }
                    }

                    // Capture block number before adding START entry
                    start_block_num = self.key_index.lock().block_count();

                    {
                        let mut key_block = self.current_key_block.lock();
                        match key_block.add_entry(chunk, Start) {
                            | Ok(_) => {
                                drop(key_block);
                                self.key_index.lock().insert_item(data);
                                match self.write_block(&Key) {
                                    | Ok(_) => {},
                                    | Err(e) => return Err(e),
                                };
                            },
                            | Err(e) => {
                                eprintln!(
                                    "Error adding START chunk to key block: {:?}, chunk size: {}",
                                    e,
                                    chunk.len()
                                );
                                return Err(SegmentError::InsufficientSpace);
                            },
                        }
                    }
                },
                | Value => {
                    // Always start with a new block
                    {
                        let val_block = self.current_val_block.lock();
                        if !val_block.is_empty() {
                            drop(val_block);
                            match self.write_block(&Value) {
                                | Ok(_) => {},
                                | Err(e) => return Err(e),
                            };
                        }
                    }

                    // Capture block number before adding START entry
                    start_block_num = self.val_block_count.load(Relaxed);

                    {
                        let mut val_block = self.current_val_block.lock();
                        match val_block.add_entry(chunk, Start) {
                            | Ok(_) => {
                                drop(val_block);
                                match self.write_block(&Value) {
                                    | Ok(_) => {},
                                    | Err(e) => return Err(e),
                                };
                            },
                            | Err(e) => {
                                eprintln!(
                                    "Error adding START chunk to value block: {:?}, chunk size: {}",
                                    e,
                                    chunk.len()
                                );
                                return Err(SegmentError::InsufficientSpace);
                            },
                        }
                    }
                },
            }

            remaining = &remaining[chunk_size..];
        } else {
            return Err(SegmentError::InsufficientSpace);
        }

        // Process middle chunks (MIDDLE flag)
        while remaining.len() > max_chunk_size {
            let chunk = &remaining[..max_chunk_size];

            match r#type {
                | Key => {
                    let mut block = Block::new();
                    match block.add_entry(chunk, Middle) {
                        | Ok(_) => {
                            let mut writer_guard = self.key_writer.lock();
                            match writer_guard.as_mut() {
                                | Some(writer) => {
                                    match writer.write_block(block) {
                                        | Ok(_) => {
                                            drop(writer_guard);
                                            // Sync index block count
                                            self.key_index.lock().inc_block_count(1);
                                        },
                                        | Err(e) => return Err(e),
                                    }
                                },
                                | None => return Err(ReadOnly),
                            }
                        },
                        | Err(e) => {
                            eprintln!(
                                "Error adding MIDDLE chunk to key block: {:?}, chunk size: {}",
                                e,
                                chunk.len()
                            );
                            return Err(SegmentError::InsufficientSpace);
                        },
                    }
                },
                | Value => {
                    let mut block = Block::new();
                    match block.add_entry(chunk, Middle) {
                        | Ok(_) => {
                            let mut writer_guard = self.val_writer.lock();
                            match writer_guard.as_mut() {
                                | Some(writer) => {
                                    match writer.write_block(block) {
                                        | Ok(_) => {
                                            drop(writer_guard);
                                            // Increment block count
                                            self.val_block_count.fetch_add(1, Relaxed);
                                        },
                                        | Err(e) => return Err(e),
                                    }
                                },
                                | None => return Err(ReadOnly),
                            }
                        },
                        | Err(e) => {
                            eprintln!(
                                "Error adding MIDDLE chunk to value block: {:?}, chunk size: {}",
                                e,
                                chunk.len()
                            );
                            return Err(SegmentError::InsufficientSpace);
                        },
                    }
                },
            }

            remaining = &remaining[max_chunk_size..];
        }

        // Process the last chunk (END flag) if there's anything left
        if !remaining.is_empty() {
            match r#type {
                | Key => {
                    let mut block = Block::new();
                    match block.add_entry(remaining, End) {
                        | Ok(_) => {
                            let mut writer_guard = self.key_writer.lock();
                            match writer_guard.as_mut() {
                                | Some(writer) => {
                                    match writer.write_block(block) {
                                        | Ok(_) => {
                                            drop(writer_guard);
                                            // Sync index block count
                                            self.key_index.lock().inc_block_count(1);
                                        },
                                        | Err(e) => return Err(e),
                                    }
                                },
                                | None => return Err(ReadOnly),
                            }
                        },
                        | Err(e) => {
                            eprintln!(
                                "Error adding END chunk to key block: {:?}, chunk size: {}",
                                e,
                                remaining.len()
                            );
                            return Err(SegmentError::InsufficientSpace);
                        },
                    }
                },
                | Value => {
                    let mut block = Block::new();
                    match block.add_entry(remaining, End) {
                        | Ok(_) => {
                            let mut writer_guard = self.val_writer.lock();
                            match writer_guard.as_mut() {
                                | Some(writer) => {
                                    match writer.write_block(block) {
                                        | Ok(_) => {
                                            drop(writer_guard);
                                            // Increment block count
                                            self.val_block_count.fetch_add(1, Relaxed);
                                        },
                                        | Err(e) => return Err(e),
                                    }
                                },
                                | None => return Err(ReadOnly),
                            }
                        },
                        | Err(e) => {
                            eprintln!(
                                "Error adding END chunk to value block: {:?}, chunk size: {}",
                                e,
                                remaining.len()
                            );
                            return Err(SegmentError::InsufficientSpace);
                        },
                    }
                },
            }
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
                let result = {
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
                        | None => return Err(ReadOnly),
                    }
                };

                result
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
                let result = {
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
                        | None => return Err(ReadOnly),
                    }
                };

                result
            },
        }
    }

    pub(crate) fn close(&self) -> Result<(), SegmentError> {
        if self.key_writer.lock().is_none() {
            return Err(ReadOnly);
        }

        match self.flush() {
            | Ok(_) => {},
            | Err(e) => return Err(e),
        }

        {
            let mut writer_guard = self.key_writer.lock();
            if let Some(writer) = writer_guard.as_mut() {
                // Index block count already synced during writes
                let block_count = writer.block_count();

                let key_index = self.key_index.lock();
                let index_size = key_index.size();
                let index_start = match writer.write_index(&*key_index) {
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
                    | Ok(_) => {},
                    | Err(e) => return Err(e),
                };
                match writer.close() {
                    | Ok(_) => {},
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

                // Value segments don't need an index since value locations are stored in key metadata
                // Mark as closing before writing metadata
                writer.begin_close();

                // Write minimal metadata (no index)
                match writer.write_metadata(Metadata::new(
                    self.val_id,
                    block_count,
                    0, // no index
                    0, // no index
                )) {
                    | Ok(_) => {},
                    | Err(e) => return Err(e),
                };
                match writer.close() {
                    | Ok(_) => {},
                    | Err(e) => return Err(e),
                };
            }
            // Set writer to None to mark segment as read-only
            *writer_guard = None;
        }

        Ok(())
    }
}

impl Drop for Segment {
    fn drop(&mut self) {
        let res = self.close();
        if let Err(e) = res && !matches!(e, ReadOnly) {
            // Log the error instead of panicking during Drop to avoid double-panic situations
            // and unwinding issues. This error indicates that segment metadata may not have
            // been flushed properly, which could lead to data loss on reopening.
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
}
