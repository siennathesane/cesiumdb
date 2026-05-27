use std::{
    fmt::Display,
    ptr,
    sync::{
        Arc,
        atomic::{
            AtomicU64,
            Ordering::Relaxed,
        },
    },
};

use bytes::{
    Bytes,
    BytesMut,
};
use parking_lot::Mutex;

use crate::{
    block::{
        BLOCK_SIZE,
        EntryFlag,
        EntryFlag::{
            Complete,
            End,
            Middle,
            Start,
        },
        MAX_ENTRY_SIZE,
        ReadOnlyBlock,
    },
    utils::Deserializer,
    errs::{
        SegmentError,
        SegmentError::{
            CantCreateReader,
            CorruptedBlock,
            MissingKey,
            ReadOnly,
            ReadOutOfBounds,
        },
    },
    index::Index,
    keypair::DEFAULT_NS,
    map::Map,
    scan::OwnedSegmentIterator,
    segment::BlockType::{
        Key,
        Value,
    },
    segment_reader::{BlockCache, SegmentReader},
    segment_writer::SegmentWriter,
};

// Constants for value location metadata embedded in keys
/// Size of value location metadata: u64 (block_num) + u16 (entry_index) = 10
/// bytes
pub(crate) const VALUE_LOCATION_SIZE: usize = size_of::<u64>() + size_of::<u16>();
/// Offset where actual key data starts (after metadata)
pub(crate) const KEY_DATA_OFFSET: usize = VALUE_LOCATION_SIZE;

// Segment file size constants
/// Default segment size: 64 MiB
pub(crate) const DEFAULT_SEGMENT_SIZE: u64 = 64 * 1024 * 1024;

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
    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub(crate) fn new(id: u64, block_count: u64, index_size: u64, index_start: u64) -> Self {
        Self {
            id,
            block_count,
            index_size,
            index_start,
        }
    }

    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
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

    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub(crate) fn block_count(&self) -> usize {
        self.block_count as usize
    }

    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub(crate) fn index_size(&self) -> usize {
        self.index_size as usize
    }

    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
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

/// Accumulator for block entries - uses same structure as BlockBuilder
/// to avoid copying. Tuple: (offsets: Vec<u16>, entries: Vec<u8>)
type BlockAccumulator = (Vec<u16>, Vec<u8>);

pub struct Segment {
    // writers are used to write data whereas the map is used to read data

    // keys
    key_writer: Mutex<Option<SegmentWriter>>,
    key_handle: Option<Arc<Map>>,
    key_index: Arc<parking_lot::RwLock<Index>>,
    current_key_entries: Mutex<BlockAccumulator>,
    key_id: u64,

    // values
    val_writer: Mutex<Option<SegmentWriter>>,
    val_handle: Option<Arc<Map>>,
    current_val_entries: Mutex<BlockAccumulator>,
    val_block_count: AtomicU64,
    val_id: u64,

    // shared
    current_ns: AtomicU64,

    /// Total bytes written (key + value) while segment is open.
    /// Used as fallback for `size_in_bytes()` when handles aren't available
    /// yet.
    bytes_written: AtomicU64,

    /// Small per-segment value block cache for hot keys.
    val_block_cache: parking_lot::Mutex<BlockCache<8>>,

    /// Pre-computed visible block counts (valid for read-only segments).
    visible_key_blocks: usize,
    visible_val_blocks: usize,
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
            key_index: Arc::new(parking_lot::RwLock::new(key_index)),
            current_key_entries: Mutex::new((Vec::new(), Vec::new())),
            current_val_entries: Mutex::new((Vec::new(), Vec::new())),
            val_block_count: AtomicU64::new(0),
            current_ns: AtomicU64::new(DEFAULT_NS),
            key_id,
            val_id,
            bytes_written: AtomicU64::new(0),
            val_block_cache: parking_lot::Mutex::new(BlockCache::new()),
            visible_key_blocks: 0,
            visible_val_blocks: 0,
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
        val_block_count: u64,
    ) -> Result<Arc<Segment>, SegmentError> {
        // For opened segments, compute bytes_written from the map sizes
        let total_bytes = key_map.len() as u64 + val_map.len() as u64;

        // Compute visible key blocks (same logic as SegmentReader)
        let segment_size = key_map.len();
        let metadata_block_count = if segment_size >= 32 {
            match key_map.read_range(segment_size - 32..segment_size, |slice| {
                Bytes::copy_from_slice(slice)
            }) {
                | Ok(bytes) => {
                    let m = Metadata::from(bytes);
                    let index_end = m.index_start() + m.index_size();
                    if index_end + 32 == segment_size {
                        Some(m.block_count())
                    } else {
                        None
                    }
                },
                | Err(_) => None,
            }
        } else {
            None
        };
        let index_blocks = key_index.num_blocks() as usize;
        let num_blocks = segment_size.div_ceil(BLOCK_SIZE);
        let visible_key_blocks = if let Some(block_count) = metadata_block_count {
            block_count
        } else if index_blocks > 0 {
            index_blocks
        } else if segment_size >= DEFAULT_SEGMENT_SIZE as usize {
            0
        } else {
            num_blocks
        };
        let visible_val_blocks = val_map.len() / BLOCK_SIZE;

        Ok(Arc::new(Segment {
            key_writer: Mutex::new(None),
            key_handle: Some(key_map),
            val_writer: Mutex::new(None),
            val_handle: Some(val_map),
            key_index: Arc::new(parking_lot::RwLock::new(key_index)),
            current_key_entries: Mutex::new((Vec::new(), Vec::new())),
            current_val_entries: Mutex::new((Vec::new(), Vec::new())),
            val_block_count: AtomicU64::new(val_block_count),
            current_ns: AtomicU64::new(DEFAULT_NS),
            key_id,
            val_id,
            bytes_written: AtomicU64::new(total_bytes),
            val_block_cache: parking_lot::Mutex::new(BlockCache::new()),
            visible_key_blocks,
            visible_val_blocks,
        }))
    }

    /// Returns true if this segment is read-only (opened from disk).
    pub fn is_read_only(&self) -> bool {
        self.key_writer.lock().is_none()
    }

    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn write(&self, key: &[u8], val: &[u8]) -> Result<(), SegmentError> {
        

        if self.key_writer.lock().is_none() {
            return Err(ReadOnly);
        }

        // set the namespace
        let ns = u64::from_le_bytes(key[0..8].as_ref().try_into().unwrap());
        if ns != self.current_ns.load(Relaxed) {
            self.current_ns.store(ns, Relaxed);
            self.key_index.write().insert_ns_offset(ns);
        }

        // Write value to value block FIRST so we know where it lands
        let (value_block_num, value_entry_index) = match self.add_entry_with_retry(val, &Value) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };

        // Now write key with embedded value location metadata
        // Format: [value_block_num:u64][value_entry_index:u16][key_data]
        // Build directly in Vec to avoid BytesMut allocation
        let mut key_with_metadata = Vec::with_capacity(VALUE_LOCATION_SIZE + key.len());
        key_with_metadata.extend_from_slice(&value_block_num.to_le_bytes());
        key_with_metadata.extend_from_slice(&value_entry_index.to_le_bytes());
        key_with_metadata.extend_from_slice(key);

        // Write key with embedded value location metadata
        // Note: We ignore the return value since we already have the value location
        if let Err(e) = self.add_entry_with_retry(&key_with_metadata, &Key) {
            return Err(e);
        }

        // Strip timestamp (last 16 bytes) before indexing
        // Index should only hash [ns:8][user_key] to map all versions to same block
        // Keys must be serialized: [ns:8][user_key][timestamp:16], minimum 24 bytes
        debug_assert!(
            key.len() >= 24,
            "Key too short: {} bytes. Keys must be serialized with KeyBytes::serialize()",
            key.len()
        );
        // DEFERRED: Index rebuilt after close() for better compaction performance
        // let key_without_ts = &key[..key.len() - 16];
        // self.key_index.write().insert_item(key_without_ts);

        Ok(())
    }

    /// Helper method to add an entry with retry logic.
    /// Returns (block_num, entry_index) for where the entry was placed.
    ///
    /// This handles the common pattern of:
    /// 1. Try adding to current entries
    /// 2. If too large -> split across blocks
    /// 3. If block full -> flush block and retry
    fn add_entry_with_retry(
        &self,
        data: &[u8],
        block_type: &BlockType,
    ) -> Result<(u64, u16), SegmentError> {
        let (entries_mutex, block_counter) = match block_type {
            | Key => (&self.current_key_entries, None),
            | Value => (&self.current_val_entries, Some(&self.val_block_count)),
        };

        let mut accumulator = entries_mutex.lock();
        let (ref mut offsets, ref mut entries) = *accumulator;

        // Calculate space needed: entry data + flag byte + offset (u16)
        let entry_size = data.len() + size_of::<u8>();
        let space_needed = entry_size + size_of::<u16>();

        // Calculate current space used
        let current_used: usize = size_of::<u16>() // num_entries header
            + offsets.len() * size_of::<u16>() // offsets
            + entries.len(); // entries (already includes flags)

        // Check if entry is too large for any block
        if entry_size > MAX_ENTRY_SIZE {
            drop(accumulator);
            return self.split_across_blocks(data, block_type);
        }

        // Check if entry will fit in current block
        if current_used + space_needed <= BLOCK_SIZE {
            // Entry fits! Append directly to vecs (ZERO intermediate copy!)
            let entry_idx = offsets.len() as u16;

            // Calculate cumulative offset (same as Block/BlockBuilder)
            let current_offset = if offsets.is_empty() {
                0
            } else {
                offsets[offsets.len() - 1]
            };
            let next_offset = current_offset + (entry_size as u16);
            offsets.push(next_offset);

            // Append flag + data
            entries.push(Complete as u8);
            entries.extend_from_slice(data);

            drop(accumulator);

            let block_num = match block_counter {
                | Some(counter) => counter.load(Relaxed),
                | None => 0,
            };
            Ok((block_num, entry_idx))
        } else {
            // Block is full, flush it
            drop(accumulator);
            if let Err(e) = self.write_block(block_type) {
                return Err(e);
            }

            // Retry with fresh block
            let mut accumulator = entries_mutex.lock();
            let (ref mut offsets, ref mut entries) = *accumulator;

            let entry_idx = offsets.len() as u16;

            // Append to fresh vecs
            let entry_size_u16 = entry_size as u16;
            offsets.push(entry_size_u16);
            entries.push(Complete as u8);
            entries.extend_from_slice(data);

            drop(accumulator);

            let block_num = match block_counter {
                | Some(counter) => counter.load(Relaxed),
                | None => 0,
            };
            Ok((block_num, entry_idx))
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
        use crate::block::BlockBuilder;

        match block_type {
            | Key => {
                let mut writer_guard = self.key_writer.lock();
                let writer = match writer_guard.as_mut().ok_or(ReadOnly) {
                    | Ok(w) => w,
                    | Err(e) => return Err(e),
                };

                // Build block directly in mmap
                if let Err(e) = writer.write_block_direct(|mmap_slice| {
                    let mut builder = BlockBuilder::new(mmap_slice);
                    let _ = builder.add_entry(chunk, flag);
                    builder.finalize();
                }) {
                    return Err(e);
                }

                drop(writer_guard);
                self.key_index.write().inc_block_count(1);
            },
            | Value => {
                let mut writer_guard = self.val_writer.lock();
                let writer = match writer_guard.as_mut().ok_or(ReadOnly) {
                    | Ok(w) => w,
                    | Err(e) => return Err(e),
                };

                // Build block directly in mmap
                if let Err(e) = writer.write_block_direct(|mmap_slice| {
                    let mut builder = BlockBuilder::new(mmap_slice);
                    let _ = builder.add_entry(chunk, flag);
                    builder.finalize();
                }) {
                    return Err(e);
                }

                drop(writer_guard);
                self.val_block_count.fetch_add(1, Relaxed);
            },
        }

        Ok(())
    }

    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn flush(&self) -> Result<(), SegmentError> {
        if self.key_writer.lock().is_none() {
            return Err(ReadOnly);
        }

        // Flush key entries if any exist
        {
            let key_entries = self.current_key_entries.lock();
            if !key_entries.0.is_empty() {
                drop(key_entries);
                match self.write_block(&Key) {
                    | Ok(_) => {},
                    | Err(e) => return Err(e),
                };
            }
        }

        // Flush value entries if any exist
        {
            let val_entries = self.current_val_entries.lock();
            if !val_entries.0.is_empty() {
                drop(val_entries);
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

        // Ensure current entries are flushed before starting
        {
            let entries = match r#type {
                | Key => self.current_key_entries.lock(),
                | Value => self.current_val_entries.lock(),
            };
            if !entries.0.is_empty() {
                drop(entries);
                if let Err(e) = self.write_block(r#type) {
                    return Err(e);
                }
            }
        }

        // Capture the block number where START will be placed
        let start_block_num = match r#type {
            | Key => self.key_index.write().block_count(),
            | Value => self.val_block_count.load(Relaxed),
        };

        // Write START chunk
        let chunk_size = std::cmp::min(max_chunk_size, remaining.len());
        let chunk = &remaining[..chunk_size];

        // For keys, also insert into index
        if matches!(r#type, Key) {
            // data format: [val_loc:10][ns:8][user_key][inv_ts:16]
            // Strip value location (first 10 bytes) and timestamp (last 16 bytes)
            // to get [ns:8][user_key] for indexing
            debug_assert!(
                data.len() > 26,
                "Multi-block key too short: {} bytes",
                data.len()
            );
            // DEFERRED: Index rebuilt after close()
            // let key_without_ts = &data[10..data.len() - 16];
            // self.key_index.write().insert_item(key_without_ts);
        }

        if let Err(e) = self.write_chunk_to_new_block(chunk, Start, r#type) {
            return Err(e);
        }
        remaining = &remaining[chunk_size..];

        // Process middle chunks (MIDDLE flag)
        while remaining.len() > max_chunk_size {
            let chunk = &remaining[..max_chunk_size];
            if let Err(e) = self.write_chunk_to_new_block(chunk, Middle, r#type) {
                return Err(e);
            }
            remaining = &remaining[max_chunk_size..];
        }

        // Process the last chunk (END flag) if there's anything left
        if !remaining.is_empty()
            && let Err(e) = self.write_chunk_to_new_block(remaining, End, r#type) {
                return Err(e);
            }

        Ok((start_block_num, start_entry_index))
    }

    fn write_block(&self, r#type: &BlockType) -> Result<(), SegmentError> {
        use crate::block::BlockBuilder;

        match r#type {
            | Key => {
                // Take pre-built vecs (ZERO intermediate copy!)
                let (offsets, entries) = {
                    let mut key_acc = self.current_key_entries.lock();
                    if key_acc.0.is_empty() {
                        return Ok(()); // Nothing to write
                    }
                    std::mem::take(&mut *key_acc)
                };

                // DEFERRED: Index rebuilt after close()
                // Extract first key for indexing from entries vec
                // First entry starts at offset 0, format: [flag:1][data...]
                // if !entries.is_empty() && entries.len() > 11 {
                //     // Skip flag byte (1), then key format:
                // [val_loc:10][ns:8][user_key][inv_ts:16]     let key_data =
                // &entries[1..]; // Skip flag byte     if key_data.len() > 26 {
                //         let key_without_ts = &key_data[10..key_data.len() - 16];
                //         self.key_index.write().insert_item(key_without_ts);
                //     }
                // }

                // Write block directly to mmap (ZERO-COPY from pre-built vecs!)
                

                {
                    let mut writer_guard = self.key_writer.lock();
                    match writer_guard.as_mut() {
                        | Some(writer) => {
                            // Build block from pre-built vecs - NO COPY!
                            let res = writer.write_block_direct(|mmap_slice| {
                                let builder =
                                    BlockBuilder::from_parts(mmap_slice, offsets, entries);
                                builder.finalize();
                            });

                            if res.is_ok() {
                                drop(writer_guard);
                                self.key_index.write().inc_block_count(1);
                                self.bytes_written.fetch_add(BLOCK_SIZE as u64, Relaxed);
                            }
                            res
                        },
                        | None => return Err(ReadOnly),
                    }
                }
            },
            | Value => {
                // Take pre-built vecs (ZERO intermediate copy!)
                let (offsets, entries) = {
                    let mut val_acc = self.current_val_entries.lock();
                    if val_acc.0.is_empty() {
                        return Ok(()); // Nothing to write
                    }
                    std::mem::take(&mut *val_acc)
                };

                // Write block directly to mmap (ZERO-COPY from pre-built vecs!)
                

                {
                    let mut writer_guard = self.val_writer.lock();
                    match writer_guard.as_mut() {
                        | Some(writer) => {
                            // Build block from pre-built vecs - NO COPY!
                            let res = writer.write_block_direct(|mmap_slice| {
                                let builder =
                                    BlockBuilder::from_parts(mmap_slice, offsets, entries);
                                builder.finalize();
                            });

                            if res.is_ok() {
                                drop(writer_guard);
                                self.val_block_count.fetch_add(1, Relaxed);
                                self.bytes_written.fetch_add(BLOCK_SIZE as u64, Relaxed);
                            }
                            res
                        },
                        | None => return Err(ReadOnly),
                    }
                }
            },
        }
    }

    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub(crate) fn close(&self) -> Result<(), SegmentError> {
        if self.key_writer.lock().is_none() {
            return Err(ReadOnly);
        }

        match self.flush() {
            | Ok(_) => {},
            | Err(e) => return Err(e),
        }

        // Rebuild index FIRST (deferred from hot write path)
        // Must happen before write_index() so the disk copy is populated
        if let Err(e) = self.rebuild_key_index() {
            return Err(e);
        }

        {
            let mut writer_guard = self.key_writer.lock();
            if let Some(writer) = writer_guard.as_mut() {
                // Index block count already synced during writes
                let block_count = writer.block_count();

                let key_index = self.key_index.write();
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
                    | Ok(_) => {},
                    | Err(e) => return Err(e),
                };
                match writer.close() {
                    | Ok(_) => {},
                    | Err(e) => return Err(e),
                };
            }
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
                    | Ok(_) => {},
                    | Err(e) => return Err(e),
                };
                match writer.close() {
                    | Ok(_) => {},
                    | Err(e) => return Err(e),
                };
            }
        }

        // Now mark as read-only
        *self.key_writer.lock() = None;
        *self.val_writer.lock() = None;

        Ok(())
    }

    /// Rebuild key index by scanning all keys in the segment.
    /// Call this after close() to populate the index that was deferred during
    /// writes.
    ///
    /// This scans all key blocks and inserts representative keys into the index
    /// for efficient lookups. Much faster than per-entry index updates during
    /// writes.
    pub fn rebuild_key_index(&self) -> Result<(), SegmentError> {
        use crate::utils::Deserializer; // Bring trait into scope

        // Get key map handle
        let key_map = match &self.key_handle {
            | Some(handle) => handle.clone(),
            | None => {
                let writer = self.key_writer.lock();
                match writer.as_ref() {
                    | Some(w) => w.map.clone(),
                    | None => return Err(CantCreateReader),
                }
            },
        };

        // Scan all key blocks and collect ALL keys for batch bloom rebuild
        // (bloom filter needs all keys for accurate lookups)
        let block_count = self.key_index.read().block_count() as usize;

        // Collect all (key, block_idx) pairs
        let mut key_block_pairs = Vec::with_capacity(block_count * 100); // Estimate ~100 keys/block

        for block_idx in 0..block_count {
            let block_offset = block_idx * BLOCK_SIZE;

            // Read block using Map::read_range
            let block_bytes = match key_map
                .read_range(block_offset..block_offset + BLOCK_SIZE, |slice| {
                    bytes::Bytes::copy_from_slice(slice)
                }) {
                | Ok(b) => b,
                | Err(e) => return Err(e),
            };

            let block = crate::block::ReadOnlyBlock::deserialize(block_bytes);

            // Collect ALL keys from this block with their block index
            for entry_idx in 0..block.num_entries() as usize {
                if let Some((_, entry_bytes)) = block.get(entry_idx) {
                    // entry_bytes format: [val_loc:10][ns:8][user_key][inv_ts:16]
                    // Strip val_loc (first 10 bytes) and timestamp (last 16 bytes)
                    if entry_bytes.len() > 26 {
                        let key_without_ts = entry_bytes[10..entry_bytes.len() - 16].to_vec();
                        key_block_pairs.push((key_without_ts, block_idx as u64));
                    }
                }
            }
        }

        // Batch rebuild bloom filter (single lock, optimized hashing)
        let mut key_index = self.key_index.write();
        key_index.rebuild_bloom_from_keys(key_block_pairs.iter().map(|(k, b)| (k.as_slice(), *b)));
        drop(key_index);

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
            0
        };

        let val_size = if let Some(ref handle) = self.val_handle {
            handle.len() as u64
        } else {
            0
        };

        let handle_size = key_size + val_size;
        if handle_size > 0 {
            handle_size
        } else {
            // Fallback to tracked bytes for open segments
            self.bytes_written.load(Relaxed)
        }
    }

    /// Quick bloom-filter check without creating a SegmentReader.
    /// Returns true if the segment *may* contain the key.
    #[inline]
    pub(crate) fn reader(&self) -> Result<SegmentReader, SegmentError> {
        if !self.is_read_only() {
            return Err(SegmentError::ReadOnly);
        }

        let key_handle = match self.key_handle.as_ref().ok_or(SegmentError::ReadOnly) {
            | Ok(h) => h.clone(),
            | Err(e) => return Err(e),
        };
        let val_handle = match self.val_handle.as_ref().ok_or(SegmentError::ReadOnly) {
            | Ok(h) => h.clone(),
            | Err(e) => return Err(e),
        };
        let key_index = self.key_index.clone();

        crate::segment_reader::SegmentReader::new(
            key_handle,
            val_handle,
            key_index,
        )
    }

    /// Point lookup. Checks bloom internally; touches at most one key block.
    /// Uses the per-segment value block cache for hot keys.
    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn get(&self, key_without_ts: &[u8]) -> Result<Option<Bytes>, SegmentError> {
        if !self.is_read_only() {
            return Err(SegmentError::ReadOnly);
        }

        let guard = self.key_index.read();
        if !guard.may_contain(key_without_ts) {
            return Ok(None);
        }
        let key_block_offset = match guard.get_block(key_without_ts) {
            | Some(v) => v,
            | None => return Ok(None),
        };
        drop(guard);

        let offset = key_block_offset as usize * BLOCK_SIZE;
        let key_handle = self.key_handle.as_ref().unwrap();
        let key_bytes = key_handle.read_bytes(offset..offset + BLOCK_SIZE)?;
        let key_block = ReadOnlyBlock::deserialize(key_bytes);

        for entry_index in 0..key_block.num_entries() as usize {
            let (flag, data) = match key_block.get(entry_index) {
                | Some(v) => v,
                | None => continue,
            };
            if data.len() < 10 {
                continue;
            }
            let value_block_num = u64::from_le_bytes(data[0..8].try_into().unwrap());
            let value_entry_index = u16::from_le_bytes(data[8..10].try_into().unwrap());
            let actual_key_data = &data[10..];

            let key_matches = match flag {
                | EntryFlag::Complete => {
                    actual_key_data.len() >= key_without_ts.len()
                        && &actual_key_data[..key_without_ts.len()] == key_without_ts
                },
                | EntryFlag::Start => {
                    match self.read_full_key(key_block_offset as usize, entry_index) {
                        | Ok(full_key_data) => {
                            if full_key_data.len() < 10 {
                                continue;
                            }
                            let full_actual_key = &full_key_data[10..];
                            full_actual_key.len() >= key_without_ts.len()
                                && &full_actual_key[..key_without_ts.len()] == key_without_ts
                        },
                        | Err(_) => continue,
                    }
                },
                | _ => continue,
            };

            if key_matches {
                return match self.read_value_from_block(
                    value_block_num as usize,
                    value_entry_index as usize,
                ) {
                    | Ok(v) => Ok(Some(v)),
                    | Err(e) => Err(e),
                };
            }
        }

        Ok(None)
    }

    /// Range scan. Creates an internal reader and returns an owned iterator.
    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn scan(
        &self,
        lower: std::ops::Bound<crate::keypair::KeyBytes>,
        upper: std::ops::Bound<crate::keypair::KeyBytes>,
    ) -> Result<OwnedSegmentIterator, SegmentError> {
        let reader = self.reader()?;
        Ok(OwnedSegmentIterator::new(reader, lower, upper))
    }

    // --- helpers for get() ---

    fn read_key_block_direct(&self, block_index: usize) -> Result<ReadOnlyBlock, SegmentError> {
        if block_index >= self.visible_key_blocks {
            return Err(ReadOutOfBounds);
        }
        let offset = block_index * BLOCK_SIZE;
        let key_handle = self.key_handle.as_ref().unwrap();
        let bytes = key_handle.read_bytes(offset..offset + BLOCK_SIZE)?;
        Ok(ReadOnlyBlock::deserialize(bytes))
    }

    fn read_val_block_cached(&self, block_index: usize) -> Result<ReadOnlyBlock, SegmentError> {
        if block_index >= self.visible_val_blocks {
            return Err(ReadOutOfBounds);
        }
        let cache = self.val_block_cache.lock();
        if let Some(cached) = cache.get(block_index) {
            Ok(cached)
        } else {
            drop(cache);
            let offset = block_index * BLOCK_SIZE;
            let val_handle = self.val_handle.as_ref().unwrap();
            let bytes = val_handle.read_bytes(offset..offset + BLOCK_SIZE)?;
            let block = ReadOnlyBlock::deserialize(bytes);
            self.val_block_cache.lock().insert(block_index, block.clone());
            Ok(block)
        }
    }

    fn read_full_key(&self, key_block_offset: usize, entry_index: usize) -> Result<Bytes, SegmentError> {
        let block = self.read_key_block_direct(key_block_offset)?;
        let (flag, data) = match block.get(entry_index).ok_or(MissingKey) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };
        self.read_multiblock_entry(flag, data, key_block_offset + 1)
    }

    fn read_multiblock_entry(
        &self,
        flag: EntryFlag,
        initial_data: &[u8],
        starting_block: usize,
    ) -> Result<Bytes, SegmentError> {
        use EntryFlag::*;
        match flag {
            | Complete => Ok(Bytes::copy_from_slice(initial_data)),
            | Start => {
                let mut buffer = BytesMut::with_capacity(initial_data.len() * 2);
                buffer.extend_from_slice(initial_data);
                let mut current_block_index = starting_block;
                let mut found_end = false;
                while current_block_index < self.visible_key_blocks && !found_end {
                    let next_block = match self.read_key_block_direct(current_block_index) {
                        | Ok(b) => b,
                        | Err(e) => return Err(e),
                    };
                    if next_block.num_entries() == 0 {
                        current_block_index += 1;
                        continue;
                    }
                    let (next_flag, next_data) = match next_block.get(0).ok_or(CorruptedBlock) {
                        | Ok(v) => v,
                        | Err(e) => return Err(e),
                    };
                    match next_flag {
                        | Middle => {
                            buffer.extend_from_slice(next_data);
                            current_block_index += 1;
                        },
                        | End => {
                            buffer.extend_from_slice(next_data);
                            found_end = true;
                        },
                        | _ => return Err(CorruptedBlock),
                    }
                }
                if !found_end {
                    return Err(CorruptedBlock);
                }
                Ok(buffer.freeze())
            },
            | Middle | End => Err(CorruptedBlock),
        }
    }

    fn read_value_from_block(
        &self,
        val_block_index: usize,
        entry_index: usize,
    ) -> Result<Bytes, SegmentError> {
        let block = self.read_val_block_cached(val_block_index)?;
        if entry_index >= block.num_entries() as usize {
            return Err(MissingKey);
        }
        match block.get_bytes(entry_index) {
            | Some((EntryFlag::Complete, data)) => Ok(data),
            | Some((EntryFlag::Start, data)) => {
                let mut buffer = BytesMut::with_capacity(data.len() * 2);
                buffer.extend_from_slice(&data);
                let mut current_block_index = val_block_index + 1;
                let mut found_end = false;
                if current_block_index >= self.visible_val_blocks {
                    return Err(CorruptedBlock);
                }
                while current_block_index < self.visible_val_blocks && !found_end {
                    let next_block = self.read_val_block_cached(current_block_index)?;
                    if next_block.num_entries() == 0 {
                        current_block_index += 1;
                        continue;
                    }
                    let (next_flag, next_data) = match next_block.get(0) {
                        | Some(v) => v,
                        | None => return Err(CorruptedBlock),
                    };
                    match next_flag {
                        | EntryFlag::Middle => {
                            buffer.extend_from_slice(next_data);
                            current_block_index += 1;
                        },
                        | EntryFlag::End => {
                            buffer.extend_from_slice(next_data);
                            found_end = true;
                        },
                        | _ => return Err(CorruptedBlock),
                    }
                }
                if !found_end {
                    return Err(CorruptedBlock);
                }
                Ok(buffer.freeze())
            },
            | Some((EntryFlag::Middle | EntryFlag::End, _)) => Err(CorruptedBlock),
            | None => Err(MissingKey),
        }
    }
}

impl Drop for Segment {
    fn drop(&mut self) {
        let res = self.close();
        if let Err(_e) = res &&
            !matches!(_e, ReadOnly)
        {
            // Error during segment close in Drop - metadata may not have been
            // flushed properly
        }
    }
}

#[cfg(test)]
#[allow(clippy::question_mark_used)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::undocumented_unsafe_blocks)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;
    use rand::{
        RngCore,
        prelude::SliceRandom,
        rng,
    };
    use tempfile::tempdir;

    use super::*;
    use crate::{
        hlc::{
            HLC,
            HybridLogicalClock,
        },
        segment_reader::SegmentReader,
        keypair::{
            DEFAULT_NS,
            KeyBytes,
            ValueBytes,
        },
        map::Map,
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

    // helper function to create serialized test key-value pair for segment
    // write/read
    fn create_serialized_kv(key: &str, value: &str, clock: &HybridLogicalClock) -> (Bytes, Bytes) {
        use crate::utils::Serializer;
        let (k, v) = create_kv(key, value, clock);
        (k.serialize(), v.serialize())
    }

    #[test]
    fn test_segment_basic_write() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();
        let clock = HybridLogicalClock::new();

        // Simple key-value pair (must be serialized)
        let (key, val) = create_serialized_kv("abc", "123", &clock);

        // Write to segment
        let result = segment.write(&key, &val);
        assert!(result.is_ok(), "Failed to write to segment: {:?}", result);
    }

    #[test]
    fn test_segment_multiple_writes() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();
        let clock = HybridLogicalClock::new();

        // Write multiple key-value pairs (must be serialized)
        for i in 0u32..10 {
            let key = format!("key_{}", i);
            let value = format!("value_{}", i * 10);
            let (k, v) = create_serialized_kv(&key, &value, &clock);

            let result = segment.write(&k, &v);
            assert!(result.is_ok(), "Failed to write entry {}: {:?}", i, result);
        }
    }

    #[test]
    fn test_segment_namespace_handling() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();
        let clock = HybridLogicalClock::new();

        // Write entries with different namespaces (must be serialized)
        for ns in &[1u64, 2u64, 1u64, 3u64, 2u64] {
            use crate::utils::Serializer;
            let key = KeyBytes::new(*ns, Bytes::from("testkey"), clock.time());
            let val = ValueBytes::new(*ns, Bytes::from("testvalue"));

            let result = segment.write(key.serialize().as_ref(), val.serialize().as_ref());
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
            segment.key_index.write().ns_offset_count() >= 3,
            "Key index should track at least 3 namespace changes"
        );
    }

    #[test]
    fn test_segment_large_entry() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();
        let clock = HybridLogicalClock::new();

        // Create a large value that spans multiple blocks
        let large_data_size = 8192; // 2 blocks worth of data
        let mut large_data = vec![0u8; large_data_size];
        rng().fill_bytes(&mut large_data);

        // Create serialized key-value pair
        use crate::utils::Serializer;
        let key = KeyBytes::new(DEFAULT_NS, Bytes::from("large_entry_key"), clock.time());
        let val = ValueBytes::new(DEFAULT_NS, Bytes::from(large_data));

        let result = segment.write(key.serialize().as_ref(), val.serialize().as_ref());
        assert!(result.is_ok(), "Failed to write large entry: {:?}", result);
    }

    #[test]
    fn test_segment_mixed_entry_sizes() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();
        let clock = HybridLogicalClock::new();

        let mut rng = rng();

        // Write entries with varying sizes (must be serialized)
        for i in 0u32..20 {
            // Generate random-sized values
            let size = match i % 4 {
                | 0 => 10,   // Small
                | 1 => 1000, // Medium
                | 2 => 4000, // Almost a block
                | 3 => 6000, // Multi-block
                | _ => unreachable!(),
            };

            let mut data = vec![0u8; size];
            rng.fill_bytes(&mut data);

            use crate::utils::Serializer;
            let key = KeyBytes::new(DEFAULT_NS, Bytes::from(format!("key_{}", i)), clock.time());
            let val = ValueBytes::new(DEFAULT_NS, Bytes::from(data));

            let result = segment.write(key.serialize().as_ref(), val.serialize().as_ref());
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

        // Get maps from writers since segment isn't read-only yet
        let km = segment.key_writer.lock().as_ref().unwrap().map.clone();
        let vm = segment.val_writer.lock().as_ref().unwrap().map.clone();

        let reader = SegmentReader::new(
            km,
            vm,
            segment.key_index.clone(),
        );
        assert!(reader.is_ok());
    }

    #[test]
    fn test_segment_with_many_small_entries() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();
        let clock = HybridLogicalClock::new();

        // Write many small entries to test block packing (must be serialized)
        for i in 0u32..1000 {
            let (key, val) =
                create_serialized_kv(&format!("key_{}", i), &format!("Value{}", i), &clock);

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
        let clock = HybridLogicalClock::new();

        // Test with lexicographically ordered keys (must be serialized)
        let mut data = vec![];
        for c in 'a'..='z' {
            let key = format!("key_{}", c);
            let value = format!("value_{}", c);
            let (k, v) = create_serialized_kv(&key, &value, &clock);
            data.push((k, v));
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
        let clock = HybridLogicalClock::new();

        // Create entries with random ordering (must be serialized)
        let mut data = vec![];
        for i in 0..100 {
            let key = format!("random_key_{:03}", i);
            let value = format!("random_value_{:03}", i);
            let (k, v) = create_serialized_kv(&key, &value, &clock);
            data.push((k, v));
        }

        // Shuffle the data
        let mut rng = rand::rng();
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
        let clock = HybridLogicalClock::new();

        // Write entries and check index growth (must be serialized)
        let _initial_key_blocks = segment.key_index.write().block_count();

        for i in 0u32..512 {
            let (key, val) =
                create_serialized_kv(&format!("key_{}", i), &format!("value_{}", i * 10), &clock);

            let result = segment.write(&key, &val);
            assert!(result.is_ok(), "Failed to write entry {}: {:?}", i, result);
        }

        segment.flush().expect("failed to flush segment");

        let final_key_blocks = segment.key_index.write().block_count();

        // The final block count depends on the serialized key size
        // Just verify that blocks were created
        assert!(
            final_key_blocks > 0,
            "there should be at least 1 block in the key index for 512 entries, found: {}",
            final_key_blocks
        );
    }

    #[test]
    fn test_val_block_count_persistence() {
        use crate::segment_builder::SegmentBuilder;

        let dir = tempdir().expect("failed to create temp dir");
        let builder = SegmentBuilder::new(dir.path().to_path_buf())
            .expect("failed to create segment builder");

        let segment_id = 100;
        let seed = 42;
        let segment = builder
            .new_segment(segment_id, seed, DEFAULT_SEGMENT_SIZE)
            .expect("failed to create segment");

        // Write enough data to span multiple value blocks (BLOCK_SIZE = 4096)
        // Each value is ~1KB, so we need at least 5 values to guarantee multiple blocks
        let num_entries: u32 = 20;
        let value_size = 1024;

        for i in 0..num_entries {
            let mut key = vec![0u8; 8]; // namespace
            key.extend_from_slice(&i.to_le_bytes());
            key.extend_from_slice(&[0u8; 16]); // timestamp

            let val = vec![i as u8; value_size];

            segment.write(&key, &val).expect("failed to write entry");
        }

        // Get the val_block_count before closing
        let before_close_count = segment
            .val_block_count
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            before_close_count > 0,
            "Expected at least one value block to be written"
        );

        // Close the segment (this will flush any partial blocks)
        segment.close().expect("failed to close segment");

        // Get the count after close to see if flush() wrote additional blocks
        let after_close_count = segment
            .val_block_count
            .load(std::sync::atomic::Ordering::Relaxed);

        // Drop the segment to release file locks
        drop(segment);

        // Reopen the segment
        let reopened = builder.open(segment_id).expect("failed to reopen segment");

        // Verify val_block_count was restored
        let restored_block_count = reopened
            .val_block_count
            .load(std::sync::atomic::Ordering::Relaxed);

        assert_eq!(
            restored_block_count, after_close_count,
            "val_block_count should be restored to {} after reopening, but got {}",
            after_close_count, restored_block_count
        );
    }
}
