use std::{
    ops::Bound,
    sync::Arc,
};
use bytes::{
    Bytes,
    BytesMut,
};
use crate::{
    block::{
        BLOCK_SIZE,
        EntryFlag,
    },
    errs::{
        SegmentError,
        SegmentError::{
            CorruptedBlock,
            MissingKey,
            ReadOutOfBounds,
        },
    },
    index::Index,
    map::Map,
    segment::{
        BlockType,
        BlockType::{
            Key,
            Value,
        },
        DEFAULT_SEGMENT_SIZE,
        Metadata,
    },
    segment_iterator::{
        RawSegmentScanIterator,
        SegmentScanIterator,
    },
    utils::Deserializer,
};

/// Simple ring-buffer block cache with linear search.
///
/// Used for per-segment and per-reader value block caching.
/// Tiny (N <= ~16) and lives behind a short-lived mutex, so
/// linear search is faster than HashMap overhead.
#[derive(Debug, Clone)]
pub(crate) struct BlockCache<const N: usize> {
    entries: [(usize, Option<crate::block::ReadOnlyBlock>); N],
    next: usize,
}

impl<const N: usize> BlockCache<N> {
    pub(crate) fn new() -> Self {
        Self {
            entries: std::array::from_fn(|_| (0, None)),
            next: 0,
        }
    }

    #[inline]
    pub(crate) fn get(&self, block_index: usize) -> Option<crate::block::ReadOnlyBlock> {
        for (idx, block) in &self.entries {
            if *idx == block_index {
                return block.clone();
            }
        }
        None
    }

    #[inline]
    pub(crate) fn insert(&mut self, block_index: usize, block: crate::block::ReadOnlyBlock) {
        self.entries[self.next] = (block_index, Some(block));
        self.next = (self.next + 1) % N;
    }
}

#[derive(Debug)]
pub(crate) struct SegmentReader {
    key_handle: Arc<Map>,
    val_handle: Arc<Map>,
    pub(crate) key_index: Arc<parking_lot::RwLock<Index>>,
    pub(crate) visible_key_blocks: usize,
    pub(crate) visible_val_blocks: usize,
    value_block_cache: parking_lot::Mutex<BlockCache<8>>,
}

impl SegmentReader {
    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub(crate) fn new(
        key_handle: Arc<Map>,
        val_handle: Arc<Map>,
        key_index: Arc<parking_lot::RwLock<Index>>,
    ) -> Result<Self, SegmentError> {
        let segment_size = key_handle.len();
        // Read metadata from the end of the file to get the authoritative block count.
        // Only trust it if the index + metadata fit exactly at the end (properly closed
        // segment). Otherwise fall back to heuristics for test segments / old files.
        let metadata_block_count = if segment_size >= 32 {
            match key_handle.read_range(segment_size - 32..segment_size, |slice| {
                Metadata::from(Bytes::copy_from_slice(slice))
            }) {
                | Ok(m) => {
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
        // Use the actual number of blocks that were written, not the file size
        let index_blocks = key_index.read().num_blocks() as usize;
        let num_blocks = segment_size.div_ceil(BLOCK_SIZE);
        // Determine visible blocks:
        // - If metadata is present and valid, trust its block_count
        // - Otherwise fall back to index_blocks or file size heuristics
        let visible_key_blocks = if let Some(block_count) = metadata_block_count {
            block_count
        } else if index_blocks > 0 {
            index_blocks
        } else if segment_size >= DEFAULT_SEGMENT_SIZE as usize {
            // Large pre-allocated file with 0 blocks in metadata - this is an empty segment
            0
        } else {
            // Small file or test segment - use file size
            num_blocks
        };
        let visible_val_blocks = val_handle.len() / BLOCK_SIZE;
        Ok(Self {
            key_handle,
            val_handle,
            key_index,
            visible_key_blocks,
            visible_val_blocks,
            value_block_cache: parking_lot::Mutex::new(BlockCache::new()),
        })
    }

    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub(crate) fn read_key_block(
        &self,
        block_index: usize,
    ) -> Result<crate::block::ReadOnlyBlock, SegmentError> {
        if block_index >= self.visible_key_blocks {
            return Err(ReadOutOfBounds);
        }
        // Read the requested block
        let block = match self.read_block_at(block_index, Key) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };
        Ok(block)
    }

    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    /// Helper method to read a potentially multi-block entry.
    ///
    /// # Arguments
    /// * `flag` - Entry flag from the initial block
    /// * `initial_data` - Data from the initial block entry
    /// * `starting_block` - Block index where the entry starts (for multi-block
    ///   entries, where to continue reading)
    ///
    /// # Returns
    /// The complete entry data, reassembled if it was split across blocks
    pub(crate) fn read_multiblock_entry(
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
                    let next_block = match self.read_key_block(current_block_index) {
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

    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub(crate) fn read_value(
        &self,
        val_block_index: usize,
        entry_index: usize,
    ) -> Result<Bytes, SegmentError> {
        // Check if the block index is within bounds
        if val_block_index >= self.visible_val_blocks {
            return Err(ReadOutOfBounds);
        }

        let block = {
            let cache = self.value_block_cache.lock();
            if let Some(cached_block) = cache.get(val_block_index) {
                cached_block
            } else {
                drop(cache);
                let block = match self.read_block_at(val_block_index, Value) {
                    | Ok(v) => v,
                    | Err(e) => return Err(e),
                };
                self.value_block_cache.lock().insert(val_block_index, block.clone());
                block
            }
        };

        // Check if the entry exists
        if entry_index >= block.num_entries() as usize {
            return Err(MissingKey);
        }
        // Handle different entry types
        match block.get_bytes(entry_index) {
            | Some((EntryFlag::Complete, data)) => {
                // Simple case - entire value is in this entry.
                // `data` is already a `Bytes` slice into the block — zero copy.
                Ok(data)
            },
            | Some((EntryFlag::Start, data)) => {
                // For multi-block values, we need to find the End flag
                let mut buffer = BytesMut::with_capacity(data.len() * 2);
                buffer.extend_from_slice(&data);
                let mut current_block_index = val_block_index + 1;
                let mut found_end = false;
                // Check if we have more blocks to read - if not, this is corrupted
                if current_block_index >= self.visible_val_blocks {
                    return Err(CorruptedBlock);
                }
                // Read subsequent blocks until we find the End flag
                while current_block_index < self.visible_val_blocks && !found_end {
                    let next_block = match self.read_block_at(current_block_index, Value) {
                        | Ok(v) => v,
                        | Err(e) => {
                            return Err(e);
                        },
                    };
                    if next_block.num_entries() == 0 {
                        current_block_index += 1;
                        continue;
                    }
                    let (next_flag, next_data) = match next_block.get(0) {
                        | Some(v) => v,
                        | None => {
                            return Err(CorruptedBlock);
                        },
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
                        | _ => {
                            return Err(CorruptedBlock);
                        },
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

    /// Returns a reference to the key map handle
    pub(crate) fn key_handle(&self) -> &Arc<Map> {
        &self.key_handle
    }
    /// Returns a reference to the value map handle
    pub(crate) fn val_handle(&self) -> &Arc<Map> {
        &self.val_handle
    }

    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    /// Internal method to read a single block directly from the map.
    fn read_block_at(
        &self,
        block_index: usize,
        block_type: BlockType,
    ) -> Result<crate::block::ReadOnlyBlock, SegmentError> {
        let offset = block_index * BLOCK_SIZE;
        // Use pre-computed visible block counts to avoid fstat on every read.
        let max_blocks = match block_type {
            | Key => self.visible_key_blocks,
            | Value => self.visible_val_blocks,
        };
        if block_index >= max_blocks {
            return Err(ReadOutOfBounds);
        }

        let handle = match block_type {
            | Key => &self.key_handle,
            | Value => &self.val_handle,
        };
        let bytes = match handle.read_bytes(offset..offset + BLOCK_SIZE) {
            | Ok(b) => b,
            | Err(e) => return Err(e),
        };
        let block = crate::block::ReadOnlyBlock::deserialize(bytes);
        Ok(block)
    }

    /// Create a new iterator to scan a range of keys in the segment.
    ///
    /// * `lower_bound` - The lower bound of the key range (inclusive if
    ///   Included, exclusive if Excluded)
    /// * `upper_bound` - The upper bound of the key range (inclusive if
    ///   Included, exclusive if Excluded)
    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub(crate) fn scan(self, lower_bound: Bound<&[u8]>, upper_bound: Bound<&[u8]>) -> SegmentScanIterator {
        // Determine starting block based on lower bound
        let start_block = match lower_bound {
            | Bound::Included(key) | Bound::Excluded(key) => {
                // Strip timestamp (last 16 bytes) before index lookup
                // Index hashes [ns:8][user_key] to map all versions to same block
                debug_assert!(
                    key.len() >= 24,
                    "Scan key too short: {} bytes. Keys must be serialized",
                    key.len()
                );
                let key_without_ts = &key[..key.len() - 16];
                // Use the index to find the block that would contain this key
                match self.key_index.read().get_block(key_without_ts) {
                    | Some(block_offset) => block_offset as usize,
                    | None => 0, // Start from the beginning if not found
                }
            },
            | Bound::Unbounded => 0, // Start from the beginning
        };
        SegmentScanIterator::new(self, (lower_bound, upper_bound), start_block)
    }
    /// Create a raw scan iterator for compaction (zero-copy, no
    /// deserialization).
    ///
    /// Same logic as `scan()` but returns `RawSegmentScanIterator` which yields
    /// `RawEntry` instead of `(KeyBytes, ValueBytes)`.
    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub(crate) fn scan_raw(
        self,
        lower_bound: Bound<&[u8]>,
        upper_bound: Bound<&[u8]>,
    ) -> RawSegmentScanIterator {
        let start_block = match lower_bound {
            | Bound::Included(key) | Bound::Excluded(key) => {
                debug_assert!(
                    key.len() >= 24,
                    "Scan key too short: {} bytes. Keys must be serialized",
                    key.len()
                );
                let key_without_ts = &key[..key.len() - 16];
                match self.key_index.read().get_block(key_without_ts) {
                    | Some(block_offset) => block_offset as usize,
                    | None => 0,
                }
            },
            | Bound::Unbounded => 0,
        };
        RawSegmentScanIterator::new(self, (lower_bound, upper_bound), start_block)
    }
}

#[cfg(test)]
#[allow(clippy::question_mark_used)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::undocumented_unsafe_blocks)]
mod tests {
    use std::sync::Arc;
    use tempfile::tempdir;
    use super::*;
    use crate::{
        block::{
            Block,
            EntryFlag,
        },
        map::Map,
    };
    // Helper function to create a temporary Map with specified size
    fn create_test_map(size: usize) -> (tempfile::TempDir, Arc<Map>) {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.map");
        // Initialize Map with the specified size
        let map = Arc::new(Map::new(file_path, size as u64).unwrap());
        (dir, map)
    }
    // Helper to prepare a Map with blocks
    fn prepare_blocks_map(num_blocks: usize) -> (tempfile::TempDir, Arc<Map>) {
        let (dir, map) = create_test_map(num_blocks * BLOCK_SIZE);
        // Fill with test blocks
        for i in 0..num_blocks {
            let mut block = Block::new();
            let data = vec![i as u8; 16]; // Use block index as data
            block.add_entry(&data, EntryFlag::Complete).unwrap();
            // Write block to map
            let offset = i * BLOCK_SIZE;
            let block_range = offset..(offset + BLOCK_SIZE);
            map.write_to_range(block_range, |slice| unsafe {
                block.finalize(slice.as_mut_ptr());
            })
            .unwrap();
        }
        (dir, map)
    }

    #[test]
    fn test_new_segment_reader() {
        let size = BLOCK_SIZE * 4;
        let (_dir, key_map) = create_test_map(size);
        let (_dir2, val_map) = create_test_map(size);
        let key_index = Index::new(1, 1234);
        let _val_index = Index::new(1, 1234);
        let reader = SegmentReader::new(
            key_map.clone(),
            val_map.clone(),
            Arc::new(parking_lot::RwLock::new(key_index)),
        );
        assert!(reader.is_ok());
        let _reader = reader.unwrap();
    }
    #[test]
    fn test_non_aligned_size_accepted() {
        // Segment files may not be exact multiples of BLOCK_SIZE due to
        // metadata/index appends or growth increments. The reader should
        // accept any size and compute visible blocks with div_ceil.
        let non_aligned_size = BLOCK_SIZE * 2 + 100; // Not a multiple of BLOCK_SIZE
        let (_dir, key_map) = create_test_map(non_aligned_size);
        let (_dir2, val_map) = create_test_map(non_aligned_size);
        let key_index = Index::new(1, 1234);
        let result = SegmentReader::new(
            key_map.clone(),
            val_map.clone(),
            Arc::new(parking_lot::RwLock::new(key_index)),
        );
        assert!(
            result.is_ok(),
            "SegmentReader should accept non-aligned sizes"
        );
        let _reader = result.unwrap();
    }
    #[test]
    fn test_read_key_block() {
        let (_, key_map) = prepare_blocks_map(4);
        let (_, val_map) = prepare_blocks_map(4);
        let key_index = Index::new(1, 1234);
        let _val_index = Index::new(1, 1234);
        let reader = SegmentReader::new(
            key_map.clone(),
            val_map.clone(),
            Arc::new(parking_lot::RwLock::new(key_index)),
        )
        .unwrap();
        // Read each block and verify contents
        for i in 0..4 {
            let block = reader.read_key_block(i).unwrap();
            let entry = block.get(0).unwrap();
            assert_eq!(entry.1, &vec![i as u8; 16]);
        }
    }
    #[test]
    fn test_read_block_out_of_bounds() {
        let (_, key_map) = prepare_blocks_map(2);
        let (_, val_map) = prepare_blocks_map(2);
        let key_index = Index::new(1, 1234);
        let _val_index = Index::new(1, 1234);
        let reader = SegmentReader::new(
            key_map.clone(),
            val_map.clone(),
            Arc::new(parking_lot::RwLock::new(key_index)),
        )
        .unwrap();
        let result = reader.read_key_block(2); // Only 2 blocks exist (0 and 1)
        assert!(result.is_err());
        assert!(matches!(result.err().unwrap(), ReadOutOfBounds));
    }
    #[test]
    fn test_read_block_random_access() {
        let (_, key_map) = prepare_blocks_map(8);
        let (_, val_map) = prepare_blocks_map(8);
        let key_index = Index::new(1, 1234);
        let _val_index = Index::new(1, 1234);
        let reader = SegmentReader::new(
            key_map.clone(),
            val_map.clone(),
            Arc::new(parking_lot::RwLock::new(key_index)),
        )
        .unwrap();
        // Access blocks in non-sequential order
        let indices = [3, 1, 5, 0, 7, 2];
        for &idx in &indices {
            let block = reader.read_key_block(idx).unwrap();
            assert_eq!(block.get(0).unwrap().1, &vec![idx as u8; 16]);
        }
    }
}
