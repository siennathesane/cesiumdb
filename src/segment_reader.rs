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
/// Simple fixed-size block cache using ring buffer
#[derive(Debug)]
struct BlockCache<const N: usize> {
    entries: [(usize, Option<crate::block::ReadOnlyBlock>); N],
    next: usize,
}
#[allow(dead_code)]
impl<const N: usize> BlockCache<N> {
    fn new() -> Self {
        Self {
            entries: core::array::from_fn(|_| (0, None)),
            next: 0,
        }
    }
    #[inline]
    fn get(&self, block_index: usize) -> Option<&crate::block::ReadOnlyBlock> {
        // Linear search - faster than HashMap for N <= 8
        for (idx, block) in &self.entries {
            if *idx == block_index {
                return block.as_ref();
            }
        }
        None
    }
    #[inline]
    fn insert(&mut self, block_index: usize, block: crate::block::ReadOnlyBlock) {
        self.entries[self.next] = (block_index, Some(block));
        self.next = (self.next + 1) % N;
    }
}
#[derive(Debug)]
pub struct SegmentReader {
    key_handle: Arc<Map>,
    val_handle: Arc<Map>,
    pub(crate) key_index: Arc<parking_lot::RwLock<Index>>,
    pub(crate) visible_key_blocks: usize,
    pub(crate) visible_val_blocks: usize,
}
impl SegmentReader {
    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn new(
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
        })
    }
    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn get(&self, key: &[u8]) -> Result<Option<Bytes>, SegmentError> {
        // Strip timestamp (last 16 bytes) before index lookup
        // Index hashes [ns:8][user_key] to map all versions to same block
        // Keys must be serialized: [ns:8][user_key][timestamp:16], minimum 24 bytes
        debug_assert!(
            key.len() >= 24,
            "Key too short: {} bytes. Keys must be serialized with KeyBytes::serialize()",
            key.len()
        );
        let key_without_ts = &key[..key.len() - 16];
        // First check the bloom filter - quick reject if not present
        if !self.key_index.read().may_contain(key_without_ts) {
            return Ok(None);
        }
        // Find which block might contain this key
        let key_block_offset = match self.key_index.read().get_block(key_without_ts) {
            | Some(v) => v,
            | None => {
                return Ok(None);
            },
        };
        // Read the key block
        let key_block = match self.read_key_block(key_block_offset as usize) {
            | Ok(v) => v,
            | Err(_e) => {
                return Err(MissingKey);
            },
        };
        // Search all entries in the block
        for entry_index in 0..key_block.num_entries() as usize {
            let (flag, data) = match key_block.get(entry_index) {
                | Some(v) => v,
                | None => continue,
            };
            // Extract metadata from key entry
            // Format: [value_block_num:u64][value_entry_index:u16][key_data]
            if data.len() < 10 {
                continue; // Invalid entry, skip
            }
            let value_block_num = u64::from_le_bytes(data[0..8].try_into().unwrap());
            let value_entry_index = u16::from_le_bytes(data[8..10].try_into().unwrap());
            let actual_key_data = &data[10..];
            // Handle based on entry flag
            let key_matches = match flag {
                | EntryFlag::Complete => {
                    // Simple comparison for complete entries
                    actual_key_data == key
                },
                | EntryFlag::Start => {
                    // For multi-block keys, read the full key and compare
                    match self.read_key(key_block_offset as usize, entry_index) {
                        | Ok(full_key_data) => {
                            // Strip metadata from full key data
                            if full_key_data.len() < 10 {
                                continue;
                            }
                            let full_actual_key = &full_key_data[10..];
                            full_actual_key == key
                        },
                        | Err(_e) => {
                            continue;
                        },
                    }
                },
                | _ => continue, // Skip middle or end entries
            };
            // If key matches, read the value using embedded metadata
            if key_matches {
                // Use the embedded value block and entry index for O(1) lookup
                return match self.read_value(value_block_num as usize, value_entry_index as usize) {
                    | Ok(v) => Ok(Some(v)),
                    | Err(e) => Err(e),
                };
            }
        }
        Ok(None)
    }
    /// Perform a point lookup that matches any version of the key.
    ///
    /// Unlike [`get`](Self::get) which requires an exact timestamp match,
    /// `get_latest` searches the indexed block for an entry whose prefix
    /// (namespace + user key) matches `key_without_ts`.  This is the fast
    /// path used by the DB read path because flushed/compacted segments
    /// already contain only one version per key.
    ///
    /// The bloom filter and block index are consulted using
    /// `key_without_ts`, so the lookup touches **at most one** key block.
    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn get_latest(&self, key_without_ts: &[u8]) -> Result<Option<Bytes>, SegmentError> {
        // Bloom filter negative check
        if !self.key_index.read().may_contain(key_without_ts) {
            return Ok(None);
        }
        // Block index lookup
        let key_block_offset = match self.key_index.read().get_block(key_without_ts) {
            | Some(v) => v,
            | None => return Ok(None),
        };
        let key_block = match self.read_key_block(key_block_offset as usize) {
            | Ok(v) => v,
            | Err(_e) => return Err(MissingKey),
        };
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
                    match self.read_key(key_block_offset as usize, entry_index) {
                        | Ok(full_key_data) => {
                            if full_key_data.len() < 10 {
                                continue;
                            }
                            let full_actual_key = &full_key_data[10..];
                            full_actual_key.len() >= key_without_ts.len()
                                && &full_actual_key[..key_without_ts.len()] == key_without_ts
                        },
                        | Err(_e) => continue,
                    }
                },
                | _ => continue,
            };
            if key_matches {
                return match self.read_value(
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
    /// Same as [`get_latest`](Self::get_latest) but skips the redundant bloom
    /// filter check — the caller already verified the key *may* be present.
    #[inline]
    pub fn get_latest_fast(&self, key_without_ts: &[u8]) -> Result<Option<Bytes>, SegmentError> {
        let key_block_offset = match self.key_index.read().get_block(key_without_ts) {
            | Some(v) => v,
            | None => return Ok(None),
        };
        let key_block = match self.read_key_block(key_block_offset as usize) {
            | Ok(v) => v,
            | Err(_e) => return Err(MissingKey),
        };
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
                    match self.read_key(key_block_offset as usize, entry_index) {
                        | Ok(full_key_data) => {
                            if full_key_data.len() < 10 {
                                continue;
                            }
                            let full_actual_key = &full_key_data[10..];
                            full_actual_key.len() >= key_without_ts.len()
                                && &full_actual_key[..key_without_ts.len()] == key_without_ts
                        },
                        | Err(_e) => continue,
                    }
                },
                | _ => continue,
            };
            if key_matches {
                return match self.read_value(
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
    fn read_key(&self, key_block_index: usize, entry_index: usize) -> Result<Bytes, SegmentError> {
        let block = match self.read_key_block(key_block_index) {
            | Ok(b) => b,
            | Err(e) => return Err(e),
        };
        let (flag, data) = match block.get(entry_index).ok_or(MissingKey) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };
        // Use helper to handle multi-block entries
        self.read_multiblock_entry(flag, data, key_block_index + 1)
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
        // Read the value block directly (no per-reader cache —
        // random-access workloads have near-zero hit rate anyway).
        let block = match self.read_block_at(val_block_index, Value) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
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
    /// Internal method to read a single block without caching
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
        // Directly create Bytes from mmap without intermediate copy
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
    /// Check if a key might be in this segment using the bloom filter.
    ///
    /// This is a fast negative check - if it returns false, the key is
    /// definitely not in the segment. If it returns true, the key might
    /// be in the segment (subject to false positive rate).
    ///
    /// The key should have the timestamp stripped (last 16 bytes removed).
    pub fn may_contain(&self, key_without_timestamp: &[u8]) -> bool {
        self.key_index.read().may_contain(key_without_timestamp)
    }
    /// Create a new iterator to scan a range of keys in the segment.
    ///
    /// * `lower_bound` - The lower bound of the key range (inclusive if
    ///   Included, exclusive if Excluded)
    /// * `upper_bound` - The upper bound of the key range (inclusive if
    ///   Included, exclusive if Excluded)
    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn scan(self, lower_bound: Bound<&[u8]>, upper_bound: Bound<&[u8]>) -> SegmentScanIterator {
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
    // Helper to create a serialized test key: [ns:8][user_key][timestamp:16]
    fn create_test_key(user_key: &[u8]) -> Vec<u8> {
        let mut key = vec![0u8; 8]; // namespace (8 bytes)
        key.extend_from_slice(user_key); // user key
        key.extend_from_slice(&[0u8; 16]); // timestamp (16 bytes)
        key
    }
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
    // helper function to prepare a segment with key-value data
    fn prepare_test_segment_for_get() -> (tempfile::TempDir, Arc<Map>, Arc<Map>, Index, Index) {
        let dir = tempdir().unwrap();
        // create key map
        let key_path = dir.path().join("key_segment");
        let key_map = Arc::new(Map::new(key_path, BLOCK_SIZE as u64 * 10).unwrap());
        // create value map
        let val_path = dir.path().join("val_segment");
        let val_map = Arc::new(Map::new(val_path, BLOCK_SIZE as u64 * 10).unwrap());
        // create indexes with a fixed seed for deterministic behavior
        let seed = 42i64;
        let key_index = Index::new(1, seed);
        let val_index = Index::new(2, seed);
        (dir, key_map, val_map, key_index, val_index)
    }
    // helper to write block to map
    fn write_block_to_mapfor_get(map: &Arc<Map>, offset: usize, block: &Block) {
        let range = offset..(offset + BLOCK_SIZE);
        map.write_to_range(range, |slice| unsafe {
            block.finalize(slice.as_mut_ptr());
        })
        .unwrap();
    }
    // helper to add metadata to key for tests
    // Format: [value_block_num:u64][value_entry_index:u16][key_data]
    fn add_key_metadata(key: &[u8], value_block: u64, value_entry: u16) -> Vec<u8> {
        let mut result = Vec::with_capacity(10 + key.len());
        result.extend_from_slice(&value_block.to_le_bytes());
        result.extend_from_slice(&value_entry.to_le_bytes());
        result.extend_from_slice(key);
        result
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
    fn test_read_block_caching() {
        let (_, key_map) = prepare_blocks_map(5);
        let (_, val_map) = prepare_blocks_map(5);
        let key_index = Index::new(1, 1234);
        let _val_index = Index::new(1, 1234);
        let reader = SegmentReader::new(
            key_map.clone(),
            val_map.clone(),
            Arc::new(parking_lot::RwLock::new(key_index)),
        )
        .unwrap();
        // First read
        let block0 = reader.read_key_block(0).unwrap();
        assert_eq!(block0.get(0).unwrap().1, &vec![0u8; 16]);
        // Next block should be in cache now
        let block1 = reader.read_key_block(1).unwrap();
        assert_eq!(block1.get(0).unwrap().1, &vec![1u8; 16]);
        // Skip to block 3, which should clear cache and rebuild
        let block3 = reader.read_key_block(3).unwrap();
        assert_eq!(block3.get(0).unwrap().1, &vec![3u8; 16]);
        // Block 4 should be in cache now
        let block4 = reader.read_key_block(4).unwrap();
        assert_eq!(block4.get(0).unwrap().1, &vec![4u8; 16]);
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
    #[test]
    fn test_get_with_complete_key() {
        let (_dir, key_map, val_map, mut key_index, mut val_index) = prepare_test_segment_for_get();
        // create test key and value (key must be serialized)
        let key = create_test_key(b"test_key");
        let value = b"test_value";
        // Value is at block 0, entry 0
        let key_with_metadata = add_key_metadata(&key, 0, 0);
        // create blocks
        let mut key_block = Block::new();
        key_block
            .add_entry(&key_with_metadata, EntryFlag::Complete)
            .unwrap();
        let mut val_block = Block::new();
        val_block.add_entry(value, EntryFlag::Complete).unwrap();
        // write blocks to maps
        write_block_to_mapfor_get(&key_map, 0, &key_block);
        write_block_to_mapfor_get(&val_map, 0, &val_block);
        // update indexes (index key without timestamp - strip last 16 bytes)
        let key_without_ts = &key[..key.len() - 16];
        key_index.insert_item(key_without_ts);
        key_index.inc_block_count(1);
        val_index.insert_item(key_without_ts);
        val_index.inc_block_count(1);
        // create segment reader
        let reader = SegmentReader::new(
            key_map.clone(),
            val_map.clone(),
            Arc::new(parking_lot::RwLock::new(key_index)),
        )
        .unwrap();
        // test get
        let result = reader.get(&key).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().as_ref(), value);
    }
    #[test]
    fn test_get_with_nonexistent_key() {
        let (_dir, key_map, val_map, key_index, _val_index) = prepare_test_segment_for_get();
        // create segment reader without any data
        let reader = SegmentReader::new(
            key_map.clone(),
            val_map.clone(),
            Arc::new(parking_lot::RwLock::new(key_index)),
        )
        .unwrap();
        // test get with non-existent key (must be serialized)
        let nonexistent_key = create_test_key(b"nonexistent_key");
        let result = reader.get(&nonexistent_key).unwrap();
        assert!(result.is_none());
    }
    #[test]
    fn test_get_with_multiblock_key() {
        let (_dir, key_map, val_map, mut key_index, mut val_index) = prepare_test_segment_for_get();
        // create a key (must be serialized)
        let key = create_test_key(b"test_key");
        // Create a moderately sized value that will span two blocks
        // Using a much smaller fixed size to avoid any size calculation issues
        let large_value = vec![b'v'; 1000]; // 1000 bytes is safe and still tests the functionality
        // Value starts at block 0, entry 0
        let key_with_metadata = add_key_metadata(&key, 0, 0);
        // create and prepare key block
        let mut key_block = Block::new();
        key_block
            .add_entry(&key_with_metadata, EntryFlag::Complete)
            .unwrap();
        // create and prepare value blocks (start, end)
        let mut val_block1 = Block::new();
        // First 500 bytes
        let val_part1 = &large_value[0..500];
        val_block1.add_entry(val_part1, EntryFlag::Start).unwrap();
        let mut val_block2 = Block::new();
        // Last 500 bytes
        let val_part2 = &large_value[500..];
        val_block2.add_entry(val_part2, EntryFlag::End).unwrap();
        // write blocks to maps
        write_block_to_mapfor_get(&key_map, 0, &key_block);
        write_block_to_mapfor_get(&val_map, 0, &val_block1);
        write_block_to_mapfor_get(&val_map, BLOCK_SIZE, &val_block2);
        // update indexes (index key without timestamp - strip last 16 bytes)
        let key_without_ts = &key[..key.len() - 16];
        key_index.insert_item(key_without_ts);
        key_index.inc_block_count(1);
        val_index.insert_item(key_without_ts);
        val_index.inc_block_count(2); // Value spans 2 blocks
        // create segment reader
        let reader = SegmentReader::new(
            key_map.clone(),
            val_map.clone(),
            Arc::new(parking_lot::RwLock::new(key_index)),
        )
        .unwrap();
        // test get with large value
        let result = reader.get(&key).unwrap();
        assert!(result.is_some());
        let retrieved_value = result.unwrap();
        assert_eq!(retrieved_value.len(), large_value.len());
        assert_eq!(retrieved_value.as_ref(), large_value.as_slice());
    }
    #[test]
    fn test_get_with_multiblock_key_and_value() {
        let (_dir, key_map, val_map, mut key_index, mut val_index) = prepare_test_segment_for_get();
        // Create a simple, small key for easier debugging (must be serialized)
        let key = create_test_key(b"multi_key");
        // Create smaller value that spans blocks but is easier to debug
        let value = vec![b'v'; 800];
        // Value starts at block 0, entry 0
        let key_with_metadata = add_key_metadata(&key, 0, 0);
        // Create key block - just use a simple single-block key
        let mut key_block = Block::new();
        key_block
            .add_entry(&key_with_metadata, EntryFlag::Complete)
            .unwrap();
        // Create value blocks with smaller chunks
        let mut val_block1 = Block::new();
        let val_part1 = &value[0..400];
        val_block1.add_entry(val_part1, EntryFlag::Start).unwrap();
        let mut val_block2 = Block::new();
        let val_part2 = &value[400..];
        val_block2.add_entry(val_part2, EntryFlag::End).unwrap();
        // Write blocks to maps
        write_block_to_mapfor_get(&key_map, 0, &key_block);
        write_block_to_mapfor_get(&val_map, 0, &val_block1);
        write_block_to_mapfor_get(&val_map, BLOCK_SIZE, &val_block2);
        // Update indexes - index key without timestamp - strip last 16 bytes
        let key_without_ts = &key[..key.len() - 16];
        key_index.insert_item(key_without_ts);
        key_index.inc_block_count(1);
        val_index.insert_item(key_without_ts);
        val_index.inc_block_count(2); // Value spans 2 blocks
        // Debug: Verify the key is in the index
        assert!(
            key_index.may_contain(key_without_ts),
            "Key should be in bloom filter"
        );
        let block_offset = key_index.get_block(key_without_ts);
        assert!(
            block_offset.is_some(),
            "Block for key should be found in index"
        );
        // Create segment reader
        let reader = SegmentReader::new(
            key_map.clone(),
            val_map.clone(),
            Arc::new(parking_lot::RwLock::new(key_index)),
        )
        .unwrap();
        // Test get with the key
        let result = reader.get(&key);
        assert!(result.is_ok(), "Result should be Ok");
        let result_value = result.unwrap();
        assert!(result_value.is_some(), "Value should be found");
        let retrieved_value = result_value.unwrap();
        assert_eq!(retrieved_value.len(), value.len());
        assert_eq!(retrieved_value.as_ref(), value.as_slice());
    }
    #[test]
    fn test_get_with_bloom_filter_no_match() {
        let (_dir, key_map, val_map, mut key_index, _val_index) = prepare_test_segment_for_get();
        // add some other keys to the index but not our test key
        key_index.insert_item(b"other_key1");
        key_index.insert_item(b"other_key2");
        // create segment reader
        let reader = SegmentReader::new(
            key_map.clone(),
            val_map.clone(),
            Arc::new(parking_lot::RwLock::new(key_index)),
        )
        .unwrap();
        // our test key should not be in the bloom filter (must be serialized)
        let test_key = create_test_key(b"test_key");
        let result = reader.get(&test_key).unwrap();
        assert!(result.is_none());
        // The early bloom filter check should prevent block lookups
    }
    #[test]
    fn test_get_with_empty_blocks() {
        let (_dir, key_map, val_map, mut key_index, _val_index) = prepare_test_segment_for_get();
        // create test key and add to bloom filter, but don't add blocks (must be
        // serialized)
        let key = create_test_key(b"test_key");
        let key_without_ts = &key[..key.len() - 16];
        key_index.insert_item(key_without_ts);
        // create segment reader
        let reader = SegmentReader::new(
            key_map.clone(),
            val_map.clone(),
            Arc::new(parking_lot::RwLock::new(key_index)),
        )
        .unwrap();
        // key is in bloom filter but not in any block
        let result = reader.get(&key).unwrap();
        assert!(result.is_none());
    }
    #[test]
    fn test_get_with_corrupted_blocks() {
        let (_dir, key_map, val_map, mut key_index, mut val_index) = prepare_test_segment_for_get();
        // create test key (must be serialized)
        let key = create_test_key(b"test_key");
        // Value starts at block 0, entry 0
        let key_with_metadata = add_key_metadata(&key, 0, 0);
        // create key block with valid entry
        let mut key_block = Block::new();
        key_block
            .add_entry(&key_with_metadata, EntryFlag::Complete)
            .unwrap();
        // add corrupted value block (with wrong flag sequence)
        let mut val_block = Block::new();
        val_block.add_entry(b"part1", EntryFlag::Start).unwrap();
        // write blocks
        write_block_to_mapfor_get(&key_map, 0, &key_block);
        write_block_to_mapfor_get(&val_map, 0, &val_block);
        // update indexes (index key without timestamp - strip last 16 bytes)
        let key_without_ts = &key[..key.len() - 16];
        key_index.insert_item(key_without_ts);
        key_index.inc_block_count(1);
        val_index.insert_item(key_without_ts);
        val_index.inc_block_count(1);
        // create segment reader
        let reader = SegmentReader::new(
            key_map.clone(),
            val_map.clone(),
            Arc::new(parking_lot::RwLock::new(key_index)),
        )
        .unwrap();
        // should return error due to corrupted blocks (missing End flag)
        let result = reader.get(&key);
        assert!(result.is_err());
        assert!(matches!(result.err().unwrap(), CorruptedBlock));
    }
    #[test]
    fn test_get_with_multiple_keys_in_block() {
        let (_dir, key_map, val_map, mut key_index, mut val_index) = prepare_test_segment_for_get();
        // create test keys (must be serialized)
        let key1 = create_test_key(b"test_key1");
        let key2 = create_test_key(b"test_key2");
        let key3 = create_test_key(b"test_key3");
        let val1 = b"value1";
        let val2 = b"value2";
        let val3 = b"value3";
        // Add metadata: key1 -> val_block 0, entry 0; key2 -> val_block 1, entry 0;
        // key3 -> val_block 2, entry 0
        let key1_with_metadata = add_key_metadata(&key1, 0, 0);
        let key2_with_metadata = add_key_metadata(&key2, 1, 0);
        let key3_with_metadata = add_key_metadata(&key3, 2, 0);
        // Create individual blocks for each key
        let mut key_block1 = Block::new();
        key_block1
            .add_entry(&key1_with_metadata, EntryFlag::Complete)
            .unwrap();
        let mut key_block2 = Block::new();
        key_block2
            .add_entry(&key2_with_metadata, EntryFlag::Complete)
            .unwrap();
        let mut key_block3 = Block::new();
        key_block3
            .add_entry(&key3_with_metadata, EntryFlag::Complete)
            .unwrap();
        // Create individual blocks for each value
        let mut val_block1 = Block::new();
        val_block1.add_entry(val1, EntryFlag::Complete).unwrap();
        let mut val_block2 = Block::new();
        val_block2.add_entry(val2, EntryFlag::Complete).unwrap();
        let mut val_block3 = Block::new();
        val_block3.add_entry(val3, EntryFlag::Complete).unwrap();
        // Write blocks to maps at offsets 0, 1, 2 to match what the index expects
        write_block_to_mapfor_get(&key_map, 0, &key_block1); // Block 0
        write_block_to_mapfor_get(&key_map, BLOCK_SIZE, &key_block2); // Block 1
        write_block_to_mapfor_get(&key_map, BLOCK_SIZE * 2, &key_block3); // Block 2
        write_block_to_mapfor_get(&val_map, 0, &val_block1); // Block 0
        write_block_to_mapfor_get(&val_map, BLOCK_SIZE, &val_block2); // Block 1
        write_block_to_mapfor_get(&val_map, BLOCK_SIZE * 2, &val_block3); // Block 2
        // Add to indexes - index keys without timestamp - strip last 16 bytes
        let key1_without_ts = &key1[..key1.len() - 16];
        let key2_without_ts = &key2[..key2.len() - 16];
        let key3_without_ts = &key3[..key3.len() - 16];
        key_index.insert_item(key1_without_ts); // Block 0
        key_index.inc_block_count(1);
        key_index.insert_item(key2_without_ts); // Block 1
        key_index.inc_block_count(1);
        key_index.insert_item(key3_without_ts); // Block 2
        key_index.inc_block_count(1);
        val_index.insert_item(key1_without_ts); // Block 0
        val_index.inc_block_count(1);
        val_index.insert_item(key2_without_ts); // Block 1
        val_index.inc_block_count(1);
        val_index.insert_item(key3_without_ts); // Block 2
        val_index.inc_block_count(1);
        // Create reader
        let reader = SegmentReader::new(
            key_map.clone(),
            val_map.clone(),
            Arc::new(parking_lot::RwLock::new(key_index)),
        )
        .unwrap();
        // Test get for each key
        let result1 = reader.get(&key1).unwrap();
        assert!(result1.is_some());
        assert_eq!(result1.unwrap().as_ref(), val1);
        let result2 = reader.get(&key2).unwrap();
        assert!(result2.is_some());
        assert_eq!(result2.unwrap().as_ref(), val2);
        let result3 = reader.get(&key3).unwrap();
        assert!(result3.is_some());
        assert_eq!(result3.unwrap().as_ref(), val3);
    }
    #[test]
    fn test_get_latest_finds_key_regardless_of_timestamp() {
        let (_dir, key_map, val_map, mut key_index, _val_index) = prepare_test_segment_for_get();
        // Create a serialized key with a specific timestamp
        let mut key = create_test_key(b"test_key");
        // Override timestamp to something non-zero
        let ts_bytes = 12345u128.to_le_bytes();
        let key_len = key.len();
        key[key_len - 16..].copy_from_slice(&ts_bytes);

        let value = b"test_value";
        let key_with_metadata = add_key_metadata(&key, 0, 0);

        let mut key_block = Block::new();
        key_block.add_entry(&key_with_metadata, EntryFlag::Complete).unwrap();
        let mut val_block = Block::new();
        val_block.add_entry(value, EntryFlag::Complete).unwrap();

        write_block_to_mapfor_get(&key_map, 0, &key_block);
        write_block_to_mapfor_get(&val_map, 0, &val_block);

        let key_without_ts = &key[..key.len() - 16];
        key_index.insert_item(key_without_ts);
        key_index.inc_block_count(1);

        let reader = SegmentReader::new(
            key_map.clone(),
            val_map.clone(),
            Arc::new(parking_lot::RwLock::new(key_index)),
        )
        .unwrap();

        // get() with the exact key (including timestamp) should find it
        let result = reader.get(&key).unwrap();
        assert!(result.is_some());

        // get() with a different timestamp should NOT find it
        let mut key2 = key.clone();
        let key2_len = key2.len();
        key2[key2_len - 16..].copy_from_slice(&99999u128.to_le_bytes());
        let result2 = reader.get(&key2).unwrap();
        assert!(result2.is_none());

        // get_latest() with key_without_ts SHOULD find it
        let result3 = reader.get_latest(key_without_ts).unwrap();
        assert!(result3.is_some());
        assert_eq!(result3.unwrap().as_ref(), value);

        // get_latest() with a non-existent key should return None
        let nonexistent = create_test_key(b"nonexistent_key");
        let nonexistent_without_ts = &nonexistent[..nonexistent.len() - 16];
        let result4 = reader.get_latest(nonexistent_without_ts).unwrap();
        assert!(result4.is_none());
    }
}
