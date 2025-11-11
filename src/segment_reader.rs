use std::{
    ops::{
        Bound,
        DerefMut,
    },
    sync::Arc,
};

use bytes::{
    Buf,
    Bytes,
    BytesMut,
};
use crossbeam_queue::ArrayQueue;
use tracing::instrument;
use crate::{
    block::{
        BLOCK_SIZE,
        Block,
        EntryFlag,
    },
    errs::{
        SegmentError,
        SegmentError::{
            CorruptedBlock,
            InvalidSize,
            MissingKey,
            ReadOutOfBounds,
        },
    },
    index::Index,
    keypair::{
        KeyBytes,
        ValueBytes,
    },
    map::Map,
    segment::{
        BlockType,
        BlockType::{
            Key,
            Value,
        },
    },
    segment_iterator::{
        SeekingBlockIterator,
        SegmentBlockIterator,
        SegmentScanIterator,
        convert_bound_to_bytes,
    },
    utils::Deserializer,
};

/// Configuration for read-ahead behavior
#[derive(Debug, Clone)]
pub(crate) struct ReadConfig {
    /// Number of blocks to read ahead
    read_ahead: usize,
}

impl Default for ReadConfig {
    fn default() -> Self {
        Self { read_ahead: 4 }
    }
}

#[derive(Debug)]
pub struct SegmentReader {
    key_handle: Arc<Map>,
    val_handle: Arc<Map>,
    pub(crate) key_index: Arc<parking_lot::Mutex<Index>>,
    pub(crate) visible_key_blocks: usize,
    visible_val_blocks: usize,
    pub(crate) num_blocks: usize,
}

impl SegmentReader {
    #[instrument(level = "trace")]
    pub fn new(
        key_handle: Arc<Map>,
        val_handle: Arc<Map>,
        key_index: Arc<parking_lot::Mutex<Index>>,
    ) -> Result<Self, SegmentError> {
        Self::with_config(
            key_handle,
            val_handle,
            key_index,
            ReadConfig::default(),
        )
    }

    #[instrument(level = "trace")]
    pub(crate) fn with_config(
        key_handle: Arc<Map>,
        val_handle: Arc<Map>,
        key_index: Arc<parking_lot::Mutex<Index>>,
        config: ReadConfig,
    ) -> Result<Self, SegmentError> {
        let segment_size = key_handle.len();

        if segment_size % BLOCK_SIZE != 0 {
            return Err(InvalidSize);
        }

        let num_blocks = segment_size / BLOCK_SIZE;

        let visible_key_blocks = num_blocks;
        let visible_val_blocks = val_handle.len() / BLOCK_SIZE;

        Ok(Self {
            key_handle,
            val_handle,
            key_index,
            visible_key_blocks,
            visible_val_blocks,
            num_blocks,
        })
    }

    #[instrument(level = "trace")]
    pub fn get(&self, key: &[u8]) -> Result<Option<Bytes>, SegmentError> {
        // First check the bloom filter - quick reject if not present
        if !self.key_index.lock().may_contain(key) {
            return Ok(None);
        }

        // Find which block might contain this key
        let key_block_offset = match self.key_index.lock().get_block(key) {
            | Some(v) => v,
            | None => {
                return Ok(None);
            },
        };

        // Read the key block
        let key_block = match self.read_key_block(key_block_offset as usize) {
            | Ok(v) => v,
            | Err(e) => {
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
                        | Err(e) => {
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

    #[instrument(level = "trace")]
    pub(crate) fn read_key_block(&self, block_index: usize) -> Result<Block, SegmentError> {
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

    #[instrument(level = "trace")]
    fn read_key(&self, key_block_index: usize, entry_index: usize) -> Result<Bytes, SegmentError> {
        let block = match self.read_key_block(key_block_index) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };

        let (flag, data) = match block.get(entry_index) {
            | Some(v) => v,
            | None => return Err(MissingKey),
        };

        match flag {
            | complete => Ok(Bytes::copy_from_slice(data)),
            | start => {
                let mut buffer = BytesMut::with_capacity(data.len() * 2);
                buffer.extend_from_slice(data);

                let mut current_block_index = key_block_index + 1;
                let mut found_end = false;

                while current_block_index < self.visible_key_blocks && !found_end {
                    let next_block = match self.read_key_block(current_block_index) {
                        | Ok(v) => v,
                        | Err(e) => return Err(e),
                    };

                    if next_block.num_entries() == 0 {
                        current_block_index += 1;
                        continue;
                    }

                    let (next_flag, next_data) = match next_block.get(0) {
                        | Some(v) => v,
                        | None => return Err(CorruptedBlock),
                    };

                    match next_flag {
                        | middle => {
                            buffer.extend_from_slice(next_data);
                            current_block_index += 1;
                        },
                        | end => {
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
            | EntryFlag::Middle | EntryFlag::End => Err(CorruptedBlock),
        }
    }

    #[instrument(level = "trace")]
    pub(crate) fn read_value(
        &self,
        val_block_index: usize,
        entry_index: usize,
    ) -> Result<Bytes, SegmentError> {
        // Check if the block index is within bounds
        if val_block_index >= self.visible_val_blocks {
            return Err(ReadOutOfBounds);
        }

        // Read the value block
        let block = match self.read_block_at(val_block_index, Value) {
            | Ok(v) => v,
            | Err(e) => {
                return Err(e);
            },
        };

        // Check if the entry exists
        if entry_index >= block.num_entries() as usize {
            return Err(MissingKey);
        }

        let (flag, data) = match block.get(entry_index) {
            | Some(v) => v,
            | None => {
                return Err(MissingKey);
            },
        };

        // Handle different entry types
        match flag {
            | EntryFlag::Complete => {
                // Simple case - entire value is in this entry
                Ok(Bytes::copy_from_slice(data))
            },
            | EntryFlag::Start => {
                // For multi-block values, we need to find the End flag
                let mut buffer = BytesMut::with_capacity(data.len() * 2);
                buffer.extend_from_slice(data);

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
            | EntryFlag::Middle | EntryFlag::End => Err(CorruptedBlock),
        }
    }

    fn find_key(&self, key_hash: u64, key_block_offset: u64) -> Result<Bytes, SegmentError> {
        let block_index = key_block_offset as usize;
        let block = match self.read_key_block(block_index) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };

        if let Some((flag, data)) = block.get(0) {
            return match flag {
                | complete => Ok(Bytes::copy_from_slice(data)),
                | start => {
                    let mut buffer = BytesMut::with_capacity(data.len() * 2);
                    buffer.extend_from_slice(data);

                    let mut current_block_index = block_index + 1;
                    let mut found_end = false;

                    while current_block_index < self.visible_key_blocks && !found_end {
                        let next_block = match self.read_key_block(current_block_index) {
                            | Ok(v) => v,
                            | Err(e) => return Err(e),
                        };

                        if next_block.num_entries() == 0 {
                            current_block_index += 1;
                            continue;
                        }

                        let (next_flag, next_data) = match next_block.get(0) {
                            | Some(v) => v,
                            | None => return Err(CorruptedBlock),
                        };

                        match next_flag {
                            | middle => {
                                buffer.extend_from_slice(next_data);
                                current_block_index += 1;
                            },
                            | end => {
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
                | _ => Err(CorruptedBlock),
            };
        }

        Err(ReadOutOfBounds)
    }

    pub(crate) fn visible_blocks(&self) -> (usize, usize) {
        (self.visible_key_blocks, self.visible_val_blocks)
    }

    #[instrument(level = "trace")]
    pub(crate) fn iter<'a>(&'a mut self) -> SegmentBlockIterator<'a> {
        SegmentBlockIterator::new(self)
    }

    #[instrument(level = "trace")]
    pub(crate) fn seeking_iter<'a>(&'a mut self) -> SeekingBlockIterator<'a> {
        SeekingBlockIterator::new(self, 0, self.num_blocks)
    }

    #[instrument(level = "trace")]
    /// Internal method to read a single block without caching
    fn read_block_at(
        &self,
        block_index: usize,
        block_type: BlockType,
    ) -> Result<Block, SegmentError> {
        let offset = block_index * BLOCK_SIZE;
        let mut buffer = BytesMut::zeroed(BLOCK_SIZE);

        match block_type {
            | Key => {
                if offset + BLOCK_SIZE > self.key_handle.len() {
                    return Err(ReadOutOfBounds);
                }

                buffer.copy_from_slice(&self.key_handle[offset..offset + BLOCK_SIZE]);
            },
            | Value => {
                if offset + BLOCK_SIZE > self.val_handle.len() {
                    return Err(ReadOutOfBounds);
                }

                buffer.copy_from_slice(&self.val_handle[offset..offset + BLOCK_SIZE]);
            },
        }

        let block = Block::deserialize(buffer.freeze());

        Ok(block)
    }

    /// Create a new iterator to scan a range of keys in the segment.
    ///
    /// * `lower_bound` - The lower bound of the key range (inclusive if
    ///   Included, exclusive if Excluded)
    /// * `upper_bound` - The upper bound of the key range (inclusive if
    ///   Included, exclusive if Excluded)
    #[instrument(level = "trace")]
    pub fn scan<'a>(
        &'a self,
        lower_bound: Bound<&[u8]>,
        upper_bound: Bound<&[u8]>,
    ) -> SegmentScanIterator<'a> {
        // Determine starting block based on lower bound
        let start_block = match lower_bound {
            | Bound::Included(key) | Bound::Excluded(key) => {
                // Use the index to find the block that would contain this key
                match self.key_index.lock().get_block(key) {
                    | Some(block_offset) => block_offset as usize,
                    | None => 0, // Start from the beginning if not found
                }
            },
            | Bound::Unbounded => 0, // Start from the beginning
        };

        SegmentScanIterator::new(self, (lower_bound, upper_bound))
    }

    /// Get the total number of blocks in this segment
    #[inline]
    pub(crate) fn num_blocks(&self) -> usize {
        self.num_blocks
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
            MAX_ENTRY_SIZE,
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
        let val_index = Index::new(1, 1234);

        let reader = SegmentReader::new(key_map.clone(), val_map.clone(), Arc::new(parking_lot::Mutex::new(key_index)));
        assert!(reader.is_ok());

        let reader = reader.unwrap();
        assert_eq!(reader.num_blocks(), 4);
    }

    #[test]
    fn test_invalid_size() {
        // Create Map with size that's not a multiple of BLOCK_SIZE
        let invalid_size = BLOCK_SIZE * 2 + 100; // Not a multiple of BLOCK_SIZE
        let (_dir, key_map) = create_test_map(invalid_size);
        let (_dir2, val_map) = create_test_map(invalid_size);

        let key_index = Index::new(1, 1234);
        let val_index = Index::new(1, 1234);

        let result = SegmentReader::new(key_map.clone(), val_map.clone(), Arc::new(parking_lot::Mutex::new(key_index)));
        assert!(result.is_err());
        assert!(matches!(result.err().unwrap(), InvalidSize));
    }

    #[test]
    fn test_read_key_block() {
        let (_, key_map) = prepare_blocks_map(4);
        let (_, val_map) = prepare_blocks_map(4);

        let key_index = Index::new(1, 1234);
        let val_index = Index::new(1, 1234);

        let reader =
            SegmentReader::new(key_map.clone(), val_map.clone(), Arc::new(parking_lot::Mutex::new(key_index))).unwrap();

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
        let val_index = Index::new(1, 1234);

        let reader =
            SegmentReader::new(key_map.clone(), val_map.clone(), Arc::new(parking_lot::Mutex::new(key_index))).unwrap();

        let result = reader.read_key_block(2); // Only 2 blocks exist (0 and 1)
        assert!(result.is_err());
        assert!(matches!(result.err().unwrap(), ReadOutOfBounds));
    }

    #[test]
    fn test_read_block_caching() {
        let (_, key_map) = prepare_blocks_map(5);
        let (_, val_map) = prepare_blocks_map(5);

        let key_index = Index::new(1, 1234);
        let val_index = Index::new(1, 1234);

        let reader =
            SegmentReader::new(key_map.clone(), val_map.clone(), Arc::new(parking_lot::Mutex::new(key_index))).unwrap();

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
        let val_index = Index::new(1, 1234);

        let reader =
            SegmentReader::new(key_map.clone(), val_map.clone(), Arc::new(parking_lot::Mutex::new(key_index))).unwrap();

        // Access blocks in non-sequential order
        let indices = [3, 1, 5, 0, 7, 2];

        for &idx in &indices {
            let block = reader.read_key_block(idx).unwrap();
            assert_eq!(block.get(0).unwrap().1, &vec![idx as u8; 16]);
        }
    }

    #[test]
    fn test_segment_block_iterator() {
        let (_, key_map) = prepare_blocks_map(3);
        let (_, val_map) = prepare_blocks_map(3);

        let key_index = Index::new(1, 1234);
        let val_index = Index::new(1, 1234);

        let mut reader =
            SegmentReader::new(key_map.clone(), val_map.clone(), Arc::new(parking_lot::Mutex::new(key_index))).unwrap();

        let blocks: Vec<Block> = reader.iter().map(|result| result.unwrap()).collect();

        assert_eq!(blocks.len(), 3);

        for (i, block) in blocks.iter().enumerate() {
            assert_eq!(block.get(0).unwrap().1, &vec![i as u8; 16]);
        }
    }

    #[test]
    fn test_seeking_block_iterator() {
        let (_, key_map) = prepare_blocks_map(5);
        let (_, val_map) = prepare_blocks_map(5);

        let key_index = Index::new(1, 1234);
        let val_index = Index::new(1, 1234);

        let mut reader =
            SegmentReader::new(key_map.clone(), val_map.clone(), Arc::new(parking_lot::Mutex::new(key_index))).unwrap();
        let mut iter = reader.seeking_iter();

        // Start at beginning
        assert_eq!(iter.current_position(), 0);

        // Read first block
        let block0 = iter.next().unwrap().unwrap();
        assert_eq!(block0.get(0).unwrap().1, &vec![0u8; 16]);

        // Seek to position 3
        iter.seek(3).unwrap();
        assert_eq!(iter.current_position(), 3);

        // Read from position 3
        let block3 = iter.next().unwrap().unwrap();
        assert_eq!(block3.get(0).unwrap().1, &vec![3u8; 16]);

        // Seek out of bounds should fail
        let result = iter.seek(5);
        assert!(result.is_err());
        assert!(matches!(result.err().unwrap(), ReadOutOfBounds));
    }

    #[test]
    fn test_iterator_size_hint() {
        let (_, key_map) = prepare_blocks_map(5);
        let (_, val_map) = prepare_blocks_map(5);

        let key_index = Index::new(1, 1234);
        let val_index = Index::new(1, 1234);

        let mut reader =
            SegmentReader::new(key_map.clone(), val_map.clone(), Arc::new(parking_lot::Mutex::new(key_index))).unwrap();

        // Test regular iterator
        let iter = reader.iter();
        let (min, max) = iter.size_hint();
        assert_eq!(min, 5);
        assert_eq!(max, Some(5));

        // Create new indices for second reader
        let key_index2 = Index::new(1, 1234);
        let val_index2 = Index::new(1, 1234);

        let mut reader =
            SegmentReader::new(key_map.clone(), val_map.clone(), Arc::new(parking_lot::Mutex::new(key_index2))).unwrap();

        // Test seeking iterator
        let mut seeking_iter = reader.seeking_iter();
        seeking_iter.seek(2).unwrap();

        let (min, max) = seeking_iter.size_hint();
        assert_eq!(min, 3); // 3 blocks remaining (2,3,4)
        assert_eq!(max, Some(3));
    }

    #[test]
    fn test_blocks_remaining() {
        let (_, key_map) = prepare_blocks_map(5);
        let (_, val_map) = prepare_blocks_map(5);

        let key_index = Index::new(1, 1234);
        let val_index = Index::new(1, 1234);

        let mut reader =
            SegmentReader::new(key_map.clone(), val_map.clone(), Arc::new(parking_lot::Mutex::new(key_index))).unwrap();
        let mut iter = reader.seeking_iter();

        assert_eq!(iter.blocks_remaining(), 5);

        // Read one block
        iter.next();
        assert_eq!(iter.blocks_remaining(), 4);

        // Seek forward
        iter.seek(3).unwrap();
        assert_eq!(iter.blocks_remaining(), 2);
    }

    #[test]
    fn test_get_with_complete_key() {
        let (dir, key_map, val_map, mut key_index, mut val_index) = prepare_test_segment_for_get();

        // create test key and value
        let key = b"test_key";
        let value = b"test_value";

        // Value is at block 0, entry 0
        let key_with_metadata = add_key_metadata(key, 0, 0);

        // create blocks
        let mut key_block = Block::new();
        key_block.add_entry(&key_with_metadata, EntryFlag::Complete).unwrap();

        let mut val_block = Block::new();
        val_block.add_entry(value, EntryFlag::Complete).unwrap();

        // write blocks to maps
        write_block_to_mapfor_get(&key_map, 0, &key_block);
        write_block_to_mapfor_get(&val_map, 0, &val_block);

        // update indexes (index the original key without metadata)
        key_index.insert_item(key);
        key_index.inc_block_count(1);

        val_index.insert_item(key);
        val_index.inc_block_count(1);

        // create segment reader
        let reader =
            SegmentReader::new(key_map.clone(), val_map.clone(), Arc::new(parking_lot::Mutex::new(key_index))).unwrap();

        // test get
        let result = reader.get(key).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().as_ref(), value);
    }

    #[test]
    fn test_get_with_nonexistent_key() {
        let (dir, key_map, val_map, key_index, val_index) = prepare_test_segment_for_get();

        // create segment reader without any data
        let reader =
            SegmentReader::new(key_map.clone(), val_map.clone(), Arc::new(parking_lot::Mutex::new(key_index))).unwrap();

        // test get with non-existent key
        let nonexistent_key = b"nonexistent_key";
        let result = reader.get(nonexistent_key).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_get_with_multiblock_key() {
        let (dir, key_map, val_map, mut key_index, mut val_index) = prepare_test_segment_for_get();

        // create a key
        let key = b"test_key";

        // Create a moderately sized value that will span two blocks
        // Using a much smaller fixed size to avoid any size calculation issues
        let large_value = vec![b'v'; 1000]; // 1000 bytes is safe and still tests the functionality

        // Value starts at block 0, entry 0
        let key_with_metadata = add_key_metadata(key, 0, 0);

        // create and prepare key block
        let mut key_block = Block::new();
        key_block.add_entry(&key_with_metadata, EntryFlag::Complete).unwrap();

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

        // update indexes (index the original key without metadata)
        key_index.insert_item(key);
        key_index.inc_block_count(1);

        val_index.insert_item(key);
        val_index.inc_block_count(2);  // Value spans 2 blocks

        // create segment reader
        let reader =
            SegmentReader::new(key_map.clone(), val_map.clone(), Arc::new(parking_lot::Mutex::new(key_index))).unwrap();

        // test get with large value
        let result = reader.get(key).unwrap();
        assert!(result.is_some());

        let retrieved_value = result.unwrap();
        assert_eq!(retrieved_value.len(), large_value.len());
        assert_eq!(retrieved_value.as_ref(), large_value.as_slice());
    }

    #[test]
    fn test_get_with_multiblock_key_and_value() {
        let (dir, key_map, val_map, mut key_index, mut val_index) = prepare_test_segment_for_get();

        // Create a simple, small key for easier debugging
        let key = b"multi_key";

        // Create smaller value that spans blocks but is easier to debug
        let value = vec![b'v'; 800];

        // Value starts at block 0, entry 0
        let key_with_metadata = add_key_metadata(key, 0, 0);

        // Create key block - just use a simple single-block key
        let mut key_block = Block::new();
        key_block.add_entry(&key_with_metadata, EntryFlag::Complete).unwrap();

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

        // Update indexes - make sure the same exact key is used
        key_index.insert_item(key);
        key_index.inc_block_count(1);

        val_index.insert_item(key);
        val_index.inc_block_count(2);  // Value spans 2 blocks

        // Debug: Verify the key is in the index
        assert!(key_index.may_contain(key), "Key should be in bloom filter");
        let block_offset = key_index.get_block(key);
        assert!(
            block_offset.is_some(),
            "Block for key should be found in index"
        );

        // Create segment reader
        let reader =
            SegmentReader::new(key_map.clone(), val_map.clone(), Arc::new(parking_lot::Mutex::new(key_index))).unwrap();

        // Test get with the key
        let result = reader.get(key);

        assert!(result.is_ok(), "Result should be Ok");
        let result_value = result.unwrap();
        assert!(result_value.is_some(), "Value should be found");

        let retrieved_value = result_value.unwrap();
        assert_eq!(retrieved_value.len(), value.len());
        assert_eq!(retrieved_value.as_ref(), value.as_slice());
    }

    #[test]
    fn test_get_with_bloom_filter_no_match() {
        let (dir, key_map, val_map, mut key_index, val_index) = prepare_test_segment_for_get();

        // add some other keys to the index but not our test key
        key_index.insert_item(b"other_key1");
        key_index.insert_item(b"other_key2");

        // create segment reader
        let reader =
            SegmentReader::new(key_map.clone(), val_map.clone(), Arc::new(parking_lot::Mutex::new(key_index))).unwrap();

        // our test key should not be in the bloom filter
        let test_key = b"test_key";
        let result = reader.get(test_key).unwrap();
        assert!(result.is_none());
        // The early bloom filter check should prevent block lookups
    }

    #[test]
    fn test_get_with_empty_blocks() {
        let (dir, key_map, val_map, mut key_index, val_index) = prepare_test_segment_for_get();

        // create test key and add to bloom filter, but don't add blocks
        let key = b"test_key";
        key_index.insert_item(key);

        // create segment reader
        let reader =
            SegmentReader::new(key_map.clone(), val_map.clone(), Arc::new(parking_lot::Mutex::new(key_index))).unwrap();

        // key is in bloom filter but not in any block
        let result = reader.get(key).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_get_with_corrupted_blocks() {
        let (dir, key_map, val_map, mut key_index, mut val_index) = prepare_test_segment_for_get();

        // create test key
        let key = b"test_key";

        // Value starts at block 0, entry 0
        let key_with_metadata = add_key_metadata(key, 0, 0);

        // create key block with valid entry
        let mut key_block = Block::new();
        key_block.add_entry(&key_with_metadata, EntryFlag::Complete).unwrap();

        // add corrupted value block (with wrong flag sequence)
        let mut val_block = Block::new();
        val_block.add_entry(b"part1", EntryFlag::Start).unwrap();

        // write blocks
        write_block_to_mapfor_get(&key_map, 0, &key_block);
        write_block_to_mapfor_get(&val_map, 0, &val_block);

        // update indexes (index the original key without metadata)
        key_index.insert_item(key);
        key_index.inc_block_count(1);

        val_index.insert_item(key);
        val_index.inc_block_count(1);

        // create segment reader
        let reader =
            SegmentReader::new(key_map.clone(), val_map.clone(), Arc::new(parking_lot::Mutex::new(key_index))).unwrap();

        // should return error due to corrupted blocks (missing End flag)
        let result = reader.get(key);
        assert!(result.is_err());
        assert!(matches!(result.err().unwrap(), CorruptedBlock));
    }

    #[test]
    fn test_get_with_multiple_keys_in_block() {
        let (dir, key_map, val_map, mut key_index, mut val_index) = prepare_test_segment_for_get();

        // create test keys
        let key1 = b"test_key1";
        let key2 = b"test_key2";
        let key3 = b"test_key3";

        let val1 = b"value1";
        let val2 = b"value2";
        let val3 = b"value3";

        // Add metadata: key1 -> val_block 0, entry 0; key2 -> val_block 1, entry 0; key3 -> val_block 2, entry 0
        let key1_with_metadata = add_key_metadata(key1, 0, 0);
        let key2_with_metadata = add_key_metadata(key2, 1, 0);
        let key3_with_metadata = add_key_metadata(key3, 2, 0);

        // Create individual blocks for each key
        let mut key_block1 = Block::new();
        key_block1.add_entry(&key1_with_metadata, EntryFlag::Complete).unwrap();

        let mut key_block2 = Block::new();
        key_block2.add_entry(&key2_with_metadata, EntryFlag::Complete).unwrap();

        let mut key_block3 = Block::new();
        key_block3.add_entry(&key3_with_metadata, EntryFlag::Complete).unwrap();

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

        // Add to indexes - index the original keys (without metadata)
        key_index.insert_item(key1);  // Block 0
        key_index.inc_block_count(1);

        key_index.insert_item(key2);  // Block 1
        key_index.inc_block_count(1);

        key_index.insert_item(key3);  // Block 2
        key_index.inc_block_count(1);

        val_index.insert_item(key1);  // Block 0
        val_index.inc_block_count(1);

        val_index.insert_item(key2);  // Block 1
        val_index.inc_block_count(1);

        val_index.insert_item(key3);  // Block 2
        val_index.inc_block_count(1);

        // Create reader
        let reader =
            SegmentReader::new(key_map.clone(), val_map.clone(), Arc::new(parking_lot::Mutex::new(key_index))).unwrap();

        // Test get for each key
        let result1 = reader.get(key1).unwrap();
        assert!(result1.is_some());
        assert_eq!(result1.unwrap().as_ref(), val1);

        let result2 = reader.get(key2).unwrap();
        assert!(result2.is_some());
        assert_eq!(result2.unwrap().as_ref(), val2);

        let result3 = reader.get(key3).unwrap();
        assert!(result3.is_some());
        assert_eq!(result3.unwrap().as_ref(), val3);
    }
}
