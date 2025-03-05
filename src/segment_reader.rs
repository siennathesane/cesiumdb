use std::{
    ops::DerefMut,
    sync::Arc,
};

use bytes::{
    Buf,
    BytesMut,
};
use crossbeam_queue::ArrayQueue;

use crate::{
    block::{
        BLOCK_SIZE,
        Block,
    },
    errs::{
        SegmentError,
        SegmentError::{
            InvalidSize,
            ReadOutOfBounds,
        },
    },
    map::Map,
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
        Self { read_ahead: 2 }
    }
}

pub struct SegmentReader {
    key_handle: Arc<Map>,
    val_handle: Arc<Map>,
    visible_key_blocks: usize,
    visible_val_blocks: usize,
    num_blocks: usize,
    config: ReadConfig,
    // Cache for read-ahead blocks using a fixed-size queue
    cache: ArrayQueue<(usize, Block)>, // (block_index, block)
}

impl<'a> SegmentReader {
    pub fn new(key_handle: Arc<Map>, val_handle: Arc<Map>) -> Result<Self, SegmentError> {
        Self::with_config(key_handle, val_handle, ReadConfig::default())
    }

    pub(crate) fn with_config(
        key_handle: Arc<Map>,
        val_handle: Arc<Map>,
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
            visible_key_blocks,
            visible_val_blocks,
            num_blocks,
            cache: ArrayQueue::new(config.read_ahead + 1),
            config,
        })
    }
    
    pub(crate) fn with_visibility(
        key_handle: Arc<Map>,
        val_handle: Arc<Map>,
        visible_key_blocks: usize,
        visible_val_blocks: usize,
        config: ReadConfig,
    ) -> Result<Self, SegmentError> {
        let segment_size = key_handle.len();

        if segment_size % BLOCK_SIZE != 0 {
            return Err(InvalidSize);
        }

        let num_blocks = segment_size / BLOCK_SIZE;

        Ok(Self {
            key_handle,
            val_handle,
            visible_key_blocks,
            visible_val_blocks,
            num_blocks,
            cache: ArrayQueue::new(config.read_ahead + 1),
            config,
        })
    }

    pub(crate) fn read_block(&self, block_index: usize) -> Result<Block, SegmentError> {
        if block_index >= self.visible_key_blocks {
            return Err(ReadOutOfBounds);
        }

        // Check cache first and collect any blocks we may want to keep
        let mut found_block = None;
        let mut keep_blocks = Vec::new();

        while let Some((idx, block)) = self.cache.pop() {
            if idx == block_index {
                found_block = Some(block);
            } else if idx > block_index && idx < block_index + self.config.read_ahead {
                // Keep blocks that are within our read-ahead window
                keep_blocks.push((idx, block));
            }
        }

        // If we found our block, restore kept blocks and return
        if let Some(block) = found_block {
            // Restore kept blocks to cache
            for b in keep_blocks {
                let _ = self.cache.push(b);
            }
            match self.fill_cache(block_index + 1) {
                | Ok(_) => {},
                | Err(e) => return Err(e),
            };
            return Ok(block);
        }

        // Read the requested block
        let block = match self.read_block_at(block_index) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };

        // Fill read-ahead cache
        match self.fill_cache(block_index + 1) {
            | Ok(_) => {},
            | Err(e) => return Err(e),
        };

        Ok(block)
    }
    
    pub(crate) fn refresh(&mut self) {
        self.visible_key_blocks = self.key_handle.len() /BLOCK_SIZE;
        self.visible_val_blocks = self.val_handle.len() / BLOCK_SIZE;
        
        self.clear_cache();
    }

    pub(crate) fn visible_blocks(&self) -> (usize, usize) {
        (self.visible_key_blocks, self.visible_val_blocks)
    }

    pub(crate) fn iter(&'a mut self) -> SegmentBlockIterator<'a> {
        SegmentBlockIterator {
            reader: self,
            current_block: 0,
        }
    }

    pub(crate) fn seeking_iter(&'a mut self) -> SeekingBlockIterator<'a> {
        SeekingBlockIterator {
            start: 0,
            end: self.num_blocks,
            current: 0,
            reader: self,
        }
    }

    /// Internal method to read a single block without caching
    fn read_block_at(&self, block_index: usize) -> Result<Block, SegmentError> {
        let offset = block_index * BLOCK_SIZE;
        let mut buffer = BytesMut::zeroed(BLOCK_SIZE);

        if offset + BLOCK_SIZE > self.key_handle.len() {
            return Err(ReadOutOfBounds);
        }

        buffer.copy_from_slice(&self.key_handle[offset..offset + BLOCK_SIZE]);

        let block = Block::deserialize(buffer.freeze());

        Ok(block)
    }

    /// Fill the read-ahead cache starting from the given block index
    fn fill_cache(&self, start_index: usize) -> Result<(), SegmentError> {
        // Clear old cache entries
        while self.cache.pop().is_some() {}

        // Fill cache with next blocks
        for idx in start_index..self.num_blocks.min(start_index + self.config.read_ahead) {
            match self.read_block_at(idx) {
                | Ok(block) => {
                    // If push fails, cache is full, so we can stop
                    if self.cache.push((idx, block)).is_err() {
                        break;
                    }
                },
                | Err(e) => {
                    // Clear cache on error but don't fail the operation
                    while self.cache.pop().is_some() {}
                    return Err(e);
                },
            }
        }

        Ok(())
    }

    /// Get the total number of blocks in this segment
    #[inline]
    pub(crate) fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Clear the read-ahead cache
    pub(crate) fn clear_cache(&mut self) {
        while self.cache.pop().is_some() {}
    }

    /// Update the reader configuration
    pub(crate) fn set_config(&mut self, config: ReadConfig) {
        // Create new cache with updated capacity
        let new_cache = ArrayQueue::new(config.read_ahead + 1);
        self.cache = new_cache;
        self.config = config;
    }

    /// Get a reference to the current configuration
    pub(crate) fn config(&self) -> &ReadConfig {
        &self.config
    }
}

pub(crate) struct SegmentBlockIterator<'a> {
    reader: &'a mut SegmentReader,
    current_block: usize,
}

impl<'a> Iterator for SegmentBlockIterator<'a> {
    type Item = Result<Block, SegmentError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_block >= self.reader.num_blocks {
            return None;
        }

        let result = self.reader.read_block(self.current_block);
        self.current_block += 1;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.reader.num_blocks - self.current_block;
        (remaining, Some(remaining))
    }
}

pub(crate) struct SeekingBlockIterator<'a> {
    start: usize,
    end: usize,
    current: usize,
    reader: &'a mut SegmentReader,
}

impl<'a> SeekingBlockIterator<'a> {
    pub(crate) fn seek(&mut self, block_index: usize) -> Result<(), SegmentError> {
        if block_index >= self.end {
            return Err(ReadOutOfBounds);
        }
        self.reader.clear_cache();
        self.current = block_index;
        Ok(())
    }

    pub(crate) fn current_position(&self) -> usize {
        self.current
    }

    pub(crate) fn blocks_remaining(&self) -> usize {
        self.end - self.current
    }
}

impl<'a> Iterator for SeekingBlockIterator<'a> {
    type Item = Result<Block, SegmentError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.end {
            return None;
        }
        let result = self.reader.read_block(self.current);
        self.current += 1;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.current;
        (remaining, Some(remaining))
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

        let reader = SegmentReader::new(key_map.clone(), val_map.clone());
        assert!(reader.is_ok());

        let reader = reader.unwrap();
        assert_eq!(reader.num_blocks(), 4);
        assert_eq!(reader.config().read_ahead, 2); // Default value
    }

    #[test]
    fn test_with_config() {
        let size = BLOCK_SIZE * 4;
        let (_dir, key_map) = create_test_map(size);
        let (_dir2, val_map) = create_test_map(size);

        let config = ReadConfig { read_ahead: 3 };
        let reader = SegmentReader::with_config(key_map.clone(), val_map.clone(), config);

        assert!(reader.is_ok());
        let reader = reader.unwrap();
        assert_eq!(reader.config().read_ahead, 3);
    }

    #[test]
    fn test_invalid_size() {
        // Create Map with size that's not a multiple of BLOCK_SIZE
        let invalid_size = BLOCK_SIZE * 2 + 100; // Not a multiple of BLOCK_SIZE
        let (_dir, key_map) = create_test_map(invalid_size);
        let (_dir2, val_map) = create_test_map(invalid_size);

        let result = SegmentReader::new(key_map.clone(), val_map.clone());
        assert!(result.is_err());
        assert!(matches!(result.err().unwrap(), InvalidSize));
    }

    #[test]
    fn test_read_block() {
        let (_, key_map) = prepare_blocks_map(4);
        let (_, val_map) = prepare_blocks_map(4);

        let reader = SegmentReader::new(key_map, val_map).unwrap();

        // Read each block and verify contents
        for i in 0..4 {
            let block = reader.read_block(i).unwrap();
            let entry = block.get(0).unwrap();
            assert_eq!(entry.1, &vec![i as u8; 16]);
        }
    }

    #[test]
    fn test_read_block_out_of_bounds() {
        let (_, key_map) = prepare_blocks_map(2);
        let (_, val_map) = prepare_blocks_map(2);

        let reader = SegmentReader::new(key_map, val_map).unwrap();

        let result = reader.read_block(2); // Only 2 blocks exist (0 and 1)
        assert!(result.is_err());
        assert!(matches!(result.err().unwrap(), ReadOutOfBounds));
    }

    #[test]
    fn test_read_block_caching() {
        let (_, key_map) = prepare_blocks_map(5);
        let (_, val_map) = prepare_blocks_map(5);

        let reader = SegmentReader::new(key_map, val_map).unwrap();

        // First read
        let block0 = reader.read_block(0).unwrap();
        assert_eq!(block0.get(0).unwrap().1, &vec![0u8; 16]);

        // Next block should be in cache now
        let block1 = reader.read_block(1).unwrap();
        assert_eq!(block1.get(0).unwrap().1, &vec![1u8; 16]);

        // Skip to block 3, which should clear cache and rebuild
        let block3 = reader.read_block(3).unwrap();
        assert_eq!(block3.get(0).unwrap().1, &vec![3u8; 16]);

        // Block 4 should be in cache now
        let block4 = reader.read_block(4).unwrap();
        assert_eq!(block4.get(0).unwrap().1, &vec![4u8; 16]);
    }

    #[test]
    fn test_read_block_random_access() {
        let (_, key_map) = prepare_blocks_map(8);
        let (_, val_map) = prepare_blocks_map(8);

        let reader = SegmentReader::new(key_map, val_map).unwrap();

        // Access blocks in non-sequential order
        let indices = [3, 1, 5, 0, 7, 2];

        for &idx in &indices {
            let block = reader.read_block(idx).unwrap();
            assert_eq!(block.get(0).unwrap().1, &vec![idx as u8; 16]);
        }
    }

    #[test]
    fn test_clear_cache() {
        let (_, key_map) = prepare_blocks_map(4);
        let (_, val_map) = prepare_blocks_map(4);

        let mut reader = SegmentReader::new(key_map, val_map).unwrap();

        // Read a block to fill cache
        reader.read_block(0).unwrap();

        // Clear cache
        reader.clear_cache();

        // Cache should be empty, but this shouldn't affect functionality
        let block = reader.read_block(1).unwrap();
        assert_eq!(block.get(0).unwrap().1, &vec![1u8; 16]);
    }

    #[test]
    fn test_set_config() {
        let (_, key_map) = prepare_blocks_map(4);
        let (_, val_map) = prepare_blocks_map(4);

        let mut reader = SegmentReader::new(key_map, val_map).unwrap();

        // Change read-ahead configuration
        let new_config = ReadConfig { read_ahead: 1 };
        reader.set_config(new_config);

        assert_eq!(reader.config().read_ahead, 1);

        // With read-ahead of 1, only the next block should be cached
        reader.read_block(0).unwrap();

        // Block 1 should be cached
        let block1 = reader.read_block(1).unwrap();
        assert_eq!(block1.get(0).unwrap().1, &vec![1u8; 16]);

        // Block 3 shouldn't be in cache since read-ahead is only 1
        reader.read_block(3).unwrap();
    }

    #[test]
    fn test_segment_block_iterator() {
        let (_, key_map) = prepare_blocks_map(3);
        let (_, val_map) = prepare_blocks_map(3);

        let mut reader = SegmentReader::new(key_map, val_map).unwrap();

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

        let mut reader = SegmentReader::new(key_map, val_map).unwrap();
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

        let mut reader = SegmentReader::new(key_map, val_map).unwrap();

        // Test regular iterator
        let iter = reader.iter();
        let (min, max) = iter.size_hint();
        assert_eq!(min, 5);
        assert_eq!(max, Some(5));

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

        let mut reader = SegmentReader::new(key_map, val_map).unwrap();
        let mut iter = reader.seeking_iter();

        assert_eq!(iter.blocks_remaining(), 5);

        // Read one block
        iter.next();
        assert_eq!(iter.blocks_remaining(), 4);

        // Seek forward
        iter.seek(3).unwrap();
        assert_eq!(iter.blocks_remaining(), 2);
    }
}
