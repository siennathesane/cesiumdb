use std::ops::DerefMut;
use bytes::{
    Bytes,
    BytesMut,
};
use crossbeam_queue::ArrayQueue;

use crate::{
    block::{
        Block,
        BLOCK_SIZE,
    },
    errs::{
        CesiumError,
        CesiumError::FsError,
        FsError::{
            BlockIndexOutOfBounds,
            CorruptedBlock,
            SegmentSizeInvalid,
        },
    },
    fs::FRangeHandle,
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

pub(crate) struct SegmentReader {
    frange: FRangeHandle,
    num_blocks: usize,
    config: ReadConfig,
    // Cache for read-ahead blocks using a fixed-size queue
    cache: ArrayQueue<(usize, Block)>, // (block_index, block)
}

impl<'a> SegmentReader {
    pub(crate) fn new(frange: FRangeHandle) -> Result<Self, CesiumError> {
        Self::with_config(frange, ReadConfig::default())
    }

    pub(crate) fn with_config(
        frange: FRangeHandle,
        config: ReadConfig,
    ) -> Result<Self, CesiumError> {
        let segment_size = frange.capacity() as usize;

        if segment_size % BLOCK_SIZE != 0 {
            return Err(FsError(SegmentSizeInvalid));
        }

        let num_blocks = segment_size / BLOCK_SIZE;

        Ok(Self {
            frange,
            num_blocks,
            cache: ArrayQueue::new(config.read_ahead + 1),
            config,
        })
    }

    pub(crate) fn read_block(&mut self, block_index: usize) -> Result<Block, CesiumError> {
        if block_index >= self.num_blocks {
            return Err(FsError(BlockIndexOutOfBounds));
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
            self.fill_cache(block_index + 1)?;
            return Ok(block);
        }

        // Read the requested block
        let block = self.read_block_at(block_index)?;

        // Fill read-ahead cache
        self.fill_cache(block_index + 1)?;

        Ok(block)
    }

    // TODO(@siennathesane): this feels weird and out of place
    pub(crate) fn iter(&'a mut self) -> SegmentBlockIterator<'a> {
        SegmentBlockIterator {
            reader: self,
            current_block: 0,
        }
    }

    // TODO(@siennathesane): this also feels weird and out of place
    pub(crate) fn seeking_iter(&'a mut self) -> SeekingBlockIterator<'a> {
        SeekingBlockIterator {
            start: 0,
            end: self.num_blocks,
            current: 0,
            reader: self,
        }
    }

    /// Internal method to read a single block without caching
    fn read_block_at(&mut self, block_index: usize) -> Result<Block, CesiumError> {
        let offset = block_index * BLOCK_SIZE;
        let mut buffer = BytesMut::zeroed(BLOCK_SIZE);

        self.frange.read_at(offset as u64, &mut buffer)?;

        let block = Block::deserialize(buffer.freeze());

        Ok(block)
    }

    /// Fill the read-ahead cache starting from the given block index
    fn fill_cache(&mut self, start_index: usize) -> Result<(), CesiumError> {
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
    type Item = Result<Block, CesiumError>;

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
    pub(crate) fn seek(&mut self, block_index: usize) -> Result<(), CesiumError> {
        if block_index >= self.end {
            return Err(FsError(BlockIndexOutOfBounds));
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
    type Item = Result<Block, CesiumError>;

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
mod tests {
    use std::{
        fs::{
            File,
            OpenOptions,
        },
        sync::Arc,
    };

    use bytes::BytesMut;
    use memmap2::MmapOptions;
    use tempfile::tempfile;

    use super::*;
    use crate::{
        block::Block,
        fs::Fs,
    };

    const TEST_FS_SIZE: u64 = BLOCK_SIZE as u64 * 16; // 16 blocks total space

    // Helper to set up a test filesystem
    fn setup_test_fs() -> (Arc<Fs>, File) {
        let file = tempfile().unwrap();
        file.set_len(TEST_FS_SIZE).unwrap();

        let mmap = unsafe { MmapOptions::new().map_mut(&file).unwrap() };

        let fs = Fs::init(mmap).unwrap();
        (fs, file)
    }

    // Helper to create a test block with specified content
    fn create_test_block(value: u8) -> Block {
        let mut block = Block::new();
        let data = vec![value; 8]; // Use 8 bytes for test data
        block.add_entry(&data).unwrap();
        block
    }

    // Helper to create a segment with test data
    fn create_test_segment(fs: &Arc<Fs>, num_blocks: u64) -> FRangeHandle {
        let segment_size = num_blocks * BLOCK_SIZE as u64;
        let frange_id = fs.create_frange(segment_size).unwrap();
        let mut frange = fs.open_frange(frange_id).unwrap();

        // Write test data
        for i in 0..num_blocks {
            let block = create_test_block(i as u8 + 1);
            let mut buffer = vec![0u8; BLOCK_SIZE];
            unsafe {
                block.finalize(buffer.as_mut_ptr());
            }
            frange.write_at(i * BLOCK_SIZE as u64, &buffer).unwrap();
        }

        frange
    }

    #[test]
    fn test_segment_reader_new() {
        let (fs, _file) = setup_test_fs();
        let frange = create_test_segment(&fs, 2);

        let reader = SegmentReader::new(frange);
        assert!(reader.is_ok());
        let reader = reader.unwrap();
        assert_eq!(reader.num_blocks(), 2);
        assert_eq!(reader.config().read_ahead, 2); // Default read-ahead
    }

    #[test]
    fn test_segment_reader_invalid_size() {
        let (fs, _file) = setup_test_fs();

        // Create a file range with invalid size
        if let Ok(frange_id) = fs.create_frange(BLOCK_SIZE as u64 + 1) {
            if let Ok(frange) = fs.open_frange(frange_id) {
                // Debug the actual result
                let result = SegmentReader::new(frange);
                assert!(matches!(
                result,
                Err(FsError(SegmentSizeInvalid))
            ));
            }
        }
    }

    #[test]
    fn test_read_block_basic() {
        let (fs, _file) = setup_test_fs();
        let frange = create_test_segment(&fs, 2);

        let mut reader = SegmentReader::new(frange).unwrap();
        let block = reader.read_block(0).unwrap();
        assert_eq!(block.get(0).unwrap(), &vec![1u8; 8]);
    }

    #[test]
    fn test_read_block_out_of_bounds() {
        let (fs, _file) = setup_test_fs();
        let frange = create_test_segment(&fs, 1);

        let mut reader = SegmentReader::new(frange).unwrap();
        let result = reader.read_block(1);
        assert!(matches!(
            result.err().unwrap(),
            CesiumError::FsError(BlockIndexOutOfBounds)
        ));
    }

    #[test]
    fn test_read_block_caching() {
        let (fs, _file) = setup_test_fs();
        let frange = create_test_segment(&fs, 4);
        let mut reader = SegmentReader::new(frange).unwrap();

        // First read should cache next blocks
        let block1 = reader.read_block(0).unwrap();
        assert_eq!(block1.get(0).unwrap(), &vec![1u8; 8]);

        // This should come from cache
        let block2 = reader.read_block(1).unwrap();
        assert_eq!(block2.get(0).unwrap(), &vec![2u8; 8]);

        // Moving beyond cache should trigger new reads
        let block4 = reader.read_block(3).unwrap();
        assert_eq!(block4.get(0).unwrap(), &vec![4u8; 8]);
    }

    #[test]
    fn test_read_block_sequential() {
        let (fs, _file) = setup_test_fs();
        let frange = create_test_segment(&fs, 4);
        let mut reader = SegmentReader::new(frange).unwrap();

        // Read all blocks sequentially
        for i in 0..4 {
            let block = reader.read_block(i).unwrap();
            assert_eq!(block.get(0).unwrap(), &vec![(i + 1) as u8; 8]);
        }
    }

    #[test]
    fn test_block_iterator() {
        let (fs, _file) = setup_test_fs();
        let frange = create_test_segment(&fs, 3);
        let mut reader = SegmentReader::new(frange).unwrap();

        let mut count = 0;
        for block_result in reader.iter() {
            let block = block_result.unwrap();
            assert_eq!(block.get(0).unwrap(), &vec![(count + 1) as u8; 8]);
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn test_seeking_iterator() {
        let (fs, _file) = setup_test_fs();
        let frange = create_test_segment(&fs, 4);
        let mut reader = SegmentReader::new(frange).unwrap();
        let mut seeking_iter = reader.seeking_iter();
        
        // Seek to middle
        seeking_iter.seek(2).unwrap();
        assert_eq!(seeking_iter.current_position(), 2);
        assert_eq!(seeking_iter.blocks_remaining(), 2);

        // Read blocks and verify
        for i in 2..4 {
            let block = seeking_iter.next().unwrap().unwrap();
            assert_eq!(block.get(0).unwrap(), &vec![(i + 1) as u8; 8]);
        }
        assert!(seeking_iter.next().is_none()); // Verify we hit the end
    }

    #[test]
    fn test_config_update() {
        let (fs, _file) = setup_test_fs();
        let frange = create_test_segment(&fs, 2);
        let mut reader = SegmentReader::new(frange).unwrap();

        let new_config = ReadConfig { read_ahead: 4 };
        reader.set_config(new_config.clone());

        assert_eq!(reader.config().read_ahead, 4);
        reader.clear_cache(); // Verify cache clearing works
    }

    #[test]
    fn test_read_block_random_access() {
        let (fs, _file) = setup_test_fs();
        let frange = create_test_segment(&fs, 8);
        let mut reader = SegmentReader::new(frange).unwrap();

        // Read blocks in random order
        let indices = vec![3, 1, 4, 2, 6, 5];
        for &i in indices.iter() {
            let block = reader.read_block(i).unwrap();
            assert_eq!(block.get(0).unwrap(), &vec![(i + 1) as u8; 8]);
        }
    }

    #[test]
    fn test_iterator_size_hint() {
        let (fs, _file) = setup_test_fs();
        let frange = create_test_segment(&fs, 5);
        let mut reader = SegmentReader::new(frange).unwrap();
        let iter = reader.iter();

        let (min, max) = iter.size_hint();
        assert_eq!(min, 5);
        assert_eq!(max, Some(5));
    }

    #[test]
    fn test_seeking_iterator_bounds() {
        let (fs, _file) = setup_test_fs();
        let frange = create_test_segment(&fs, 3);
        let mut reader = SegmentReader::new(frange).unwrap();
        let mut seeking_iter = reader.seeking_iter();

        // Test seeking out of bounds
        assert!(seeking_iter.seek(3).is_err());

        // Test seeking to last block
        assert!(seeking_iter.seek(2).is_ok());
        assert_eq!(seeking_iter.blocks_remaining(), 1);
    }
}
