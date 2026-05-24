use std::{
    fmt::{
        Debug,
        Formatter,
    },
    sync::{
        Arc,
        atomic::{
            AtomicBool,
            AtomicU64,
            AtomicUsize,
            Ordering::Relaxed,
        },
    },
    thread,
    time::Duration,
};

use bytes::{
    BufMut,
    Bytes,
};
use crossbeam_queue::SegQueue;
use parking_lot::{
    Condvar,
    Mutex,
};
use tracing::instrument;

use crate::{
    block::{
        BLOCK_SIZE,
        Block,
    },
    errs::{
        SegmentError,
        SegmentError::{
            Closing,
            CorruptedBlock,
            NotClosing,
        },
    },
    index::Index,
    map::{
        MAX_GROWTH_INCREMENT,
        Map,
    },
    segment::Metadata,
    stats::STATS,
};

pub struct SegmentWriter {
    pub(crate) map: Arc<Map>,
    current_offset: Mutex<usize>,
    block_count: AtomicU64,
    closing: AtomicBool,
    closed: AtomicBool,
}

impl SegmentWriter {
    #[instrument(level = "trace")]
    pub fn new(map: Arc<Map>) -> Result<Self, SegmentError> {
        Ok(Self {
            map,
            current_offset: Mutex::new(0),
            block_count: AtomicU64::new(0),
            closing: AtomicBool::new(false),
            closed: AtomicBool::new(false),
        })
    }

    #[instrument(level = "trace")]
    fn calculate_new_size(&self, required_size: usize) -> u64 {
        // start with the minimum required size
        let mut new_size = required_size as u64;

        // add a growth increment of 4MiB or less
        let current_size = self.map.len() as u64;
        if new_size <= current_size + MAX_GROWTH_INCREMENT {
            // round up to the next multiple of MAX_GROWTH_INCREMENT
            new_size = new_size.div_ceil(MAX_GROWTH_INCREMENT) * MAX_GROWTH_INCREMENT;
        } else {
            // required size is already more than current + 4MiB, just use that
            // exact size this handles cases where a very large
            // batch of blocks is being written
        }

        // ensure we don't shrink
        new_size.max(current_size)
    }

    #[instrument(level = "trace")]
    pub(crate) fn write_block(&mut self, block: Block) -> Result<(), SegmentError> {
        if self.closing.load(Relaxed) {
            return Err(Closing);
        }

        let mut current_offset = self.current_offset.lock();
        let required_size = *current_offset + BLOCK_SIZE;

        // check if we need to grow the map
        if required_size > self.map.len() {
            let new_size = self.calculate_new_size(required_size);
            match self.map.grow(new_size) {
                | Ok(_) => {},
                | Err(e) => {
                    return Err(e);
                },
            };
        }

        // Write block to the map at the current offset
        let block_range = *current_offset..(*current_offset + BLOCK_SIZE);

        // SAFETY: We know the block is exactly BLOCK_SIZE bytes and we also know the
        // space is available
        match self.map.write_to_range(block_range, |slice| unsafe {
            block.finalize(slice.as_mut_ptr());
        }) {
            | Ok(_) => {},
            | Err(e) => {
                return Err(e);
            },
        };

        // Update offset for the next write
        *current_offset += BLOCK_SIZE;

        // Increment block counter
        self.block_count.fetch_add(1, Relaxed);

        Ok(())
    }

    /// Build a block directly in mmap memory using BlockBuilder (zero-copy).
    ///
    /// The builder_fn receives a BLOCK_SIZE mutable slice and should use
    /// BlockBuilder to write entries directly to mmap, avoiding
    /// intermediate BytesMut copies.
    pub(crate) fn write_block_direct<F>(&mut self, builder_fn: F) -> Result<(), SegmentError>
    where
        F: FnOnce(&mut [u8]), {
        if self.closing.load(Relaxed) {
            return Err(Closing);
        }

        let mut current_offset = self.current_offset.lock();
        let required_size = *current_offset + BLOCK_SIZE;

        // Grow map if needed
        if required_size > self.map.len() {
            let new_size = self.calculate_new_size(required_size);
            if let Err(e) = self.map.grow(new_size) {
                return Err(e);
            }
        }

        let block_range = *current_offset..(*current_offset + BLOCK_SIZE);

        // Write block directly to mmap using the builder function
        if let Err(e) = self.map.write_to_range(block_range, builder_fn) {
            return Err(e);
        }

        // Update offset and count
        *current_offset += BLOCK_SIZE;
        self.block_count.fetch_add(1, Relaxed);

        Ok(())
    }

    /// Write multiple blocks in a batch
    #[instrument(level = "trace")]
    pub(crate) fn write_blocks(&self, blocks: &[Block]) -> Result<(), SegmentError> {
        if self.closing.load(Relaxed) {
            return Err(Closing);
        }

        if blocks.is_empty() {
            return Ok(());
        }

        let mut current_offset = self.current_offset.lock();
        let total_size = blocks.len() * BLOCK_SIZE;
        let required_size = *current_offset + total_size;

        // check if we need to grow the map
        if required_size > self.map.len() {
            let new_size = self.calculate_new_size(required_size);
            match self.map.grow(new_size) {
                | Ok(_) => {},
                | Err(e) => {
                    return Err(e);
                },
            };
        }

        // Write all blocks in sequence
        for (i, block) in blocks.iter().enumerate() {
            let block_offset = *current_offset + (i * BLOCK_SIZE);
            let block_range = block_offset..(block_offset + BLOCK_SIZE);

            // SAFETY: We know the block is exactly BLOCK_SIZE bytes and we also know the
            // space is available
            match self.map.write_to_range(block_range, |slice| unsafe {
                block.finalize(slice.as_mut_ptr());
            }) {
                | Ok(_) => {},
                | Err(e) => {
                    return Err(e);
                },
            };
        }

        // Update offset for the next write
        *current_offset += total_size;

        // Increment block counter
        self.block_count.fetch_add(blocks.len() as u64, Relaxed);

        Ok(())
    }

    /// Mark the segment as closing. This must be called before write_metadata.
    /// For segments with an index, write_index calls this automatically.
    #[instrument(level = "trace")]
    pub(crate) fn begin_close(&self) {
        self.closing.store(true, Relaxed);
    }

    /// Write the index to the map. Once the index has been written, no more
    /// blocks can be written.
    #[instrument(level = "trace")]
    pub(crate) fn write_index(&self, index: &Index) -> Result<u64, SegmentError> {
        self.begin_close();

        let mut current_offset = self.current_offset.lock();
        let index_start = *current_offset;
        let index_end = *current_offset + index.size();

        let index_bytes = Bytes::from(index);
        let index_size = index_bytes.len();

        if index_size == 0 || index_size < 56 {
            return Err(CorruptedBlock);
        }

        // check if we need to grow the map
        if index_end > self.map.len() {
            let new_size = self.calculate_new_size(index_end);
            match self.map.grow(new_size) {
                | Ok(_) => {},
                | Err(e) => {
                    return Err(e);
                },
            };
        }

        let index_range = index_start..index_end;

        // SAFETY: We know the block is exactly BLOCK_SIZE bytes, and we also know the
        // space is available
        match self.map.write_to_range(index_range, |slice| unsafe {
            index.finalize(slice.as_mut_ptr());
        }) {
            | Ok(_) => {},
            | Err(e) => {
                return Err(e);
            },
        };

        *current_offset = index_end;

        Ok(index_start as u64)
    }

    #[instrument(level = "trace")]
    pub(crate) fn write_metadata(&self, metadata: Metadata) -> Result<(), SegmentError> {
        if !self.closing.load(Relaxed) {
            return Err(NotClosing);
        }

        let metadata_size = metadata.serialized_size();
        let mut current_offset = self.current_offset.lock();
        let metadata_start = *current_offset;
        let metadata_end = *current_offset + metadata_size;

        // Grow map if needed
        if metadata_end > self.map.len() {
            let new_size = self.calculate_new_size(metadata_end);
            match self.map.grow(new_size) {
                | Ok(_) => {},
                | Err(e) => return Err(e),
            };
        }

        let metadata_range = metadata_start..metadata_end;

        // Write metadata at current offset so it becomes the last bytes after shrink
        // SAFETY: we know this memory range will exist
        match self.map.write_to_range(metadata_range, |slice| unsafe {
            metadata.finalize(slice.as_mut_ptr());
        }) {
            | Ok(_) => {},
            | Err(e) => return Err(e),
        };

        *current_offset = metadata_end;
        Ok(())
    }

    pub(crate) fn current_offset(&self) -> usize {
        *self.current_offset.lock()
    }

    /// Get the number of blocks written
    pub(crate) fn block_count(&self) -> u64 {
        self.block_count.load(Relaxed)
    }

    /// Close the segment writer. This will flush any remaining data to the map,
    /// truncate the file to the actual written size, and it is now safe to
    /// `drop`.
    #[instrument(level = "trace")]
    pub(crate) fn close(&self) -> Result<(), SegmentError> {
        if self.closed.load(Relaxed) {
            return Ok(());
        }

        let current_offset = self.current_offset();
        self.map.close()?;

        // Truncate file to actual written size to eliminate space amplification
        // from pre-allocated unused space.
        self.map.shrink(current_offset as u64)?;

        self.closed.store(true, Relaxed);
        Ok(())
    }
}

impl Debug for SegmentWriter {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SegmentWriter")
            .field("current_offset", &self.current_offset())
            .field("block_count", &self.block_count())
            .field("map_size", &self.map.len())
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::question_mark_used)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::undocumented_unsafe_blocks)]
mod tests {
    use std::time::Duration;

    use tempfile::tempdir;

    use super::*;
    use crate::block::{
        BLOCK_SIZE,
        Block,
        EntryFlag,
    };

    // helper function to create a temporary map for testing
    fn create_test_map() -> Result<(Arc<Map>, tempfile::TempDir), SegmentError> {
        let dir = tempdir().expect("failed to create temp dir");
        let file_path = dir.path().join("test-segment");
        let map = Arc::new(Map::new(file_path, BLOCK_SIZE as u64 * 10)?);
        Ok((map, dir))
    }

    #[test]
    fn test_segment_writer_creation() {
        let (map, _dir) = create_test_map().expect("failed to create map");

        let writer = SegmentWriter::new(map);
        assert!(writer.is_ok(), "failed to create segment writer");
    }

    #[test]
    fn test_write_single_block() {
        let (map, _dir) = create_test_map().expect("failed to create map");

        let mut writer = SegmentWriter::new(map.clone()).expect("failed to create segment writer");

        // create a test block with some data
        let mut block = Block::new();
        let test_data = b"test block data";
        block
            .add_complete_entry(test_data)
            .expect("failed to add entry to block");

        // write the block
        writer.write_block(block).expect("failed to write block");

        // wait a bit to ensure the block is written
        thread::sleep(Duration::from_millis(50));

        // Verify by reading the first few bytes of the map
        // The block structure starts with num_entries (u16)
        let num_entries = map
            .read_range(0..2, |bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .expect("failed to read num_entries");
        assert_eq!(num_entries, 1, "expected 1 entry in the block");
    }

    #[test]
    fn test_auto_growing() {
        // start with a very small map that needs to grow
        let dir = tempdir().expect("failed to create temp dir");
        let file_path = dir.path().join("small-map");
        let map =
            Arc::new(Map::new(file_path, BLOCK_SIZE as u64 / 2).expect("failed to create map")); // smaller than one block

        let mut writer = SegmentWriter::new(map.clone()).expect("failed to create segment writer");

        // create a block that will require the map to grow
        let mut block = Block::new();
        let test_data = b"this data should force the map to grow";
        block
            .add_complete_entry(test_data)
            .expect("failed to add entry to block");

        // initial size check
        let initial_size = map.len();
        assert!(
            initial_size < BLOCK_SIZE,
            "initial map size should be less than block size"
        );

        // write the block
        writer.write_block(block).expect("failed to write block");

        // wait for the block to be processed
        thread::sleep(Duration::from_millis(100));

        // check if the map grew
        let new_size = map.len();
        assert!(
            new_size >= BLOCK_SIZE,
            "map should have grown to at least block size"
        );

        // Verify the block was written by checking num_entries
        let num_entries = map
            .read_range(0..2, |bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .expect("failed to read");
        assert_eq!(num_entries, 1, "expected 1 entry in the block");
    }

    #[test]
    fn test_concurrent_writing() {
        let (map, _dir) = create_test_map().expect("failed to create map");

        let writer = Arc::new(Mutex::new(
            SegmentWriter::new(map.clone()).expect("failed to create segment writer"),
        ));

        // spawn multiple threads to write blocks concurrently
        let thread_count = 5;
        let blocks_per_thread = 10;

        let handles: Vec<_> = (0..thread_count)
            .map(|thread_id| {
                let writer_clone = writer.clone();

                thread::spawn(move || {
                    for i in 0..blocks_per_thread {
                        let mut block = Block::new();
                        let test_data = format!("Thread {} Block {}", thread_id, i).into_bytes();
                        block
                            .add_complete_entry(&test_data)
                            .expect("failed to add entry to block");

                        let mut writer_guard = writer_clone.lock();
                        writer_guard
                            .write_block(block)
                            .expect("failed to write block");
                    }
                })
            })
            .collect();

        // wait for all threads to complete
        for handle in handles {
            handle.join().expect("thread panicked");
        }

        // shutdown and drop the writer
        let writer = Arc::try_unwrap(writer)
            .expect("failed to unwrap Arc")
            .into_inner();
        drop(writer);

        // verify the map has enough data (we can't verify exact content due to thread
        // ordering)
        let expected_min_size = BLOCK_SIZE * (thread_count * blocks_per_thread) as usize;
        assert!(
            map.len() >= expected_min_size,
            "map size is less than expected"
        );
    }

    #[test]
    fn test_multiple_entries_per_block() {
        let (map, _dir) = create_test_map().expect("failed to create map");

        let mut writer = SegmentWriter::new(map.clone()).expect("failed to create segment writer");

        // Create a block with multiple entries
        let mut block = Block::new();

        // Add several entries to the same block
        let entries = [
            b"first entry",
            b"secon entry",
            b"third entry",
            b"fourt entry",
        ];

        for entry in &entries {
            block
                .add_complete_entry(*entry)
                .expect("failed to add entry to block");
        }

        // write the block
        writer.write_block(block).expect("failed to write block");

        // wait a bit to ensure the block is written
        thread::sleep(Duration::from_millis(50));

        // Verify the correct number of entries
        let num_entries = map
            .read_range(0..2, |bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .expect("failed to read");
        assert_eq!(
            num_entries,
            entries.len() as u16,
            "expected {} entries in the block",
            entries.len()
        );
    }

    #[test]
    fn test_block_with_fragmented_entry() {
        let (map, _dir) = create_test_map().expect("failed to create map");

        let mut writer = SegmentWriter::new(map.clone()).expect("failed to create segment writer");

        // Create blocks with fragmented entries
        let mut start_block = Block::new();
        let mut middle_block = Block::new();
        let mut end_block = Block::new();

        let start_data = b"This is the start of a fragmented entry";
        let middle_data = b"This is the middle part of the fragmented entry";
        let end_data = b"This is the end of the fragmented entry";

        start_block
            .add_entry(start_data, EntryFlag::Start)
            .expect("failed to add start entry");
        middle_block
            .add_entry(middle_data, EntryFlag::Middle)
            .expect("failed to add middle entry");
        end_block
            .add_entry(end_data, EntryFlag::End)
            .expect("failed to add end entry");

        // write the blocks
        writer
            .write_block(start_block)
            .expect("failed to write start block");
        writer
            .write_block(middle_block)
            .expect("failed to write middle block");
        writer
            .write_block(end_block)
            .expect("failed to write end block");

        // wait a bit to ensure blocks are written
        thread::sleep(Duration::from_millis(100));

        // Verify flags in the written blocks
        // This requires understanding the exact memory layout of blocks.
        // We know that after num_entries (2 bytes) and offset (2 bytes),
        // the first byte of the first entry is the flag

        // For simplicity, we'll just check that the blocks were written
        let size_used = map.len();
        assert!(
            size_used >= BLOCK_SIZE * 3,
            "expected at least 3 blocks to be used"
        );
    }

    #[test]
    fn test_write_blocks_empty() {
        let (map, _dir) = create_test_map().expect("failed to create map");
        let writer = SegmentWriter::new(map.clone()).expect("failed to create segment writer");

        // Empty blocks vector should succeed
        let blocks = Vec::new();
        let result = writer.write_blocks(&blocks);
        assert!(result.is_ok(), "writing empty blocks vector should succeed");
        assert_eq!(writer.current_offset(), 0, "offset should not change");
    }

    #[test]
    fn test_write_blocks_batch() {
        let (map, _dir) = create_test_map().expect("failed to create map");
        let writer = SegmentWriter::new(map.clone()).expect("failed to create segment writer");

        // Create three blocks with different data
        let mut block1 = Block::new();
        let mut block2 = Block::new();
        let mut block3 = Block::new();

        block1
            .add_complete_entry(b"first block data")
            .expect("failed to add entry");
        block2
            .add_complete_entry(b"second block data")
            .expect("failed to add entry");
        block3
            .add_complete_entry(b"third block data")
            .expect("failed to add entry");

        let blocks = vec![block1, block2, block3];

        // Write all blocks in one batch
        writer
            .write_blocks(&blocks)
            .expect("failed to write blocks batch");

        // Check if offset was updated correctly
        assert_eq!(
            writer.current_offset(),
            BLOCK_SIZE * 3,
            "offset should advance by 3 blocks"
        );

        // Verify first block
        let num_entries_1 = map
            .read_range(0..2, |bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .expect("failed to read");
        assert_eq!(num_entries_1, 1, "expected 1 entry in the first block");

        // Verify second block
        let num_entries_2 = map
            .read_range(BLOCK_SIZE..BLOCK_SIZE + 2, |bytes| {
                u16::from_le_bytes([bytes[0], bytes[1]])
            })
            .expect("failed to read");
        assert_eq!(num_entries_2, 1, "expected 1 entry in the second block");

        // Verify third block
        let num_entries_3 = map
            .read_range(BLOCK_SIZE * 2..BLOCK_SIZE * 2 + 2, |bytes| {
                u16::from_le_bytes([bytes[0], bytes[1]])
            })
            .expect("failed to read");
        assert_eq!(num_entries_3, 1, "expected 1 entry in the third block");
    }

    #[test]
    fn test_write_blocks_growth() {
        // Create a map that's just big enough for 1.5 blocks
        let dir = tempdir().expect("failed to create temp dir");
        let file_path = dir.path().join("small-grow-map");
        let map = Arc::new(
            Map::new(file_path, (BLOCK_SIZE + BLOCK_SIZE / 2) as u64)
                .expect("failed to create map"),
        );

        let writer = SegmentWriter::new(map.clone()).expect("failed to create segment writer");

        // Create three blocks that together exceed initial map size
        let mut blocks = Vec::new();
        for i in 0..3 {
            let mut block = Block::new();
            let data = format!("block data {}", i).into_bytes();
            block
                .add_complete_entry(&data)
                .expect("failed to add entry");
            blocks.push(block);
        }

        let initial_size = map.len();

        // Write all blocks at once
        writer
            .write_blocks(&blocks)
            .expect("failed to write blocks");

        // Check if the map grew
        let new_size = map.len();
        assert!(
            new_size > initial_size,
            "map should have grown when writing multiple blocks"
        );
        assert!(
            new_size >= BLOCK_SIZE * 3,
            "map should be able to accommodate all three blocks"
        );

        // Check offset was updated correctly
        assert_eq!(
            writer.current_offset(),
            BLOCK_SIZE * 3,
            "offset should advance by 3 blocks"
        );
    }

    #[test]
    fn test_write_blocks_with_varying_content() {
        let (map, _dir) = create_test_map().expect("failed to create map");
        let writer = SegmentWriter::new(map.clone()).expect("failed to create segment writer");

        let mut blocks = Vec::new();

        // Block with a single entry
        let mut block1 = Block::new();
        block1
            .add_complete_entry(b"single entry")
            .expect("failed to add entry");

        // Block with multiple entries
        let mut block2 = Block::new();
        block2
            .add_complete_entry(b"entry 1")
            .expect("failed to add entry");
        block2
            .add_complete_entry(b"entry 2")
            .expect("failed to add entry");

        // Block with a fragmented entry
        let mut block3 = Block::new();
        block3
            .add_entry(b"start fragment", EntryFlag::Start)
            .expect("failed to add entry");

        blocks.push(block1);
        blocks.push(block2);
        blocks.push(block3);

        writer
            .write_blocks(&blocks)
            .expect("failed to write blocks");

        // Verify first block has 1 entry
        let num_entries_1 = map
            .read_range(0..2, |bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .expect("failed to read");
        assert_eq!(num_entries_1, 1);

        // Verify second block has 2 entries
        let num_entries_2 = map
            .read_range(BLOCK_SIZE..BLOCK_SIZE + 2, |bytes| {
                u16::from_le_bytes([bytes[0], bytes[1]])
            })
            .expect("failed to read");
        assert_eq!(num_entries_2, 2);

        // Verify third block has 1 entry
        let num_entries_3 = map
            .read_range(BLOCK_SIZE * 2..BLOCK_SIZE * 2 + 2, |bytes| {
                u16::from_le_bytes([bytes[0], bytes[1]])
            })
            .expect("failed to read");
        assert_eq!(num_entries_3, 1);
    }

    #[test]
    fn test_sequential_write_blocks_calls() {
        let (map, _dir) = create_test_map().expect("failed to create map");
        let writer = SegmentWriter::new(map.clone()).expect("failed to create segment writer");

        // First batch
        let mut block1 = Block::new();
        let mut block2 = Block::new();
        block1
            .add_complete_entry(b"batch1-block1")
            .expect("failed to add entry");
        block2
            .add_complete_entry(b"batch1-block2")
            .expect("failed to add entry");

        writer
            .write_blocks(&[block1, block2])
            .expect("failed to write first batch");
        assert_eq!(writer.current_offset(), BLOCK_SIZE * 2);

        // Second batch
        let mut block3 = Block::new();
        let mut block4 = Block::new();
        block3
            .add_complete_entry(b"batch2-block1")
            .expect("failed to add entry");
        block4
            .add_complete_entry(b"batch2-block2")
            .expect("failed to add entry");

        writer
            .write_blocks(&[block3, block4])
            .expect("failed to write second batch");
        assert_eq!(writer.current_offset(), BLOCK_SIZE * 4);

        // Verify all blocks were written in sequence
        for i in 0..4 {
            let offset = i * BLOCK_SIZE;
            let num_entries = map
                .read_range(offset..offset + 2, |bytes| {
                    u16::from_le_bytes([bytes[0], bytes[1]])
                })
                .expect("failed to read");
            assert_eq!(num_entries, 1, "Block {} should have 1 entry", i);
        }
    }

    #[test]
    fn test_close_truncates_to_actual_size() {
        let (map, _dir) = create_test_map().expect("failed to create map");
        let initial_size = map.len();

        let mut writer = SegmentWriter::new(map.clone()).expect("failed to create segment writer");

        // Write a single block
        let mut block = Block::new();
        block
            .add_complete_entry(b"hello world")
            .expect("failed to add entry");
        writer.write_block(block).expect("failed to write block");

        let written = writer.current_offset();
        assert_eq!(written, BLOCK_SIZE);
        assert!(written < initial_size);

        // Close should truncate the file to the written size
        writer.close().expect("close failed");

        assert_eq!(
            map.len(),
            written,
            "map should be truncated to actual written size"
        );
    }
}
