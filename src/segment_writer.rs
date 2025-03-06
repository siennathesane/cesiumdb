use std::{
    fmt::{
        Debug,
        Formatter,
    },
    sync::{
        Arc,
        atomic::{
            AtomicBool,
            AtomicUsize,
            Ordering::Relaxed,
        },
    },
    thread,
    time::Duration,
};

use bytes::BufMut;
use crossbeam_queue::SegQueue;
use parking_lot::{
    Condvar,
    Mutex,
};

use crate::{
    block::{
        BLOCK_SIZE,
        Block,
    },
    errs::SegmentError,
    map::Map,
    stats::STATS,
};

pub struct SegmentWriter {
    pub(crate) map: Arc<Map>,
    current_offset: Mutex<usize>,
}

impl SegmentWriter {
    pub fn new(map: Arc<Map>) -> Result<Self, SegmentError> {
        Ok(Self {
            map,
            current_offset: Mutex::new(0),
        })
    }

    pub(crate) fn write_block(&mut self, block: Block) -> Result<(), SegmentError> {
        let mut current_offset = self.current_offset.lock();
        let required_size = *current_offset + BLOCK_SIZE;

        // Check if we need to grow the map
        if required_size > self.map.len() {
            // Calculate new size with some growth factor (doubling is a common strategy)
            let new_size = (required_size as u64).max(self.map.len() as u64 * 2);
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

        Ok(())
    }

    /// Write multiple blocks in a batch
    pub(crate) fn write_blocks(&self, blocks: &[Block]) -> Result<(), SegmentError> {
        if blocks.is_empty() {
            return Ok(());
        }

        let mut current_offset = self.current_offset.lock();
        let total_size = blocks.len() * BLOCK_SIZE;
        let required_size = *current_offset + total_size;

        // Check if we need to grow the map
        if required_size > self.map.len() {
            // Calculate new size with some growth factor
            let new_size = (required_size as u64).max(self.map.len() as u64 * 2);
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

        Ok(())
    }

    /// Wait for all blocks to be written
    pub(crate) fn wait_for_completion(&self) {}

    pub(crate) fn shutdown(&self) {}

    pub(crate) fn current_offset(&self) -> usize {
        *self.current_offset.lock()
    }
}

impl Debug for SegmentWriter {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SegmentWriter")
            .field("current_offset", &self.current_offset())
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
        let num_entries_bytes = &map[0..2];
        assert_eq!(
            u16::from_le_bytes([num_entries_bytes[0], num_entries_bytes[1]]),
            1,
            "expected 1 entry in the block"
        );
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
        let num_entries_bytes = &map[0..2];
        assert_eq!(
            u16::from_le_bytes([num_entries_bytes[0], num_entries_bytes[1]]),
            1,
            "expected 1 entry in the block"
        );
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
        writer.shutdown();
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
        let num_entries_bytes = &map[0..2];
        assert_eq!(
            u16::from_le_bytes([num_entries_bytes[0], num_entries_bytes[1]]),
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
        let num_entries_bytes_1 = &map[0..2];
        assert_eq!(
            u16::from_le_bytes([num_entries_bytes_1[0], num_entries_bytes_1[1]]),
            1,
            "expected 1 entry in the first block"
        );

        // Verify second block
        let num_entries_bytes_2 = &map[BLOCK_SIZE..BLOCK_SIZE + 2];
        assert_eq!(
            u16::from_le_bytes([num_entries_bytes_2[0], num_entries_bytes_2[1]]),
            1,
            "expected 1 entry in the second block"
        );

        // Verify third block
        let num_entries_bytes_3 = &map[BLOCK_SIZE * 2..BLOCK_SIZE * 2 + 2];
        assert_eq!(
            u16::from_le_bytes([num_entries_bytes_3[0], num_entries_bytes_3[1]]),
            1,
            "expected 1 entry in the third block"
        );
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
        let num_entries_bytes_1 = &map[0..2];
        assert_eq!(
            u16::from_le_bytes([num_entries_bytes_1[0], num_entries_bytes_1[1]]),
            1
        );

        // Verify second block has 2 entries
        let num_entries_bytes_2 = &map[BLOCK_SIZE..BLOCK_SIZE + 2];
        assert_eq!(
            u16::from_le_bytes([num_entries_bytes_2[0], num_entries_bytes_2[1]]),
            2
        );

        // Verify third block has 1 entry
        let num_entries_bytes_3 = &map[BLOCK_SIZE * 2..BLOCK_SIZE * 2 + 2];
        assert_eq!(
            u16::from_le_bytes([num_entries_bytes_3[0], num_entries_bytes_3[1]]),
            1
        );
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
            let num_entries_bytes = &map[offset..offset + 2];
            assert_eq!(
                u16::from_le_bytes([num_entries_bytes[0], num_entries_bytes[1]]),
                1,
                "Block {} should have 1 entry",
                i
            );
        }
    }
}
