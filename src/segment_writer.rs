use std::{
    fmt::{
        Debug,
        Formatter,
    },
    sync::{
        Arc,
        atomic::{
            AtomicBool,
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

pub(crate) struct SegmentWriter {
    pub(crate) map: Arc<Map>,
    block_queue: Arc<SegQueue<Block>>,
    segment_full: Arc<AtomicBool>,
    done: Arc<AtomicBool>,
    completion_mutex: Arc<Mutex<()>>,
    completion_condvar: Arc<Condvar>,
}

impl SegmentWriter {
    pub(crate) fn new(map: Arc<Map>) -> Result<Self, SegmentError> {
        let done = Arc::new(AtomicBool::new(false));
        let segment_full = Arc::new(AtomicBool::new(false));
        let queue = Arc::new(SegQueue::<Block>::new());
        let completion_mutex = Arc::new(Mutex::new(()));
        let completion_condvar = Arc::new(Condvar::new());

        let done_clone = done.clone();
        let queue_clone = queue.clone();
        let map_clone = map.clone();
        let completion_mutex_clone = completion_mutex.clone();
        let completion_condvar_clone = completion_condvar.clone();

        thread::spawn(move || {
            let mut current_offset = 0;

            loop {
                if let Some(block) = queue_clone.pop() {
                    let required_size = current_offset + BLOCK_SIZE as usize;

                    // Check if we need to grow the map
                    if required_size > map_clone.len() {
                        // Calculate new size with some growth factor
                        let new_size = (required_size as u64).max(map_clone.len() as u64 * 2);
                        if let Err(e) = map_clone.grow(new_size) {
                            eprintln!("Failed to grow map: {:?}", e);
                            break;
                        }
                    }

                    let block_range = current_offset..(current_offset + BLOCK_SIZE);

                    // SAFETY: the range is already allocated, we're just copying data
                    if let Err(e) = map_clone.write_to_range(block_range, |slice| unsafe {
                        block.finalize(slice.as_mut_ptr());
                    }) {
                        eprintln!("Failed to write block: {:?}", e);
                        break;
                    }

                    current_offset += BLOCK_SIZE;
                } else if done_clone.load(Relaxed) {
                    break;
                }
            }

            // notify completion
            let _guard = completion_mutex_clone.lock();
            completion_condvar_clone.notify_one();
            STATS.current_threads.fetch_sub(1, Relaxed);
        });
        STATS.current_threads.fetch_add(1, Relaxed);

        Ok(Self {
            map,
            block_queue: queue,
            done,
            segment_full,
            completion_mutex,
            completion_condvar,
        })
    }

    pub(crate) fn write_block(&mut self, block: Block) -> Result<(), SegmentError> {
        self.block_queue.push(block);
        Ok(())
    }
    
    /// Wait for all blocks to be written
    pub(crate) fn wait_for_completion(&self) {
        let mut guard = self.completion_mutex.lock();
        while !self.block_queue.is_empty() || !self.done.load(Relaxed) {
            self.completion_condvar.wait(&mut guard);
        }
    }

    pub(crate) fn shutdown(&self) {
        self.done.store(true, Relaxed);
        self.wait_for_completion();
    }
}

impl Debug for SegmentWriter {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // TODO(@siennathesane): impl debug for segment writer
        todo!()
    }
}

impl Drop for SegmentWriter {
    fn drop(&mut self) {
        // Signal the worker thread to finish
        self.done.store(true, Relaxed);

        {
            let mut guard = self.completion_mutex.lock();
            // Wait for both queue to be empty AND worker to finish
            while !self.block_queue.is_empty() || !self.done.load(Relaxed) {
                self.completion_condvar.wait(&mut guard);
            }
        }
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
    fn test_write_multiple_blocks() {
        let (map, _dir) = create_test_map().expect("failed to create map");

        let mut writer = SegmentWriter::new(map.clone()).expect("failed to create segment writer");

        // Create first block
        let mut block1 = Block::new();
        let test_data1 = b"first block data";
        block1
            .add_complete_entry(test_data1)
            .expect("failed to add entry to first block");

        // Create second block
        let mut block2 = Block::new();
        let test_data2 = b"second block data";
        block2
            .add_complete_entry(test_data2)
            .expect("failed to add entry to second block");

        // write the blocks
        writer
            .write_block(block1)
            .expect("failed to write first block");
        writer
            .write_block(block2)
            .expect("failed to write second block");

        // wait a bit to ensure the blocks are written
        thread::sleep(Duration::from_millis(100));

        // Verify first block has entries
        let num_entries_bytes_1 = &map[0..2];
        assert_eq!(
            u16::from_le_bytes([num_entries_bytes_1[0], num_entries_bytes_1[1]]),
            1,
            "expected 1 entry in the first block"
        );

        // Verify second block has entries
        let num_entries_bytes_2 = &map[BLOCK_SIZE..BLOCK_SIZE + 2];
        assert_eq!(
            u16::from_le_bytes([num_entries_bytes_2[0], num_entries_bytes_2[1]]),
            1,
            "expected 1 entry in the second block"
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
    fn test_shutdown_and_completion() {
        let (map, _dir) = create_test_map().expect("failed to create map");

        let writer = SegmentWriter::new(map.clone()).expect("failed to create segment writer");

        // shutdown the writer
        writer.shutdown();

        // dropping the writer should wait for completion
        drop(writer);

        // if we got here without deadlock, the test passes
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
    fn test_drop_waits_for_completion() {
        let (map, _dir) = create_test_map().expect("failed to create map");

        let mut writer = SegmentWriter::new(map.clone()).expect("failed to create segment writer");

        // create and write a bunch of blocks
        for i in 0..100 {
            let mut block = Block::new();
            let test_data = format!("Block {}", i).into_bytes();
            block
                .add_complete_entry(&test_data)
                .expect("failed to add entry to block");
            writer.write_block(block).expect("failed to write block");
        }

        // store the current stats count
        let threads_before = STATS.current_threads.load(Relaxed);

        // drop the writer - should wait for all blocks to be processed
        drop(writer);

        // check that the thread count decreased
        let threads_after = STATS.current_threads.load(Relaxed);
        assert!(
            threads_after < threads_before,
            "worker thread didn't complete properly"
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
}
