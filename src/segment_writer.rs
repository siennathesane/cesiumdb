use std::{
    marker,
    marker::PhantomData,
    mem,
    mem::ManuallyDrop,
    sync::{
        atomic::{
            AtomicBool,
            AtomicUsize,
            Ordering::Relaxed,
        },
        Arc,
    },
    thread,
    thread::JoinHandle,
    time::Duration,
};

use bytes::BytesMut;
use crossbeam_queue::SegQueue;
use parking_lot::{
    Condvar,
    Mutex,
    RwLock,
};

use crate::{
    block::{
        Block,
        EntryFlag::Complete,
        BLOCK_SIZE,
    },
    fs::{
        FRangeHandle,
        Fs,
    },
    segment::Segment,
    stats::STATS,
};
use crate::errs::SegmentError;
use crate::errs::SegmentError::{InsufficientSpace, InvalidSize};

pub(crate) struct SegmentWriter {
    handle: Arc<FRangeHandle>,
    blocks_left: Arc<AtomicUsize>,
    block_queue: Arc<SegQueue<Block>>,
    segment_full: Arc<AtomicBool>,
    done: Arc<AtomicBool>,
    completion_mutex: Arc<Mutex<()>>,
    completion_condvar: Arc<Condvar>,
}

impl SegmentWriter {
    pub(crate) fn new(handle: Arc<FRangeHandle>) -> Result<Self, SegmentError> {
        let segment_size = handle.capacity();
        if segment_size % BLOCK_SIZE as u64 != 0 {
            return Err(InvalidSize);
        }
        
        let blocks_left = Arc::new(AtomicUsize::new(segment_size as usize / BLOCK_SIZE));
        let done = Arc::new(AtomicBool::new(false));
        let segment_full = Arc::new(AtomicBool::new(false));
        let queue = Arc::new(SegQueue::<Block>::new());
        let completion_mutex = Arc::new(Mutex::new(()));
        let completion_condvar = Arc::new(Condvar::new());

        let done_clone = done.clone();
        let queue_clone = queue.clone();
        let handle_clone = handle.clone();
        let completion_mutex_clone = completion_mutex.clone();
        let completion_condvar_clone = completion_condvar.clone();

        // Spawn the worker thread
        thread::spawn(move || {
            let mut current_offset = 0;

            loop {
                if let Some(block) = queue_clone.pop() {
                    // TODO(@siennathesane): the block should be written directly to the handle
                    // instead of a buffer.

                    // Create a buffer for the block data
                    let mut buffer = BytesMut::with_capacity(BLOCK_SIZE);
                    buffer.resize(BLOCK_SIZE, 0); // Important: resize buffer to full block size

                    #[allow(clippy::missing_safety_doc)]
                    #[allow(clippy::undocumented_unsafe_blocks)]
                    unsafe {
                        block.finalize(buffer.as_mut_ptr());
                    }

                    // Write the block to the handle
                    if let Err(e) = handle_clone.write_at(current_offset, &buffer) {
                        break;
                    }

                    current_offset += BLOCK_SIZE as u64;
                } else if done_clone.load(Relaxed) {
                    break;
                }
            }

            // Notify completion
            let _guard = completion_mutex_clone.lock();
            completion_condvar_clone.notify_one();
            STATS.current_threads.fetch_sub(1, Relaxed);
        });
        STATS.current_threads.fetch_add(1, Relaxed);

        Ok(Self {
            handle,
            blocks_left,
            block_queue: queue,
            done,
            segment_full,
            completion_mutex,
            completion_condvar,
        })
    }

    pub(crate) fn write_block(&mut self, block: Block) -> Result<(), SegmentError> {
        if self.blocks_left.load(Relaxed) == 0 {
            self.segment_full.store(true, Relaxed);
            return Err(InsufficientSpace);
        }

        self.block_queue.push(block);
        self.blocks_left.fetch_sub(1, Relaxed);

        Ok(())
    }

    pub(crate) fn shutdown(&self) {
        self.done.store(true, Relaxed);
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
    use std::{
        fs::OpenOptions,
        path::PathBuf,
        sync::Arc,
        thread,
        time::Duration,
    };

    use bytes::Bytes;
    use memmap2::MmapMut;
    use parking_lot::RwLock;
    use rand::Rng;
    use tempfile::tempdir;
    use crate::errs::BlockError;
    use super::*;
    use crate::utils::Deserializer;

    struct TestContext {
        handle: Arc<FRangeHandle>,
        _dir: tempfile::TempDir,
    }

    fn setup_fs() -> (Arc<Fs>, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)
            .unwrap();

        file.set_len(1024 * 1024 * 10).unwrap(); // 10MB
        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let fs = Fs::init(mmap).unwrap();

        (fs, dir)
    }

    // Helper function to create test context with required resources
    // Update the test context to use Arc<Fs>
    fn create_test_context(fs: &Arc<Fs>, size: u64) -> TestContext {
        let dir = tempdir().unwrap();

        let frange_id = fs.create_frange(size).unwrap();
        let handle = fs.open_frange(frange_id).unwrap();

        TestContext {
            handle: Arc::new(handle),
            _dir: dir,
        }
    }

    fn create_test_block(entries: &[(Vec<u8>, Vec<u8>)]) -> Result<Block, BlockError> {
        let mut block = Block::new();
        // Only store values, not key-value pairs
        for (_, value) in entries {
            block.add_entry(value, Complete)?;
        }
        Ok(block)
    }

    fn verify_block_contents(data: &[u8], expected_values: &[Vec<u8>]) -> bool {
        let block = Block::deserialize(Bytes::copy_from_slice(data));

        // Verify each entry
        if let Some((idx, expected_value)) = expected_values.iter().enumerate().next() {
            let entry = match block.get(idx) {
                | Some(entry) => entry.1,
                | None => return false,
            };
            if entry.len() != expected_value.len() {
                return false;
            }
            return entry == expected_value;
        }
        true
    }

    #[test]
    fn test_basic_write_operation() {
        println!("Starting test_basic_write_operation");
        let segment_size = BLOCK_SIZE as u64 * 10;
        let (fs, _dir) = setup_fs();
        let fs = Arc::new(fs); // Wrap in Arc
        let ctx = create_test_context(&fs, segment_size);

        let entries = vec![
            (vec![0u8; 1], vec![1u8; 100]), // Simple small entry
        ];

        let block = create_test_block(&entries).unwrap();

        let mut writer = SegmentWriter::new(ctx.handle.clone()).unwrap();

        assert!(writer.write_block(block).is_ok());

        writer.shutdown();

        // Wait for writer to finish and properly sync
        drop(writer);

        // Add a small delay to ensure filesystem sync
        thread::sleep(Duration::from_millis(100));

        let mut buffer = vec![0u8; BLOCK_SIZE];
        ctx.handle.read_at(0, &mut buffer).unwrap();

        let values: Vec<Vec<u8>> = entries.iter().map(|(_, value)| value.clone()).collect();
    }

    #[test]
    fn test_segment_full() {
        let segment_size = BLOCK_SIZE as u64 * 2; // Only 2 blocks
        let (fs, _dir) = setup_fs();
        let ctx = create_test_context(&fs, segment_size);

        let mut writer = SegmentWriter::new(ctx.handle).unwrap();

        // Create test entries that fill blocks
        let test_entries = vec![(vec![1u8; 2000], vec![2u8; 1000])]; // Large entries

        // Write blocks until full
        let block1 = create_test_block(&test_entries).unwrap();
        let block2 = create_test_block(&test_entries).unwrap();
        let block3 = create_test_block(&test_entries).unwrap();

        assert!(writer.write_block(block1).is_ok());
        assert!(writer.write_block(block2).is_ok());
        assert!(matches!(
            writer.write_block(block3).unwrap_err(),
            InsufficientSpace
        ));

        writer.shutdown();
    }

    #[test]
    fn test_concurrent_writes() {
        let segment_size = BLOCK_SIZE as u64 * 100; // 100 blocks
        let (fs, _dir) = setup_fs();
        let ctx = create_test_context(&fs, segment_size);

        let writer = Arc::new(Mutex::new(SegmentWriter::new(ctx.handle).unwrap()));
        let mut handles = vec![];

        for thread_id in 0..10 {
            let writer_clone = writer.clone();
            let handle = thread::spawn(move || {
                for i in 0..5 {
                    let entries = vec![(
                        format!("key_{}_{}", thread_id, i).into_bytes(),
                        format!("value_{}_{}", thread_id, i).into_bytes(),
                    )];

                    let block = create_test_block(&entries).unwrap();
                    let mut writer = writer_clone.lock();
                    assert!(writer.write_block(block).is_ok());
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let writer = writer.lock();
        assert_eq!(writer.blocks_left.load(Relaxed), 50); // 100 - (10 threads * 5 blocks)
        writer.shutdown();
    }

    #[test]
    fn test_block_queue_ordering() {
        let segment_size = BLOCK_SIZE as u64 * 10;
        let (fs, _dir) = setup_fs();
        let ctx = create_test_context(&fs, segment_size);

        let mut writer = SegmentWriter::new(ctx.handle.clone()).unwrap();
        let mut expected_entries = Vec::new();

        // Write blocks with sequential numbers
        for i in 0..5 {
            let entries = vec![(
                format!("key_{}", i).into_bytes(),
                format!("value_{}", i).into_bytes(),
            )];
            expected_entries.push(entries[0].clone());

            let block = create_test_block(&entries).unwrap();
            assert!(writer.write_block(block).is_ok());
        }

        // Wait for writes to complete
        thread::sleep(Duration::from_millis(100));
        writer.shutdown();

        // Wait for cleanup
        thread::sleep(Duration::from_millis(100));

        // Read back and verify order
        {
            let handle = ctx.handle;
            let mut results = Vec::new();

            for i in 0..5 {
                let mut buffer = vec![0u8; BLOCK_SIZE];
                assert!(handle.read_at(i * BLOCK_SIZE as u64, &mut buffer).is_ok());

                // Extract just the value for verification
                let values = vec![expected_entries[i as usize].1.clone()];
                results.push(verify_block_contents(&buffer, &values));
            }

            // Now verify all results
            for result in results {
                assert!(result);
            }
        }
    }

    #[test]
    fn test_stress_write() {
        let segment_size = BLOCK_SIZE as u64 * 100; // 100 blocks
        let (fs, _dir) = setup_fs();

        let frange_id = fs.create_frange(segment_size).unwrap();
        let handle = fs.open_frange(frange_id).unwrap();
        let handle = Arc::new(handle);

        let writer = Arc::new(Mutex::new(SegmentWriter::new(handle).unwrap()));
        let mut handles = vec![];

        let total_threads = 10;
        let blocks_per_thread = 5; // Reduced from 50 to ensure we don't overflow
        let total_blocks = total_threads * blocks_per_thread;

        // Verify we have enough space
        assert!(segment_size >= (total_blocks * BLOCK_SIZE) as u64);

        // Track errors across threads
        let error_count = Arc::new(AtomicUsize::new(0));

        for thread_id in 0..total_threads {
            let writer_clone = writer.clone();
            let error_count = error_count.clone();

            let handle = thread::spawn(move || {
                let mut rng = rand::thread_rng();
                for i in 0..blocks_per_thread {
                    // Create entries with random sizes but ensure they fit in a block
                    let key_size = rng.gen_range(1..100);
                    let value_size = rng.gen_range(1..1000);

                    let key = format!("key_{}_{}", thread_id, i).into_bytes();
                    let mut value = vec![0u8; value_size];
                    rng.fill(&mut value[..]);

                    let entries = vec![(key, value)];
                    match create_test_block(&entries) {
                        | Ok(block) => {
                            let mut writer = writer_clone.lock();
                            if writer.write_block(block).is_err() {
                                error_count.fetch_add(1, Relaxed);
                                break; // Exit if we hit an error
                            }
                        },
                        | Err(_) => {
                            error_count.fetch_add(1, Relaxed);
                            break;
                        },
                    }

                    // Small random sleep to increase concurrency variations
                    if rng.gen_ratio(1, 10) {
                        thread::sleep(Duration::from_millis(1));
                    }
                }
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // Get the final writer state
        let writer = writer.lock();
        writer.shutdown();

        // Verify results
        assert_eq!(
            error_count.load(Relaxed),
            0,
            "Some threads encountered errors"
        );
        assert_eq!(
            writer.blocks_left.load(Relaxed),
            segment_size as usize / BLOCK_SIZE - total_blocks
        );
    }

    #[test]
    fn test_multiple_entry_block() {
        let segment_size = BLOCK_SIZE as u64 * 10;
        let (fs, _dir) = setup_fs();
        let ctx = create_test_context(&fs, segment_size);

        let entries = vec![
            (vec![1u8; 1000], vec![2u8; 1000]),
            (vec![3u8; 500], vec![4u8; 500]),
        ];

        // Write data
        let mut writer = SegmentWriter::new(ctx.handle.clone()).unwrap();
        let block = create_test_block(&entries).unwrap();
        assert!(writer.write_block(block).is_ok());
        writer.shutdown();
        drop(writer);

        // Verify written data
        let mut buffer = vec![0u8; BLOCK_SIZE];
        {
            let handle = ctx.handle;
            assert!(handle.read_at(0, &mut buffer).is_ok());
        }

        let values: Vec<Vec<u8>> = entries.iter().map(|(_, value)| value.clone()).collect();
        assert!(
            verify_block_contents(&buffer, &values),
            "Block contents do not match"
        );
    }
}
