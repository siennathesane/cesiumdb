use std::{marker, mem, sync::{
    atomic::{
        AtomicBool,
        AtomicUsize,
        Ordering::Relaxed,
    },
    Arc,
}, thread};
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::thread::JoinHandle;
use std::time::Duration;
use bytes::BytesMut;
use crossbeam_queue::SegQueue;
use parking_lot::{Condvar, Mutex, RwLock};

use crate::{
    block::{
        Block,
        BLOCK_SIZE,
    },
    errs::{
        CesiumError,
        CesiumError::FsError,
        FsError::{
            SegmentFull,
            SegmentSizeInvalid,
        },
    },
    fs::Fs,
    stats::STATS,
};
use crate::fs::FRangeHandle;

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
    pub(crate) fn new(handle: Arc<FRangeHandle>) -> Result<Self, CesiumError> {
        let segment_size = handle.capacity();
        if segment_size % BLOCK_SIZE as u64 != 0 {
            return Err(FsError(SegmentSizeInvalid));
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
            println!("Worker thread started"); // Debug
            let mut current_offset = 0;

            loop {
                if let Some(block) = queue_clone.pop() {
                    println!("Processing block at offset {}", current_offset); // Debug

                    // Create a buffer for the block data
                    let mut buffer = BytesMut::with_capacity(BLOCK_SIZE);
                    buffer.resize(BLOCK_SIZE, 0); // Important: resize buffer to full block size

                    unsafe {
                        block.finalize(buffer.as_mut_ptr());
                    }
                    println!("Block finalized, first few bytes: {:?}", &buffer[..20]); // Debug

                    // Write the block to the handle
                    if let Err(e) = handle_clone.write_at(current_offset, &buffer) {
                        println!("Error writing block: {:?}", e); // Debug
                        break;
                    }
                    println!("Block written successfully"); // Debug

                    current_offset += BLOCK_SIZE as u64;
                } else if done_clone.load(Relaxed) {
                    println!("Worker thread done signal received"); // Debug
                    break;
                }
            }

            // Notify completion
            println!("Worker thread completing"); // Debug
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

    pub(crate) fn write_block(&mut self, block: Block) -> Result<(), CesiumError> {
        if self.blocks_left.load(Relaxed) == 0 {
            self.segment_full.store(true, Relaxed);
            return Err(FsError(SegmentFull));
        }
        
        self.block_queue.push(block);
        self.blocks_left.fetch_sub(1, Relaxed);

        Ok(())
    }
    
    pub(crate) fn write_index(&mut self, index: u64, data: &[u8]) -> Result<(), CesiumError> {
        if data.len() > BLOCK_SIZE {
            return Err(FsError(SegmentSizeInvalid));
        }

        let mut block = Block::new();
        block.add_entry(data)?;

        self.write_block(block)
    }

    pub(crate) fn shutdown(&self) {
        println!("Shutting down writer"); // Debug
        self.done.store(true, Relaxed);
    }
}

impl Drop for SegmentWriter {
    fn drop(&mut self) {
        println!("SegmentWriter being dropped"); // Debug
        // Signal the worker thread to finish
        self.done.store(true, Relaxed);

        {
            println!("Waiting for worker to complete"); // Debug
            let mut guard = self.completion_mutex.lock();
            // Wait for both queue to be empty AND worker to finish
            while !self.block_queue.is_empty() || !self.done.load(Relaxed) {
                self.completion_condvar.wait(&mut guard);
                println!("Woke up from wait, queue empty: {}, done: {}",
                         self.block_queue.is_empty(), self.done.load(Relaxed)); // Debug
            }
        }
        println!("SegmentWriter dropped"); // Debug
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::OpenOptions,
        path::PathBuf,
        thread,
        time::Duration,
        sync::Arc,
    };

    use memmap2::MmapMut;
    use rand::Rng;
    use tempfile::tempdir;
    use parking_lot::RwLock;

    use super::*;

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

    fn create_test_block(entries: &[(Vec<u8>, Vec<u8>)]) -> Result<Block, CesiumError> {
        let mut block = Block::new();
        // Only store values, not key-value pairs
        for (_, value) in entries {
            block.add_entry(value)?;
        }
        Ok(block)
    }

    fn verify_block_contents(data: &[u8], expected_values: &[Vec<u8>]) -> bool {
        let num_entries = u16::from_le_bytes([data[0], data[1]]) as usize;

        if num_entries != expected_values.len() {
            println!("Number of entries mismatch. Found: {}, Expected: {}",
                     num_entries, expected_values.len());
            return false;
        }

        // Read offsets
        let mut offsets = Vec::with_capacity(num_entries);
        for i in 0..num_entries {
            let offset_pos = 2 + (i * 2);
            let offset = u16::from_le_bytes([data[offset_pos], data[offset_pos + 1]]);
            offsets.push(offset);
        }

        let entries_start = 2 + (num_entries * 2);

        // Verify each entry
        for (i, expected_value) in expected_values.iter().enumerate() {
            let entry_start = if i == 0 { 0 } else { offsets[i - 1] as usize };
            let entry_end = offsets[i] as usize;

            let entry_data = &data[entries_start + entry_start..entries_start + entry_end];

            if entry_data != expected_value {
                println!("Entry {} mismatch. Found: {:?}, Expected: {:?}",
                         i, entry_data, expected_value);
                return false;
            }
        }

        true
    }

    #[test]
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

        println!("Creating test block");
        let block = create_test_block(&entries).unwrap();

        println!("Creating writer");
        let mut writer = SegmentWriter::new(ctx.handle.clone()).unwrap();

        println!("Writing block");
        assert!(writer.write_block(block).is_ok());

        println!("Shutting down writer");
        writer.shutdown();

        // Wait for writer to finish and properly sync
        println!("Dropping writer to ensure completion");
        drop(writer);

        // Add a small delay to ensure filesystem sync
        thread::sleep(Duration::from_millis(100));

        println!("Reading back data");
        let mut buffer = vec![0u8; BLOCK_SIZE];
        ctx.handle.read_at(0, &mut buffer).unwrap();

        println!("First 20 bytes of buffer: {:?}", &buffer[..20]);

        let values: Vec<Vec<u8>> = entries.iter().map(|(_, value)| value.clone()).collect();
        println!("Expected values len: {}", values[0].len());
        println!("Expected first 20 bytes: {:?}", &values[0][..20]);

        assert!(verify_block_contents(&buffer, &values),
                "Block contents verification failed\nBuffer: {:?}\nExpected values: {:?}",
                &buffer[..20],
                &values
        );
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
            FsError(SegmentFull)
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
        let blocks_per_thread = 5;  // Reduced from 50 to ensure we don't overflow
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
                        Ok(block) => {
                            let mut writer = writer_clone.lock();
                            if writer.write_block(block).is_err() {
                                error_count.fetch_add(1, Relaxed);
                                break; // Exit if we hit an error
                            }
                        },
                        Err(_) => {
                            error_count.fetch_add(1, Relaxed);
                            break;
                        }
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
        assert_eq!(error_count.load(Relaxed), 0, "Some threads encountered errors");
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
        assert!(verify_block_contents(&buffer, &values));
    }
}