use std::{
    sync::{
        atomic::{
            AtomicBool,
            AtomicUsize,
            Ordering::Relaxed,
        },
        Arc,
    },
    thread,
};

use crossbeam_queue::SegQueue;
use parking_lot::{
    Condvar,
    Mutex,
};

use crate::{
    block::{
        Block,
        BLOCK_SIZE,
    },
    errs::CesiumError,
    stats::STATS,
};
use crate::errs::CesiumError::FsError;
use crate::errs::FsError::{SegmentFull, SegmentSizeInvalid};
use crate::fs::Fs;

pub(crate) struct SegmentWriter {
    fs: Arc<Fs>,
    frange_id: u64,
    blocks_left: Arc<AtomicUsize>,
    block_queue: Arc<SegQueue<Block>>,
    segment_full: Arc<AtomicBool>,
    done: Arc<AtomicBool>,
    completion_mutex: Arc<Mutex<()>>,
    completion_condvar: Arc<Condvar>,
}

impl SegmentWriter {
    pub(crate) fn new(fs: Arc<Fs>, segment_size: u64) -> Result<Self, CesiumError> {
        if segment_size % BLOCK_SIZE as u64 != 0 {
            return Err(FsError(SegmentSizeInvalid));
        }

        // Create a new frange for this segment
        let frange_id = fs.create_frange(segment_size)?;

        let blocks_left = Arc::new(AtomicUsize::new(segment_size as usize / BLOCK_SIZE));
        let done = Arc::new(AtomicBool::new(false));
        let segment_full = Arc::new(AtomicBool::new(false));
        let queue = Arc::new(SegQueue::<Block>::new());
        let completion_mutex = Arc::new(Mutex::new(()));
        let completion_condvar = Arc::new(Condvar::new());

        let done_clone = done.clone();
        let queue_clone = queue.clone();
        let segment_full_clone = segment_full.clone();
        let fs_clone = fs.clone();
        let completion_mutex_clone = completion_mutex.clone();
        let completion_condvar_clone = completion_condvar.clone();
        let frange_id_clone = frange_id;

        // Spawn the worker thread
        thread::spawn(move || {
            let mut current_offset = 0;
            let mut frange = match fs_clone.open_frange(frange_id_clone) {
                | Ok(handle) => handle,
                | Err(_) => {
                    // Handle error by notifying waiting threads and returning
                    let _guard = completion_mutex_clone.lock();
                    completion_condvar_clone.notify_one();
                    STATS.current_threads.fetch_sub(1, Relaxed);
                    return;
                },
            };

            loop {
                if let Some(block) = queue_clone.pop() {
                    // Create a buffer for the block data
                    let mut buffer = vec![0u8; BLOCK_SIZE];
                    unsafe {
                        block.finalize(buffer.as_mut_ptr());
                    }

                    // Write the block to the frange
                    if let Err(_) = frange.write_at(current_offset, &buffer) {
                        break;
                    }

                    current_offset += BLOCK_SIZE as u64;
                } else if done_clone.load(Relaxed) {
                    // No more blocks and we're done
                    break;
                }
            }

            // Close the frange before exiting
            let _ = fs_clone.close_frange(frange);

            // Notify completion after frange is closed
            let _guard = completion_mutex_clone.lock();
            completion_condvar_clone.notify_one();
            STATS.current_threads.fetch_sub(1, Relaxed);
        });
        STATS.current_threads.fetch_add(1, Relaxed);

        Ok(Self {
            fs,
            frange_id,
            blocks_left,
            block_queue: queue,
            done,
            segment_full,
            completion_mutex,
            completion_condvar,
        })
    }

    pub(crate) fn write(&mut self, block: Block) -> Result<(), CesiumError> {
        if self.blocks_left.load(Relaxed) == 0 {
            self.segment_full.store(true, Relaxed);
            return Err(FsError(SegmentFull));
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
mod tests {
    use std::{
        fs::OpenOptions,
        path::PathBuf,
        thread,
        time::Duration,
    };

    use memmap2::MmapMut;
    use rand::Rng;
    use tempfile::tempdir;

    use super::*;

    // Helper function to create a test filesystem
    fn create_test_fs() -> (Arc<Fs>, PathBuf) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)
            .unwrap();

        file.set_len(1024 * 1024 * 10) // 10MB
            .unwrap();

        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let fs = Arc::new(Fs::init(mmap).unwrap());

        (fs, path)
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
                return false;
            }
        }

        true
    }

    #[test]
    fn test_basic_write_operation() {
        let (fs, _path) = create_test_fs();
        let segment_size = BLOCK_SIZE as u64 * 10;

        let entries = vec![
            (vec![0u8; 1], vec![1u8; 100]), // Simple small entry
        ];

        // Write data and get frange_id
        let frange_id = {
            let mut writer = SegmentWriter::new(fs.clone(), segment_size).unwrap();
            let block = create_test_block(&entries).unwrap();
            assert!(writer.write(block).is_ok());
            writer.frange_id
        }; // Writer dropped here, cleanup should happen

        // Verify written data
        let mut buffer = vec![0u8; BLOCK_SIZE];
        {
            let frange = fs.open_frange(frange_id).unwrap();
            assert!(frange.read_at(0, &mut buffer).is_ok());
            drop(frange);
        }

        let values: Vec<Vec<u8>> = entries.iter().map(|(_, value)| value.clone()).collect();
        assert!(verify_block_contents(&buffer, &values));

        // Should be able to delete if properly closed
        match fs.delete_frange(frange_id) {
            Ok(_) => {},
            Err(e) => panic!("Failed to delete frange: {:?}", e)
        }
    }

    #[test]
    fn test_segment_full() {
        let (fs, _path) = create_test_fs();
        let segment_size = BLOCK_SIZE as u64 * 2; // Only 2 blocks

        let mut writer = SegmentWriter::new(fs, segment_size).unwrap();

        // Create test entries that fill blocks
        let test_entries = vec![(vec![1u8; 2000], vec![2u8; 1000])]; // Large entries

        // Write blocks until full
        let block1 = create_test_block(&test_entries).unwrap();
        let block2 = create_test_block(&test_entries).unwrap();
        let block3 = create_test_block(&test_entries).unwrap();

        assert!(writer.write(block1).is_ok());
        assert!(writer.write(block2).is_ok());
        assert!(matches!(
            writer.write(block3).unwrap_err(),
            FsError(SegmentFull)
        ));
    }

    #[test]
    fn test_concurrent_writes() {
        let (fs, _path) = create_test_fs();
        let segment_size = BLOCK_SIZE as u64 * 100; // 100 blocks

        let writer = Arc::new(Mutex::new(SegmentWriter::new(fs, segment_size).unwrap()));
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
                    assert!(writer.write(block).is_ok());
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let writer = writer.lock();
        assert_eq!(writer.blocks_left.load(Relaxed), 50); // 100 - (10 threads *
                                                          // 5 blocks)
    }

    #[test]
    fn test_block_queue_ordering() {
        let (fs, _path) = create_test_fs();
        let segment_size = BLOCK_SIZE as u64 * 10;

        let mut writer = SegmentWriter::new(fs.clone(), segment_size).unwrap();
        let mut expected_entries = Vec::new();

        // Write blocks with sequential numbers
        for i in 0..5 {
            let entries = vec![(
                format!("key_{}", i).into_bytes(),
                format!("value_{}", i).into_bytes(),
            )];
            expected_entries.push(entries[0].clone());

            let block = create_test_block(&entries).unwrap();
            assert!(writer.write(block).is_ok());
        }

        // Wait for writes to complete
        thread::sleep(Duration::from_millis(100));
        writer.shutdown();

        // Wait for cleanup
        thread::sleep(Duration::from_millis(100));

        // Read back and verify order
        {
            let frange = fs.open_frange(writer.frange_id).unwrap();
            let mut results = Vec::new();

            for i in 0..5 {
                let mut buffer = vec![0u8; BLOCK_SIZE];
                assert!(frange.read_at(i * BLOCK_SIZE as u64, &mut buffer).is_ok());

                // Extract just the value for verification
                let values = vec![expected_entries[i as usize].1.clone()];
                results.push(verify_block_contents(&buffer, &values));
            }

            // Drop frange before assertions
            drop(frange);

            // Now verify all results
            for result in results {
                assert!(result);
            }
        }
    }

    #[test]
    fn test_stress_write() {
        let (fs, _path) = create_test_fs();
        let segment_size = BLOCK_SIZE as u64 * 1000; // 1000 blocks

        let writer = Arc::new(Mutex::new(SegmentWriter::new(fs, segment_size).unwrap()));
        let mut handles = vec![];

        // Spawn 20 threads writing 50 blocks each
        for thread_id in 0..20 {
            let writer_clone = writer.clone();
            let handle = thread::spawn(move || {
                let mut rng = rand::thread_rng();
                for i in 0..50 {
                    // Create entries with random sizes
                    let key_size = rng.gen_range(1..100);
                    let value_size = rng.gen_range(1..1000);

                    let key = format!("key_{}_{}", thread_id, i).into_bytes();
                    let mut value = vec![0u8; value_size];
                    rng.fill(&mut value[..]);

                    let entries = vec![(key, value)];
                    let block = create_test_block(&entries).unwrap();

                    let mut writer = writer_clone.lock();
                    assert!(writer.write(block).is_ok());

                    if rng.gen_ratio(1, 10) {
                        thread::sleep(Duration::from_millis(1));
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let writer = writer.lock();
        writer.shutdown();

        assert_eq!(writer.blocks_left.load(Relaxed), 1000 - (20 * 50));
    }

    #[test]
    fn test_multiple_entry_block() {
        let (fs, _path) = create_test_fs();
        let segment_size = BLOCK_SIZE as u64 * 10;

        let entries = vec![
            (vec![1u8; 1000], vec![2u8; 1000]),
            (vec![3u8; 500], vec![4u8; 500]),
        ];

        // Get the frange id and write data
        let frange_id = {
            let mut writer = SegmentWriter::new(fs.clone(), segment_size).unwrap();
            let block = create_test_block(&entries).unwrap();
            assert!(writer.write(block).is_ok());
            writer.frange_id
        }; // Writer is dropped here, should handle cleanup

        // Verify written data
        let mut buffer = vec![0u8; BLOCK_SIZE];
        {
            let frange = fs.open_frange(frange_id).unwrap();
            assert!(frange.read_at(0, &mut buffer).is_ok());
            drop(frange);
        }

        let values: Vec<Vec<u8>> = entries.iter().map(|(_, value)| value.clone()).collect();
        assert!(verify_block_contents(&buffer, &values));

        // Should be able to delete if properly closed
        match fs.delete_frange(frange_id) {
            Ok(_) => {},
            Err(e) => panic!("Failed to delete frange: {:?}", e)
        }
    }
}
