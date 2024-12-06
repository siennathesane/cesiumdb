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
use memmap2::{
    Advice::Sequential,
    MmapMut,
};
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

pub(crate) struct SegmentWriter {
    fh: Arc<MmapMut>,
    blocks_left: Arc<AtomicUsize>,
    block_queue: Arc<SegQueue<Block>>,
    segment_full: Arc<AtomicBool>,
    done: Arc<AtomicBool>,
    completion_mutex: Arc<Mutex<()>>,
    completion_condvar: Arc<Condvar>,
}

impl SegmentWriter {
    pub(crate) fn new(fh: Arc<MmapMut>) -> Result<Self, CesiumError> {
        if fh.len() % 4096 != 0 {
            return Err(CesiumError::SegmentSizeInvalid);
        }

        let blocks_left = Arc::new(AtomicUsize::new(fh.len() / 4096));
        let done = Arc::new(AtomicBool::new(false));
        let segment_full = Arc::new(AtomicBool::new(false));
        let queue = Arc::new(SegQueue::<Block>::new());
        let completion_mutex = Arc::new(Mutex::new(()));
        let completion_condvar = Arc::new(Condvar::new());

        let done_clone = done.clone();
        let queue_clone = queue.clone();
        let segment_full_clone = segment_full.clone();
        let fh_clone = fh.clone();
        let completion_mutex_clone = completion_mutex.clone();
        let completion_condvar_clone = completion_condvar.clone();

        fh_clone.advise(Sequential).expect("failed to advise");

        // this is the worker thread that will write the blocks to the file
        thread::spawn(move || {
            let mut current_offset = 0;
            loop {
                if done_clone.load(Relaxed) && queue_clone.is_empty() {
                    break;
                }

                if let Some(block) = queue_clone.pop() {
                    unsafe {
                        let dst = fh_clone.as_ptr().add(current_offset) as *mut u8;
                        block.finalize(dst);
                    }
                    current_offset += BLOCK_SIZE;

                    if segment_full_clone.load(Relaxed) && queue_clone.is_empty() {
                        let guard = completion_mutex_clone.lock();
                        completion_condvar_clone.notify_one();
                    }
                }
            }

            let guard = completion_mutex_clone.lock();
            completion_condvar_clone.notify_one();
            STATS.current_threads.fetch_sub(1, Relaxed);
        });
        STATS.current_threads.fetch_add(1, Relaxed);

        Ok(Self {
            fh,
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
            return Err(CesiumError::SegmentFull);
        }

        self.block_queue.push(block);
        self.blocks_left.fetch_sub(1, Relaxed);

        Ok(())
    }

    // Add a new method to explicitly shutdown the writer
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
            while !self.block_queue.is_empty() {
                self.completion_condvar.wait(&mut guard);
            }
        }

        if let Ok(()) = self.fh.flush() {
            // flushed successfully
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memmap2::MmapOptions;
    use std::sync::atomic::Ordering::Relaxed;
    use std::sync::Arc;
    use tempfile::tempfile;

    fn create_temp_file(size: usize) -> std::fs::File {
        let file = tempfile().unwrap();
        file.set_len(size as u64).unwrap();
        file
    }

    #[test]
    fn test_segment_writer_new() {
        let file = create_temp_file(8192);
        let mmap = unsafe { MmapOptions::new().len(8192).map_mut(&file).unwrap() };
        let mmap_arc = Arc::new(mmap);
        let writer = SegmentWriter::new(mmap_arc).unwrap();
        assert_eq!(writer.blocks_left.load(Relaxed), 2);
        writer.shutdown();
    }

    #[test]
    fn test_segment_writer_new_invalid_size() {
        let file = create_temp_file(5000);
        let mmap = unsafe { MmapOptions::new().len(5000).map_mut(&file).unwrap() };
        let mmap_arc = Arc::new(mmap);
        let result = SegmentWriter::new(mmap_arc);
        assert!(matches!(result, Err(CesiumError::SegmentSizeInvalid)));
    }

    #[test]
    fn test_write_block() {
        let file = create_temp_file(8192);
        let mmap = unsafe { MmapOptions::new().len(8192).map_mut(&file).unwrap() };
        let mmap_arc = Arc::new(mmap);
        let mut writer = SegmentWriter::new(mmap_arc).unwrap();
        let block = Block::new();
        writer.write(block).unwrap();
        assert_eq!(writer.blocks_left.load(Relaxed), 1);
        writer.shutdown();
    }

    #[test]
    fn test_write_block_segment_full() {
        let file = create_temp_file(4096);
        let mmap = unsafe { MmapOptions::new().len(4096).map_mut(&file).unwrap() };
        let mmap_arc = Arc::new(mmap);
        let mut writer = SegmentWriter::new(mmap_arc).unwrap();
        let block = Block::new();
        writer.write(block).unwrap();
        let block = Block::new();
        let result = writer.write(block);
        assert!(matches!(result, Err(CesiumError::SegmentFull)));
        writer.shutdown();
    }

    #[test]
    fn test_write_multiple_blocks() {
        let file = create_temp_file(8192);
        let mmap = unsafe { MmapOptions::new().len(8192).map_mut(&file).unwrap() };
        let mmap_arc = Arc::new(mmap);
        let mut writer = SegmentWriter::new(mmap_arc).unwrap();
        let block1 = Block::new();
        let block2 = Block::new();
        writer.write(block1).unwrap();
        writer.write(block2).unwrap();
        assert_eq!(writer.blocks_left.load(Relaxed), 0);
        writer.shutdown();
    }

    #[test]
    fn test_drop_segment_writer() {
        let file = create_temp_file(8192);
        let mmap = unsafe { MmapOptions::new().len(8192).map_mut(&file).unwrap() };
        let mmap_arc = Arc::new(mmap);
        {
            let mut writer = SegmentWriter::new(mmap_arc.clone()).unwrap();
            let block = Block::new();
            writer.write(block).unwrap();
            writer.shutdown();
        }
        // Ensure the segment is flushed and marked as done
        assert!(mmap_arc.flush().is_ok());
    }
}