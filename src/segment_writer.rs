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
            Ordering::Relaxed,
        },
    },
};

use bytes::Bytes;
use parking_lot::Mutex;
use tracing::instrument;

use crate::{
    block::BLOCK_SIZE,
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
    use tempfile::tempdir;

    use super::*;
    use crate::block::BLOCK_SIZE;

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
}
