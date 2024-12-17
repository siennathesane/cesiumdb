use std::sync::Arc;

use bytes::Bytes;
use crossbeam_queue::SegQueue;

use crate::{
    block::Block,
    errs::CesiumError,
    segment_reader::SegmentReader,
    segment_writer::SegmentWriter,
};

pub(crate) struct BlockAllocator {
    key_writer: Arc<SegmentWriter>,
    key_reader: Arc<SegmentReader>,
    val_writer: Arc<SegmentWriter>,
    val_reader: Arc<SegmentReader>,
    queue: SegQueue<Block>,
}

impl BlockAllocator {
    pub fn new(
        key_writer: Arc<SegmentWriter>,
        key_reader: Arc<SegmentReader>,
        val_writer: Arc<SegmentWriter>,
        val_reader: Arc<SegmentReader>,
    ) -> Self {
        Self {
            key_writer,
            key_reader,
            val_writer,
            val_reader,
            queue: SegQueue::new(),
        }
    }

    pub(crate) fn write<K: AsRef<[u8]>, V: AsRef<[u8]>>(
        &self,
        key: K,
        val: V,
    ) -> Result<(), CesiumError> {
        Ok(())
    }

    pub(crate) fn read<K: AsRef<[u8]>>(&self, key: K) -> Result<Option<Bytes>, CesiumError> {
        Ok(None)
    }
}
