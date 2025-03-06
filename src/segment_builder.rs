use std::{
    path::PathBuf,
    sync::Arc,
};

use crate::{
    errs::{
        SegmentError,
        SegmentError::CantCreateWriter,
    },
    map::Map,
    segment::{
        BlockType::Value,
        Segment,
    },
    segment_reader::SegmentReader,
    segment_writer::SegmentWriter,
};

pub(crate) struct SegmentBuilder {
    path: PathBuf,
}

impl SegmentBuilder {
    pub(crate) fn new(path: PathBuf) -> Result<SegmentBuilder, SegmentError> {
        Ok(Self { path })
    }

    pub(crate) fn new_segment(
        &self,
        id: u64,
        seed: i64,
        size: u64,
    ) -> Result<Arc<Segment>, SegmentError> {
        let key_segment_id = id;
        let val_segment_id = id + 1;

        let key_path = self.path.join(key_segment_id.to_string());
        let key_mmap = match Map::new(key_path, size) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };

        let val_path = self.path.join(val_segment_id.to_string());
        let val_mmap = match Map::new(val_path, size) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };

        let key_handle = Arc::new(key_mmap);
        let val_handle = Arc::new(val_mmap);

        let key_seg_writer = match SegmentWriter::new(key_handle.clone()) {
            | Ok(v) => v,
            | Err(e) => return Err(CantCreateWriter(Value, key_segment_id)),
        };

        let val_seg_writer = match SegmentWriter::new(val_handle.clone()) {
            | Ok(v) => v,
            | Err(e) => return Err(CantCreateWriter(Value, val_segment_id)),
        };

        let segment = Arc::new(Segment::new(
            key_segment_id,
            val_segment_id,
            seed,
            key_seg_writer,
            val_seg_writer,
        ));

        Ok(segment)
    }
}
