use std::sync::Arc;

use crate::{
    errs::{
    },
    fs::{
        Fs,
    },
    segment::{
        BlockType::{
            Key,
            Value,
        },
        Segment,
    },
};
use crate::errs::SegmentError;
use crate::errs::SegmentError::{CantCreateFRange, CantCreateWriter, CantOpenFRange};
use crate::segment_writer::SegmentWriter;

pub(crate) struct SegmentBuilder {
    fs: Arc<Fs>,
}

impl SegmentBuilder {
    pub(crate) fn new(fs: Arc<Fs>) -> Self {
        Self { fs }
    }

    pub(crate) fn new_segment(&self, id: u64, seed: i64, size: u64) -> Result<Arc<Segment>, SegmentError> {
        let key_segment_id = id;
        let val_segment_id = id + 1;

        let key_fid = match self.fs.create_frange(size) {
            | Ok(handle) => handle,
            | Err(e) => return Err(CantCreateFRange(Key, key_segment_id, e)),
        };

        let val_fid = match self.fs.create_frange(size) {
            | Ok(handle) => handle,
            | Err(e) => return Err(CantCreateFRange(Value, val_segment_id, e)),
        };

        let key_handle = match self.fs.open_frange(key_fid) {
            | Ok(v) => v,
            | Err(e) => return Err(CantOpenFRange(Key, key_segment_id, e)),
        };

        let val_handle = match self.fs.open_frange(val_fid) {
            | Ok(v) => v,
            | Err(e) => return Err(CantOpenFRange(Value, val_segment_id, e)),
        };
        
        let key_seg_writer = match SegmentWriter::new(Arc::new(key_handle)) {
            Ok(v) => v,
            Err(e) => return Err(CantCreateWriter(Value, key_segment_id)),
        };

        let val_seg_writer = match SegmentWriter::new(Arc::new(val_handle)) {
            Ok(v) => v,
            Err(e) => return Err(CantCreateWriter(Value, val_segment_id)),
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
