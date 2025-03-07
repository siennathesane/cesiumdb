use std::{
    path::PathBuf,
    sync::Arc,
};

use bytes::Bytes;

use crate::{
    errs::{
        SegmentError,
        SegmentError::CantCreateWriter,
    },
    index::Index,
    map::Map,
    segment::{
        BlockType::Value,
        Metadata,
        Segment,
    },
    segment_reader::SegmentReader,
    segment_writer::SegmentWriter,
};

pub(crate) struct SegmentBuilder {
    root: PathBuf,
}

impl SegmentBuilder {
    pub(crate) fn new(path: PathBuf) -> Result<SegmentBuilder, SegmentError> {
        Ok(Self { root: path })
    }

    pub(crate) fn new_segment(
        &self,
        id: u64,
        seed: i64,
        size: u64,
    ) -> Result<Arc<Segment>, SegmentError> {
        let key_segment_id = id;
        let val_segment_id = id + 1;

        let key_path = self.root.join(key_segment_id.to_string());
        let key_mmap = match Map::new(key_path, size) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };

        let val_path = self.root.join(val_segment_id.to_string());
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

    pub(crate) fn open(&self, id: u64) -> Result<Arc<Segment>, SegmentError> {
        let key_segment_id = id;
        let val_segment_id = id + 1;

        let key_path = self.root.join(key_segment_id.to_string());
        let key_mmap = match Map::open(key_path) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };

        // load the metadata from the end of the key mmap
        let mdata_size = size_of::<Metadata>();
        let key_mdata_payload = key_mmap[key_mmap.len() - mdata_size..key_mmap.len()].as_ref();
        let key_metadata = Metadata::from(Bytes::copy_from_slice(key_mdata_payload));

        let key_index_payload = key_mmap[key_mmap.len() - key_metadata.index_start()..
            key_mmap.len() - key_metadata.index_size()]
            .as_ref();
        let key_index = Index::from(Bytes::copy_from_slice(key_index_payload));

        let val_path = self.root.join(val_segment_id.to_string());
        let val_mmap = match Map::open(val_path) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };

        let val_mdata_payload = val_mmap[val_mmap.len() - mdata_size..val_mmap.len()].as_ref();
        let val_metadata = Metadata::from(Bytes::copy_from_slice(val_mdata_payload));

        let val_index_payload = val_mmap[val_mmap.len() - val_metadata.index_start()..
            val_mmap.len() - val_metadata.index_size()]
            .as_ref();
        let val_index = Index::from(Bytes::copy_from_slice(val_index_payload));

        let key_handle = Arc::new(key_mmap);
        let val_handle = Arc::new(val_mmap);

        Segment::open(
            key_handle,
            key_index,
            key_metadata.id(),
            val_handle,
            val_index,
            val_metadata.id(),
        )
    }
}
