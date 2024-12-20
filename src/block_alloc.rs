// use std::sync::Arc;
// 
// use bytes::{
//     Bytes,
//     BytesMut,
// };
// use crossbeam_queue::SegQueue;
// use parking_lot::{Mutex, RwLock};
// 
// use crate::{
//     block::Block,
//     errs::CesiumError,
//     index::SegmentIndex,
//     segment_reader::SegmentReader,
//     segment_writer::SegmentWriter,
// };
// 
// pub(crate) struct BlockAllocator {
//     key_writer: Arc<SegmentWriter>,
//     key_reader: Arc<SegmentReader>,
//     val_writer: Arc<SegmentWriter>,
//     val_reader: Arc<SegmentReader>,
//     queue: SegQueue<Block>,
//     idx: Arc<SegmentIndex>,
// 
//     known_ns: Vec<u64>,
// }
// 
// impl BlockAllocator {
//     pub fn new(
//         id: u64,
//         key_writer: Arc<SegmentWriter>,
//         key_reader: Arc<SegmentReader>,
//         val_writer: Arc<SegmentWriter>,
//         val_reader: Arc<SegmentReader>,
//     ) -> Self {
//         Self {
//             key_writer,
//             key_reader,
//             val_writer,
//             val_reader,
//             queue: SegQueue::new(),
//             // TODO(@siennathesane): figure out what to do with the seed config
//             idx: SegmentIndex::new(id, 0),
//             known_ns: Vec::new(),
//         }
//     }
// 
//     pub(crate) fn write(&mut self, key: &[u8], val: &[u8]) -> Result<(), CesiumError> {
//         let ns = u64::from_le_bytes(key[0..8].try_into().unwrap());
// 
//         // if the namespace is not known, add it to the list of known namespaces
//         match self.known_ns.binary_search_by(|f| f.cmp(&ns)) {
//             | Ok(_) => (),
//             | Err(idx) => {
//                 self.known_ns.insert(idx, ns);
//                 
//             },
//         };
// 
//         Ok(())
//     }
// 
//     pub(crate) fn read(&self, key: &[u8]) -> Result<Option<Bytes>, CesiumError> {
//         Ok(None)
//     }
// }
