use std::cmp::Ordering;
use std::io::Cursor;
use crc32fast::Hasher;
use memmap2::MmapMut;
use rkyv::{to_bytes, Archive, Deserialize, Serialize};
use rkyv::rancor::Error;

pub(crate) struct SSTable {
    fh: Cursor<MmapMut>,
    buf: Vec<u8>,
}

impl SSTable {
    pub fn new(fh: Cursor<MmapMut>) -> Self {
        SSTable {
            fh,
            buf: vec![]
        }
    }

    pub fn append(&self, record: &Record) {
        let _bytes = to_bytes::<Error>(record).unwrap();

        let mut hasher = Hasher::new();
        hasher.update(_bytes.clone().as_ref());
        let cksum = hasher.finalize();
    }
}

#[derive(Archive,Serialize,Deserialize)]
struct InternalRecord {
    length: u64,
    checksum: u32,
    value: Record,
}

impl Ord for InternalRecord {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.value.ns.cmp(&other.value.ns) {
            Ordering::Equal => {
                self.value.key.cmp(&other.value.key)
            }
            other => other,
        }
    }
}

impl PartialOrd for InternalRecord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for InternalRecord {
    fn eq(&self, other: &Self) -> bool {
        self.value.ns == other.value.ns && self.value.key == other.value.key
    }
}

impl Eq for InternalRecord {}

#[derive(Archive,Serialize,Deserialize)]
pub struct Record {
    ts: u64,
    ns: u64,
    key: Vec<u8>,
    value: Vec<u8>
}
