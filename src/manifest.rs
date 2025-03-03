use bytes::{
    Buf,
    BufMut,
    Bytes,
    BytesMut,
};

use crate::utils::{
    Deserializer,
    Serializer,
};

struct SegmentMetadata {
    fname: String,
    starting_key: Bytes,
    seed: i64,
}

impl Serializer for SegmentMetadata {
    fn serialize_for_memory(&self) -> Bytes {
        self.serialize()
    }

    fn serialize(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(100);
        buf.put_u64(self.fname.len() as u64);
        buf.put_slice(self.fname.as_bytes());
        buf.put_i64(self.seed);
        buf.put_u64(self.starting_key.len() as u64);
        buf.put_slice(&self.starting_key);
        buf.freeze()
    }
}

impl Deserializer for SegmentMetadata {
    fn deserialize_from_memory(payload: Bytes) -> Self {
        Self::deserialize(payload)
    }

    fn deserialize(payload: Bytes) -> Self {
        let mut bytes = payload;
        let fname_len = bytes.get_u64() as usize;
        let fname = bytes.split_to(fname_len);
        let seed = bytes.get_i64();
        let key_len = bytes.get_u64() as usize;
        let starting_key = bytes.split_to(key_len);
        Self {
            fname: String::from_utf8(fname.to_vec()).unwrap(),
            seed,
            starting_key,
        }
    }
}
