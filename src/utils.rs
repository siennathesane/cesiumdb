use bytes::{
    Bytes,
    BytesMut,
};

pub(crate) trait Serializer {
    fn serialize_for_memory(&self) -> Bytes;
    fn serialize(&self) -> Bytes;
}

// TODO(@siennathesane): it should be:
// `fn deserialize<D>(payload: Bytes) -> Result<Self, CesiumError>`
pub(crate) trait Deserializer {
    fn deserialize_from_memory(payload: Bytes) -> Self;
    fn deserialize(payload: Bytes) -> Self;
}
