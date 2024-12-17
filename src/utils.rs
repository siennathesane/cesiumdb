use bytes::Bytes;

pub(crate) trait Serializer {
    fn serialize_for_memory(&self) -> Bytes;
    fn serialize(&self) -> Bytes;
}

pub(crate) trait Deserializer {
    fn deserialize_from_memory(payload: Bytes) -> Self;
    fn deserialize(payload: Bytes) -> Self;
}
