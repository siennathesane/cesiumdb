use bytes::Bytes;

pub(crate) trait Serializer {
    fn serialize_for_memory(&self) -> Bytes;
    fn serialize_for_storage(&self) -> Bytes;
}

pub(crate) trait Deserializer {
    fn deserialize_from_memory(payload: Bytes) -> Self;
    fn deserialize_from_storage(payload: Bytes) -> Self;
}
