use bytes::{
    Bytes,
    BytesMut,
};

pub trait Serializer {
    fn serialize(&self) -> Bytes;
}

// TODO(@siennathesane): it should be:
// `fn deserialize<D>(payload: Bytes) -> Result<Self, CesiumError>`
pub trait Deserializer {
    fn deserialize(payload: Bytes) -> Self;
}
