use bytes::Bytes;

/// A shared trait for encoding and decoding data in and out of binary formats.
pub trait BinaryMarshaller {
    /// Encode the data structure into a byte array.
    fn encode(self) -> Bytes;
    /// Decode a byte array into the target type.
    fn decode(src: Bytes) -> Self;
    /// The size of the encoded structure in bytes.
    fn encoded_size(&self) -> usize;
}
