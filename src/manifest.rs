// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

//! Manifest file format for crash recovery.
//!
//! The manifest is a write-ahead log that records all LSM-tree state changes
//! (VersionEdits). On startup, the manifest is replayed to reconstruct the
//! VersionSet.
//!
//! File format:
//! ```text
//! MANIFEST
//! ├─ File Header (48 bytes)
//! │  ├─ Magic: 0x43455349 ("CESI")
//! │  ├─ Version: 1 (u32)
//! │  ├─ Flags: compression/checksum bits (u16)
//! │  ├─ Created HLC timestamp (u64)
//! │  ├─ Edit count (u64)
//! │  └─ Padding to 48 bytes
//! │
//! ├─ Edit Entry 1
//! │  ├─ Entry header
//! │  │  ├─ Type discriminant (u8)
//! │  │  ├─ Payload length (varint)
//! │  │  └─ CRC32 checksum (u32)
//! │  └─ Payload (VersionEdit data)
//! │
//! ├─ Edit Entry 2...
//! ```

use bytes::{
    Buf,
    BufMut,
    Bytes,
    BytesMut,
};

use crate::errs::ManifestError;

/// Magic number for manifest files: "CESI" in ASCII
pub const MANIFEST_MAGIC: u32 = 0x43455349;

/// Current manifest format version
pub const MANIFEST_VERSION: u32 = 1;

/// Manifest file header (48 bytes)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestHeader {
    pub magic: u32,
    pub version: u32,
    pub flags: u16,
    pub created_hlc: u64,
    pub edit_count: u64,
}

impl ManifestHeader {
    /// Create a new manifest header
    pub fn new(created_hlc: u64) -> Self {
        Self {
            magic: MANIFEST_MAGIC,
            version: MANIFEST_VERSION,
            flags: 0x0001, // bit 0: CRC enabled
            created_hlc,
            edit_count: 0,
        }
    }

    /// Encode header to bytes (48 bytes)
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(48);
        buf.put_u32_le(self.magic);
        buf.put_u32_le(self.version);
        buf.put_u16_le(self.flags);
        buf.put_u16_le(0); // reserved
        buf.put_u64_le(self.created_hlc);
        buf.put_u64_le(self.edit_count);
        // Pad to 48 bytes
        buf.put_bytes(0, 48 - buf.len());
        buf.freeze()
    }

    /// Decode header from bytes
    pub fn decode(mut data: Bytes) -> Result<Self, ManifestError> {
        if data.len() < 48 {
            return Err(ManifestError::CorruptedHeader);
        }

        let magic = data.get_u32_le();
        if magic != MANIFEST_MAGIC {
            return Err(ManifestError::InvalidMagic(magic));
        }

        let version = data.get_u32_le();
        if version != MANIFEST_VERSION {
            return Err(ManifestError::UnsupportedVersion(version));
        }

        let flags = data.get_u16_le();
        let _reserved = data.get_u16_le();
        let created_hlc = data.get_u64_le();
        let edit_count = data.get_u64_le();

        Ok(Self {
            magic,
            version,
            flags,
            created_hlc,
            edit_count,
        })
    }

    /// Check if CRC is enabled
    pub fn crc_enabled(&self) -> bool {
        (self.flags & 0x0001) != 0
    }
}

/// A single edit entry in the manifest
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditEntry {
    pub edit_type: u8,
    pub payload: Bytes,
    pub crc32: u32,
}

impl EditEntry {
    /// Create a new edit entry
    pub fn new(edit_type: u8, payload: Bytes) -> Self {
        let crc32 = crc32fast::hash(&payload);
        Self {
            edit_type,
            payload,
            crc32,
        }
    }

    /// Encode entry to bytes
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(1 + 5 + 4 + self.payload.len());
        buf.put_u8(self.edit_type);
        put_varint(&mut buf, self.payload.len() as u64);
        buf.put_u32_le(self.crc32);
        buf.put_slice(&self.payload);
        buf.freeze()
    }

    /// Decode entry from bytes
    pub fn decode(mut data: Bytes) -> Result<(Self, usize), ManifestError> {
        if data.is_empty() {
            return Err(ManifestError::CorruptedHeader);
        }

        let edit_type = data.get_u8();
        let (payload_len, varint_bytes) = match read_varint(&mut data) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };

        if data.len() < 4 {
            return Err(ManifestError::CorruptedHeader);
        }
        let crc32 = data.get_u32_le();

        if data.len() < payload_len as usize {
            return Err(ManifestError::CorruptedHeader);
        }
        let payload = data.split_to(payload_len as usize);

        // Verify CRC
        let computed_crc = crc32fast::hash(&payload);
        if computed_crc != crc32 {
            return Err(ManifestError::CrcMismatch {
                expected: crc32,
                actual: computed_crc,
            });
        }

        let total_bytes = 1 + varint_bytes + 4 + payload_len as usize;
        Ok((
            Self {
                edit_type,
                payload,
                crc32,
            },
            total_bytes,
        ))
    }
}

/// Write a varint (LEB128-style variable-length integer)
pub fn put_varint(buf: &mut BytesMut, mut value: u64) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buf.put_u8(byte);
        if value == 0 {
            break;
        }
    }
}

/// Read a varint, returning (value, bytes_consumed)
pub fn read_varint(data: &mut Bytes) -> Result<(u64, usize), ManifestError> {
    let mut result = 0u64;
    let mut shift = 0;
    let mut bytes_read = 0;

    loop {
        if data.is_empty() {
            return Err(ManifestError::CorruptedHeader);
        }
        if bytes_read >= 10 {
            return Err(ManifestError::CorruptedHeader);
        }

        let byte = data.get_u8();
        bytes_read += 1;

        result |= ((byte & 0x7f) as u64) << shift;
        if (byte & 0x80) == 0 {
            return Ok((result, bytes_read));
        }
        shift += 7;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_header_encode_decode() {
        let header = ManifestHeader::new(12345);
        let encoded = header.encode();
        assert_eq!(encoded.len(), 48);

        let decoded = ManifestHeader::decode(encoded).unwrap();
        assert_eq!(decoded.magic, MANIFEST_MAGIC);
        assert_eq!(decoded.version, MANIFEST_VERSION);
        assert_eq!(decoded.created_hlc, 12345);
        assert!(decoded.crc_enabled());
    }

    #[test]
    fn test_manifest_header_invalid_magic() {
        let mut buf = BytesMut::with_capacity(48);
        buf.put_u32_le(0xdeadbeef); // wrong magic
        buf.put_bytes(0, 44);

        let result = ManifestHeader::decode(buf.freeze());
        assert!(matches!(
            result,
            Err(ManifestError::InvalidMagic(0xdeadbeef))
        ));
    }

    #[test]
    fn test_manifest_header_too_short() {
        let buf = BytesMut::with_capacity(10);
        let result = ManifestHeader::decode(buf.freeze());
        assert!(matches!(result, Err(ManifestError::CorruptedHeader)));
    }

    #[test]
    fn test_edit_entry_encode_decode() {
        let payload = Bytes::from("test payload");
        let entry = EditEntry::new(0x01, payload.clone());

        let encoded = entry.encode();
        let (decoded, bytes_read) = EditEntry::decode(encoded.clone()).unwrap();

        assert_eq!(decoded.edit_type, 0x01);
        assert_eq!(decoded.payload, payload);
        assert_eq!(decoded.crc32, entry.crc32);
        assert_eq!(bytes_read, encoded.len());
    }

    #[test]
    fn test_edit_entry_crc_mismatch() {
        let payload = Bytes::from("test");
        let entry = EditEntry::new(0x01, payload);

        let mut encoded = BytesMut::from(entry.encode().as_ref());
        // Corrupt the CRC
        encoded[2] ^= 0xff;

        let result = EditEntry::decode(encoded.freeze());
        assert!(matches!(result, Err(ManifestError::CrcMismatch { .. })));
    }

    #[test]
    fn test_varint_roundtrip() {
        let test_values = vec![0, 1, 127, 128, 255, 256, 16383, 16384, u64::MAX];

        for value in test_values {
            let mut buf = BytesMut::new();
            put_varint(&mut buf, value);

            let mut bytes = buf.freeze();
            let (decoded, _) = read_varint(&mut bytes).unwrap();
            assert_eq!(decoded, value);
        }
    }

    #[test]
    fn test_varint_sizes() {
        // 1 byte: 0-127
        let mut buf = BytesMut::new();
        put_varint(&mut buf, 127);
        assert_eq!(buf.len(), 1);

        // 2 bytes: 128-16383
        let mut buf = BytesMut::new();
        put_varint(&mut buf, 128);
        assert_eq!(buf.len(), 2);

        // 10 bytes: u64::MAX
        let mut buf = BytesMut::new();
        put_varint(&mut buf, u64::MAX);
        assert_eq!(buf.len(), 10);
    }

    #[test]
    fn test_varint_too_long() {
        // Create a varint that's too long (11 bytes)
        let mut buf = BytesMut::new();
        for _ in 0..11 {
            buf.put_u8(0x80);
        }

        let mut bytes = buf.freeze();
        let result = read_varint(&mut bytes);
        assert!(matches!(result, Err(ManifestError::CorruptedHeader)));
    }

    #[test]
    fn test_varint_truncated() {
        let mut buf = BytesMut::new();
        buf.put_u8(0x80); // continuation bit set, but no more bytes

        let mut bytes = buf.freeze();
        let result = read_varint(&mut bytes);
        assert!(matches!(result, Err(ManifestError::CorruptedHeader)));
    }

    #[test]
    fn test_edit_entry_empty_payload() {
        let entry = EditEntry::new(0x05, Bytes::new());
        let encoded = entry.encode();
        let (decoded, _) = EditEntry::decode(encoded).unwrap();

        assert_eq!(decoded.edit_type, 0x05);
        assert_eq!(decoded.payload.len(), 0);
    }

    #[test]
    fn test_edit_entry_large_payload() {
        let large_payload = Bytes::from(vec![0xab; 10000]);
        let entry = EditEntry::new(0x02, large_payload.clone());

        let encoded = entry.encode();
        let (decoded, _) = EditEntry::decode(encoded).unwrap();

        assert_eq!(decoded.payload, large_payload);
    }

    #[test]
    fn test_manifest_header_crc_flag() {
        let mut header = ManifestHeader::new(0);
        assert!(header.crc_enabled());

        header.flags = 0x0000;
        assert!(!header.crc_enabled());

        header.flags = 0x0001;
        assert!(header.crc_enabled());
    }
}
