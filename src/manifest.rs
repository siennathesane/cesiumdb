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

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::SegmentMetadata;
    use crate::utils::{
        Deserializer,
        Serializer,
    };

    #[test]
    fn test_segment_metadata_serialization_roundtrip() {
        let metadata = SegmentMetadata {
            fname: "segment-001.sst".to_string(),
            starting_key: Bytes::from("start-key"),
            seed: 12345,
        };

        let serialized = metadata.serialize();
        let deserialized = SegmentMetadata::deserialize(serialized);

        assert_eq!(deserialized.fname, metadata.fname);
        assert_eq!(deserialized.starting_key, metadata.starting_key);
        assert_eq!(deserialized.seed, metadata.seed);
    }

    #[test]
    fn test_segment_metadata_serialize() {
        let metadata = SegmentMetadata {
            fname: "test.sst".to_string(),
            starting_key: Bytes::from("key"),
            seed: 999,
        };

        let serialized = metadata.serialize();
        let deserialized = SegmentMetadata::deserialize(serialized);

        assert_eq!(deserialized.fname, metadata.fname);
        assert_eq!(deserialized.starting_key, metadata.starting_key);
        assert_eq!(deserialized.seed, metadata.seed);
    }

    #[test]
    fn test_segment_metadata_empty_filename() {
        let metadata = SegmentMetadata {
            fname: String::new(),
            starting_key: Bytes::from("key"),
            seed: 0,
        };

        let serialized = metadata.serialize();
        let deserialized = SegmentMetadata::deserialize(serialized);

        assert_eq!(deserialized.fname, "");
        assert_eq!(deserialized.starting_key, metadata.starting_key);
        assert_eq!(deserialized.seed, metadata.seed);
    }

    #[test]
    fn test_segment_metadata_empty_key() {
        let metadata = SegmentMetadata {
            fname: "segment.sst".to_string(),
            starting_key: Bytes::new(),
            seed: 42,
        };

        let serialized = metadata.serialize();
        let deserialized = SegmentMetadata::deserialize(serialized);

        assert_eq!(deserialized.fname, metadata.fname);
        assert_eq!(deserialized.starting_key.len(), 0);
        assert_eq!(deserialized.seed, metadata.seed);
    }

    #[test]
    fn test_segment_metadata_long_filename() {
        let long_name = "a".repeat(1000);
        let metadata = SegmentMetadata {
            fname: long_name.clone(),
            starting_key: Bytes::from("key"),
            seed: 777,
        };

        let serialized = metadata.serialize();
        let deserialized = SegmentMetadata::deserialize(serialized);

        assert_eq!(deserialized.fname, long_name);
        assert_eq!(deserialized.starting_key, metadata.starting_key);
        assert_eq!(deserialized.seed, metadata.seed);
    }

    #[test]
    fn test_segment_metadata_large_key() {
        let large_key = vec![b'k'; 10000];
        let metadata = SegmentMetadata {
            fname: "segment.sst".to_string(),
            starting_key: Bytes::from(large_key.clone()),
            seed: 123,
        };

        let serialized = metadata.serialize();
        let deserialized = SegmentMetadata::deserialize(serialized);

        assert_eq!(deserialized.fname, metadata.fname);
        assert_eq!(deserialized.starting_key.len(), 10000);
        assert_eq!(deserialized.seed, metadata.seed);
    }

    #[test]
    fn test_segment_metadata_negative_seed() {
        let metadata = SegmentMetadata {
            fname: "segment.sst".to_string(),
            starting_key: Bytes::from("key"),
            seed: -999999,
        };

        let serialized = metadata.serialize();
        let deserialized = SegmentMetadata::deserialize(serialized);

        assert_eq!(deserialized.fname, metadata.fname);
        assert_eq!(deserialized.starting_key, metadata.starting_key);
        assert_eq!(deserialized.seed, -999999);
    }

    #[test]
    fn test_segment_metadata_boundary_seeds() {
        // test with i64::MAX
        let metadata_max = SegmentMetadata {
            fname: "max.sst".to_string(),
            starting_key: Bytes::from("key"),
            seed: i64::MAX,
        };

        let serialized = metadata_max.serialize();
        let deserialized = SegmentMetadata::deserialize(serialized);
        assert_eq!(deserialized.seed, i64::MAX);

        // test with i64::MIN
        let metadata_min = SegmentMetadata {
            fname: "min.sst".to_string(),
            starting_key: Bytes::from("key"),
            seed: i64::MIN,
        };

        let serialized = metadata_min.serialize();
        let deserialized = SegmentMetadata::deserialize(serialized);
        assert_eq!(deserialized.seed, i64::MIN);
    }

    #[test]
    fn test_segment_metadata_special_characters_in_filename() {
        let metadata = SegmentMetadata {
            fname: "segment-2024_01_01-v1.0.sst".to_string(),
            starting_key: Bytes::from("start"),
            seed: 42,
        };

        let serialized = metadata.serialize();
        let deserialized = SegmentMetadata::deserialize(serialized);

        assert_eq!(deserialized.fname, metadata.fname);
    }

    #[test]
    fn test_segment_metadata_unicode_in_key() {
        let metadata = SegmentMetadata {
            fname: "segment.sst".to_string(),
            starting_key: Bytes::from("key-with-émojis-🔑"),
            seed: 123,
        };

        let serialized = metadata.serialize();
        let deserialized = SegmentMetadata::deserialize(serialized);

        assert_eq!(deserialized.starting_key, metadata.starting_key);
    }

    #[test]
    fn test_segment_metadata_serialization_size() {
        let metadata = SegmentMetadata {
            fname: "test.sst".to_string(),
            starting_key: Bytes::from("key"),
            seed: 100,
        };

        let serialized = metadata.serialize();

        // size should be: fname_len(8) + fname + seed(8) + key_len(8) + key
        let expected_size = 8 + "test.sst".len() + 8 + 8 + "key".len();
        assert_eq!(serialized.len(), expected_size);
    }
}
