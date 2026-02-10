use std::{
    path::PathBuf,
    sync::Arc,
};

use bytes::Bytes;

use crate::{
    errs::{
        SegmentError,
        SegmentError::{
            CantCreateWriter,
            InvalidSize,
        },
    },
    index::{
        Index,
        MIN_INDEX_SIZE,
    },
    map::Map,
    segment::{
        BlockType::Value,
        Metadata,
        Segment,
    },
    segment_reader::SegmentReader,
    segment_writer::SegmentWriter,
};

pub(crate) struct SegmentBuilder {
    root: PathBuf,
}

impl SegmentBuilder {
    pub(crate) fn new(path: PathBuf) -> Result<SegmentBuilder, SegmentError> {
        Ok(Self { root: path })
    }

    pub(crate) fn new_segment(
        &self,
        id: u64,
        seed: i64,
        size: u64,
    ) -> Result<Arc<Segment>, SegmentError> {
        if size == 0 {
            return Err(InvalidSize);
        }

        let key_segment_id = id;
        let val_segment_id = id + 1;

        let key_path = self.root.join(key_segment_id.to_string());
        let key_mmap = match Map::new(key_path, size) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };

        let val_path = self.root.join(val_segment_id.to_string());
        let val_mmap = match Map::new(val_path, size) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };

        let key_handle = Arc::new(key_mmap);
        let val_handle = Arc::new(val_mmap);

        let key_seg_writer = match SegmentWriter::new(key_handle.clone()) {
            | Ok(v) => v,
            | Err(e) => return Err(CantCreateWriter(Value, key_segment_id)),
        };

        let val_seg_writer = match SegmentWriter::new(val_handle.clone()) {
            | Ok(v) => v,
            | Err(e) => return Err(CantCreateWriter(Value, val_segment_id)),
        };

        let segment = Arc::new(Segment::new(
            key_segment_id,
            val_segment_id,
            seed,
            key_seg_writer,
            val_seg_writer,
        ));

        Ok(segment)
    }

    pub(crate) fn open(&self, id: u64) -> Result<Arc<Segment>, SegmentError> {
        let key_segment_id = id;
        let val_segment_id = id + 1;

        let key_path = self.root.join(key_segment_id.to_string());

        // Check if file exists and has content
        if !key_path.exists() {
            return Err(SegmentError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Key segment file not found: {}", key_path.display()),
            )));
        }

        let key_mmap = match Map::open(key_path) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };

        // Get file size for validation
        let key_file_size = key_mmap.len();

        if key_file_size < 32 {
            return Err(SegmentError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Key segment file too small: size={}", key_file_size),
            )));
        }

        // load the metadata from the end of the key mmap
        let mdata_size = size_of::<Metadata>();
        let metadata_offset = key_file_size - mdata_size;

        let key_metadata = match key_mmap.read_range(metadata_offset..key_file_size, |slice| {
            Metadata::from(Bytes::copy_from_slice(slice))
        }) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };

        // Validate index location
        let index_start = key_metadata.index_start();
        let index_size = key_metadata.index_size();
        let block_count = key_metadata.block_count();

        // Allow empty segments (block_count=0, index_start=0)
        // but reject malformed metadata (index_start=0 with blocks, or index_size=0)
        if block_count > 0 && index_start == 0 {
            return Err(SegmentError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Invalid index location: segment has {} blocks but index_start=0",
                    block_count
                ),
            )));
        }

        if index_size == 0 {
            return Err(SegmentError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid index: index_size=0".to_string(),
            )));
        }

        // For non-empty segments, validate index bounds
        if block_count > 0 {
            if index_start >= key_file_size || index_start + index_size > key_file_size {
                return Err(SegmentError::IoError(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Index location out of bounds: start={}, size={}, file_size={}",
                        index_start, index_size, key_file_size
                    ),
                )));
            }
        }

        // Read and deserialize the index
        let mut key_index = if block_count == 0 {
            // Empty segment - create a new empty index instead of reading from file
            // (index_start=0 for empty segments, so we can't read from there)
            Index::new(key_metadata.id(), 0)
        } else {
            match key_mmap.read_range(
                index_start as usize..(index_start + index_size) as usize,
                |key_index_payload| {
                    if key_index_payload.len() < MIN_INDEX_SIZE {
                        return Err(SegmentError::IoError(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("Index data too small: {}", key_index_payload.len()),
                        )));
                    }

                    Ok(Index::from(Bytes::copy_from_slice(key_index_payload)))
                },
            ) {
                | Ok(result) => match result {
                    | Ok(v) => v,
                    | Err(e) => return Err(e),
                },
                | Err(e) => return Err(e),
            }
        };

        // Update the index's num_blocks from the metadata (the index is deserialized
        // with whatever was saved, but the authoritative block count is in the
        // metadata)
        key_index.set_num_blocks(key_metadata.block_count() as u64);

        // Repeat similar process for value segment
        let val_path = self.root.join(val_segment_id.to_string());

        if !val_path.exists() {
            return Err(SegmentError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Value segment file not found: {}", val_path.display()),
            )));
        }

        let val_mmap = match Map::open(val_path) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };

        let val_file_size = val_mmap.len();

        if val_file_size < 32 {
            return Err(SegmentError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Value segment file too small: size={}", val_file_size),
            )));
        }

        let val_metadata_offset = val_file_size - mdata_size;

        let val_metadata = match val_mmap.read_range(val_metadata_offset..val_file_size, |slice| {
            Metadata::from(Bytes::copy_from_slice(slice))
        }) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };

        // Value segments no longer have indices - value locations are stored in key
        // metadata So we skip reading the value index entirely

        let key_handle = Arc::new(key_mmap);
        let val_handle = Arc::new(val_mmap);
        let val_block_count = val_metadata.block_count() as u64;

        Segment::open(
            key_handle,
            key_index,
            key_metadata.id(),
            val_handle,
            val_metadata.id(),
            val_block_count,
        )
    }
}

#[cfg(test)]
mod tests {
    use rand::{
        Rng,
        SeedableRng,
        random_range,
        rng,
    };
    use tempfile::tempdir;

    use super::*;
    use crate::{
        block::BLOCK_SIZE,
        hlc::{
            HLC,
            HybridLogicalClock,
        },
        index::Index,
        keypair,
        keypair::{
            DEFAULT_NS,
            Key,
            KeyBytes,
            ValueBytes,
        },
        segment::Segment,
        utils::Serializer,
    };

    /// Test helper to create a temporary directory for segment files
    fn create_temp_dir() -> tempfile::TempDir {
        tempdir().expect("failed to create temporary directory")
    }

    /// Test helper to generate test key-value pairs with timestamps
    fn generate_test_data(count: usize) -> Vec<(Key<Bytes>, ValueBytes)> {
        let clock = HybridLogicalClock::new();
        let mut rng = rng();
        let mut result = Vec::with_capacity(count);

        for i in 0..count {
            let ns = rng.random_range(DEFAULT_NS..100);
            let key = KeyBytes::new(
                ns,
                Bytes::copy_from_slice(format!("key_{:05}", i).as_bytes()),
                clock.time(),
            );

            let val_size = rng.random_range(10..50000);
            let random_data: Vec<u8> = (0..val_size).map(|_| rng.random::<u8>()).collect();

            let val = ValueBytes::new(ns, Bytes::copy_from_slice(random_data.as_slice()));

            result.push((key, val));
        }

        sort_by_key(result)
    }

    fn sort_by_key(mut pairs: Vec<(KeyBytes, ValueBytes)>) -> Vec<(KeyBytes, ValueBytes)> {
        pairs.sort_by(|(key_a, _), (key_b, _)| {
            // Sort by namespace first
            match key_a.ns().cmp(&key_b.ns()) {
                | std::cmp::Ordering::Equal => {
                    // If namespace is equal, sort by key content
                    key_a.as_bytes().cmp(&key_b.as_bytes())
                },
                | other => other,
            }
        });
        pairs
    }

    #[test]
    fn test_segment_builder_create_new() {
        let temp_dir = create_temp_dir();
        let builder = SegmentBuilder::new(temp_dir.path().to_path_buf()).unwrap();

        // Creating a new segment should succeed
        let segment = builder.new_segment(1, 42, (BLOCK_SIZE * 2) as u64).unwrap();
        assert!(
            Arc::strong_count(&segment) == 1,
            "Expected single reference to segment"
        );
    }

    #[test]
    fn test_segment_builder_write_and_reopen() {
        let temp_dir = create_temp_dir();
        let builder = SegmentBuilder::new(temp_dir.path().to_path_buf()).unwrap();
        let segment_id = 100;
        let seed = 42; // Fixed seed

        // Use fixed values instead of random ones
        let ns = 1; // Fixed namespace
        let fixed_ts = 12345678; // Fixed timestamp

        // Create controlled test data
        let mut test_data = Vec::new();
        for i in 0..5 {
            let key = KeyBytes::new(
                ns,
                Bytes::copy_from_slice(format!("key_{:05}", i).as_bytes()),
                fixed_ts,
            );

            let val = ValueBytes::new(
                ns,
                Bytes::copy_from_slice(format!("value_{:05}", i).as_bytes()),
            );

            test_data.push((key, val));
        }

        // Store serialized forms
        let serialized_pairs: Vec<(Bytes, Bytes)> = test_data
            .iter()
            .map(|(k, v)| (k.serialize(), v.serialize()))
            .collect();

        // Write data
        {
            let mut segment_arc = builder.new_segment(segment_id, seed, 4096 * 10).unwrap();
            let segment = Arc::get_mut(&mut segment_arc).unwrap();

            for (key_bytes, val_bytes) in &serialized_pairs {
                segment
                    .write(key_bytes.as_ref(), val_bytes.as_ref())
                    .expect("Failed to write to segment");
            }

            segment.flush().expect("Failed to flush segment");
            segment.close().expect("Failed to close segment");
        }

        // Reopen and verify
        {
            let reopened_segment = builder
                .open(segment_id)
                .expect("Failed to open existing segment");
            let reader = reopened_segment
                .new_reader()
                .expect("Failed to create reader");

            for (key_bytes, expected_val_bytes) in &serialized_pairs {
                let result = reader
                    .get(key_bytes.as_ref())
                    .expect("Error during get operation");

                assert!(
                    result.is_some(),
                    "Key not found: key_bytes={:?}",
                    &key_bytes[0..20]
                );
                assert_eq!(result.unwrap().as_ref(), expected_val_bytes.as_ref());
            }
        }
    }

    // #[test]
    // fn test_segment_builder_large_data() {
    //     let temp_dir = create_temp_dir();
    //     let builder =
    // SegmentBuilder::new(temp_dir.path().to_path_buf()).unwrap();
    //
    //     // Create a new segment
    //     let segment_id = 200;
    //     let seed = 100;
    //
    //     let vals = generate_test_data(1000);
    //
    //     // Step 1: Create segment and write large data
    //     {
    //         let mut segment_arc = builder.new_segment(segment_id, seed, 4096 *
    // 50).unwrap();         let segment =
    //             Arc::get_mut(&mut segment_arc).expect("Failed to get mutable
    // reference to segment");
    //
    //         for (key, value) in &vals {
    //             segment
    //                 .write(key.as_ref(), value.as_ref())
    //                 .expect("Failed to write to segment");
    //         }
    //
    //         segment.flush().expect("Failed to flush segment");
    //     }
    //
    //     // Step 2: Reopen segment and verify large data
    //     {
    //         let reopened_segment = builder.open(segment_id).expect("Failed to
    // reopen segment");         let reader = reopened_segment
    //             .new_reader()
    //             .expect("Failed to create reader");
    //
    //         for (key, _) in &vals {
    //             let result = reader
    //                 .get(key.as_ref())
    //                 .expect("Error during get operation");
    //             assert!(result.is_some(), "Key not found in reopened segment");
    //         }
    //     }
    // }

    #[test]
    fn test_segment_builder_multiple_instances() {
        let temp_dir = create_temp_dir();

        // Step 1: First builder creates and writes to segment
        let segment_id = 300;
        let seed = 200;
        let test_data = generate_test_data(20);

        {
            let builder1 = SegmentBuilder::new(temp_dir.path().to_path_buf()).unwrap();
            // Fix: use let binding to prevent temporary value from being dropped
            let mut segment_arc = builder1.new_segment(segment_id, seed, 4096 * 10).unwrap();
            let segment =
                Arc::get_mut(&mut segment_arc).expect("Failed to get mutable reference to segment");

            // Write data (keys must be serialized)
            for (key, value) in &test_data[0..10] {
                use crate::utils::Serializer;
                segment
                    .write(key.serialize().as_ref(), value.serialize().as_ref())
                    .expect("Failed to write to segment");
            }
            segment.flush().expect("Failed to flush segment");
            segment.close().expect("Failed to close segment");
        }

        // Step 2: Second builder opens segment and verifies data
        {
            let builder2 = SegmentBuilder::new(temp_dir.path().to_path_buf()).unwrap();
            let reopened_segment = builder2
                .open(segment_id)
                .expect("Failed to open existing segment");
            let reader = reopened_segment
                .new_reader()
                .expect("Failed to create reader");

            // Verify data (keys must be serialized for get)
            for (key, expected_value) in &test_data[0..10] {
                use crate::utils::Serializer;
                let result = reader
                    .get(key.serialize().as_ref())
                    .expect("Error during get operation");
                assert!(result.is_some(), "Key not found in reopened segment");
                assert_eq!(
                    result.unwrap().as_ref(),
                    expected_value.serialize().as_ref()
                );
            }
        }
    }

    #[test]
    fn test_segment_builder_sequential_operations() {
        let temp_dir = create_temp_dir();
        let builder = SegmentBuilder::new(temp_dir.path().to_path_buf()).unwrap();

        // Create multiple segments in sequence to ensure they don't interfere
        let total_segments = 3;
        let keys_per_segment = 5;

        // Step 1: Create and populate multiple segments
        let mut test_data_sets = Vec::with_capacity(total_segments);

        for i in 0..total_segments {
            let segment_id = 400 + (i as u64 * 2); // Key segments are even, value segments odd
            let seed = 300 + i as i64;

            let test_data = generate_test_data(keys_per_segment);
            test_data_sets.push((segment_id, test_data.clone()));

            // Fix: use let binding to prevent temporary value from being dropped
            let mut segment_arc = builder.new_segment(segment_id, seed, 4096 * 10).unwrap();
            let segment =
                Arc::get_mut(&mut segment_arc).expect("Failed to get mutable reference to segment");

            // Write data to this segment (keys must be serialized)
            for (key, value) in &test_data {
                use crate::utils::Serializer;
                segment
                    .write(key.serialize().as_ref(), value.serialize().as_ref())
                    .expect("Failed to write to segment");
            }
            segment.flush().expect("Failed to flush segment");
            segment.close().expect("Failed to close segment");
        }

        // Step 2: Reopen each segment and verify data
        for (segment_id, test_data) in test_data_sets {
            let reopened_segment = builder.open(segment_id).expect("Failed to reopen segment");
            let reader = reopened_segment
                .new_reader()
                .expect("Failed to create reader");

            // Verify data (keys must be serialized for get)
            for (key, expected_value) in &test_data {
                use crate::utils::Serializer;
                let result = reader
                    .get(key.serialize().as_ref())
                    .expect("Error during get operation");
                assert!(result.is_some(), "Key not found in segment");
                assert_eq!(
                    result.unwrap().as_ref(),
                    expected_value.serialize().as_ref()
                );
            }
        }
    }

    #[test]
    fn test_segment_builder_error_handling() {
        let temp_dir = create_temp_dir();
        let builder = SegmentBuilder::new(temp_dir.path().to_path_buf()).unwrap();

        // Try to open a non-existent segment
        let result = builder.open(9999);
        assert!(result.is_err(), "Opening non-existent segment should fail");

        // Try to create a segment with invalid size
        let result = builder.new_segment(1000, 42, 0); // Zero size
        assert!(
            result.is_err(),
            "Creating segment with zero size should fail"
        );
    }
}
