use std::{
    path::PathBuf,
    sync::Arc,
};

use bytes::Bytes;

use crate::{
    errs::{
        SegmentError,
        SegmentError::CantCreateWriter,
    },
    index::Index,
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
        let key_mmap = match Map::open(key_path) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };

        // load the metadata from the end of the key mmap
        let mdata_size = size_of::<Metadata>();
        let key_mdata_payload = key_mmap[key_mmap.len() - mdata_size..key_mmap.len()].as_ref();
        let key_metadata = Metadata::from(Bytes::copy_from_slice(key_mdata_payload));

        let key_index_payload = key_mmap
            [key_metadata.index_start()..key_metadata.index_start() + key_metadata.index_size()]
            .as_ref();
        let key_index = Index::from(Bytes::copy_from_slice(key_index_payload));

        let val_path = self.root.join(val_segment_id.to_string());
        let val_mmap = match Map::open(val_path) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };

        let val_mdata_payload = val_mmap[val_mmap.len() - mdata_size..val_mmap.len()].as_ref();
        let val_metadata = Metadata::from(Bytes::copy_from_slice(val_mdata_payload));

        let val_index_payload = val_mmap
            [val_metadata.index_start()..val_metadata.index_start() + val_metadata.index_size()]
            .as_ref();
        let val_index = Index::from(Bytes::copy_from_slice(val_index_payload));

        let key_handle = Arc::new(key_mmap);
        let val_handle = Arc::new(val_mmap);

        Segment::open(
            key_handle,
            key_index,
            key_metadata.id(),
            val_handle,
            val_index,
            val_metadata.id(),
        )
    }
}

#[cfg(test)]
mod tests {
    use rand::{
        Rng,
        random_range,
        rng,
    };
    use tempfile::tempdir;

    use super::*;
    use crate::{
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

            let val = ValueBytes::new(DEFAULT_NS, Bytes::copy_from_slice(random_data.as_slice()));

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
        let segment = builder.new_segment(1, 42, 4096 * 10).unwrap();
        assert!(
            Arc::strong_count(&segment) == 1,
            "Expected single reference to segment"
        );
    }

    #[test]
    fn test_segment_builder_write_and_reopen() {
        let temp_dir = create_temp_dir();
        let builder = SegmentBuilder::new(temp_dir.path().to_path_buf()).unwrap();

        // Create a new segment
        let segment_id = 100;
        let seed = 42;

        // Step 1: Create segment and write data
        {
            // Fix: use let binding to prevent temporary value from being dropped
            let mut segment_arc = builder.new_segment(segment_id, seed, 4096 * 10).unwrap();
            let segment =
                Arc::get_mut(&mut segment_arc).expect("Failed to get mutable reference to segment");

            // Write some data
            let test_data = generate_test_data(10);
            for (key, value) in &test_data {
                segment
                    .write(key.as_ref(), value.as_ref())
                    .expect("Failed to write to segment");
            }

            // Ensure data is flushed to disk
            segment.flush().expect("Failed to flush segment");

            segment.close().expect("Failed to close segment");
        }

        // Step 2: Reopen segment and verify data
        {
            let reopened_segment = builder
                .open(segment_id)
                .expect("Failed to open existing segment");

            // Create a reader to read data back
            let reader = reopened_segment
                .new_reader()
                .expect("Failed to create reader");

            // Verify some data
            let test_data = generate_test_data(10);
            for (key, expected_value) in &test_data {
                let result = reader
                    .get(key.as_ref())
                    .expect("Error during get operation");
                assert!(result.is_some(), "Key not found: {:?}", key);
                assert_eq!(result.unwrap().as_ref(), expected_value.as_bytes());
            }
        }
    }

    #[test]
    fn test_segment_builder_large_data() {
        let temp_dir = create_temp_dir();
        let builder = SegmentBuilder::new(temp_dir.path().to_path_buf()).unwrap();

        // Create a new segment
        let segment_id = 200;
        let seed = 100;

        let vals = generate_test_data(1000);

        // Step 1: Create segment and write large data
        {
            // Fix: use let binding to prevent temporary value from being dropped
            let mut segment_arc = builder.new_segment(segment_id, seed, 4096 * 50).unwrap();
            let segment =
                Arc::get_mut(&mut segment_arc).expect("Failed to get mutable reference to segment");

            for (key, value) in &vals {
                segment
                    .write(key.as_ref(), value.as_ref())
                    .expect("Failed to write to segment");
            }

            segment.flush().expect("Failed to flush segment");
        }

        // Step 2: Reopen segment and verify large data
        {
            let reopened_segment = builder.open(segment_id).expect("Failed to reopen segment");
            let reader = reopened_segment
                .new_reader()
                .expect("Failed to create reader");

            for (key, _) in &vals {
                let result = reader
                    .get(key.as_ref())
                    .expect("Error during get operation");
                assert!(result.is_some(), "Key not found in reopened segment");
            }
        }
    }

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

            // Write data
            for (key, value) in &test_data[0..10] {
                segment
                    .write(key.as_ref(), value.as_ref())
                    .expect("Failed to write to segment");
            }
            segment.flush().expect("Failed to flush segment");
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

            // Verify data
            for (key, expected_value) in &test_data[0..10] {
                let result = reader
                    .get(key.as_ref())
                    .expect("Error during get operation");
                assert!(result.is_some(), "Key not found in reopened segment");
                assert_eq!(result.unwrap().as_ref(), expected_value.as_bytes());
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

            // Write data to this segment
            for (key, value) in &test_data {
                segment
                    .write(key.as_ref(), value.as_ref())
                    .expect("Failed to write to segment");
            }
            segment.flush().expect("Failed to flush segment");
        }

        // Step 2: Reopen each segment and verify data
        for (segment_id, test_data) in test_data_sets {
            let reopened_segment = builder.open(segment_id).expect("Failed to reopen segment");
            let reader = reopened_segment
                .new_reader()
                .expect("Failed to create reader");

            // Verify data
            for (key, expected_value) in &test_data {
                let result = reader
                    .get(key.as_ref())
                    .expect("Error during get operation");
                assert!(result.is_some(), "Key not found in segment");
                assert_eq!(result.unwrap().as_ref(), expected_value.as_bytes());
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
