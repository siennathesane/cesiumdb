use std::{
    fmt::Display,
    mem,
    sync::{
        Arc,
        atomic::{
            AtomicU64,
            Ordering::Relaxed,
        },
    },
};

use crate::{
    block::{
        Block,
        EntryFlag,
        EntryFlag::{
            Complete,
            End,
            Middle,
            Start,
        },
        MAX_ENTRY_SIZE,
    },
    errs::{
        SegmentError,
        SegmentError::CantCreateReader,
    },
    index::Index,
    keypair::DEFAULT_NS,
    segment::BlockType::{
        Key,
        Value,
    },
    segment_reader::{
        ReadConfig,
        SegmentReader,
    },
    segment_writer::SegmentWriter,
};

#[derive(Debug)]
pub enum BlockType {
    Key,
    Value,
}

impl Display for BlockType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            | Key => write!(f, "key"),
            | Value => write!(f, "value"),
        }
    }
}

pub struct Segment {
    // keys
    key_writer: SegmentWriter,
    key_block_count: AtomicU64,
    key_index: Index,
    current_key_block: Block,

    // values
    val_writer: SegmentWriter,
    val_block_count: AtomicU64,
    current_val_block: Block,
    val_index: Index,

    // shared
    current_ns: AtomicU64,
}

impl Segment {
    pub fn new(
        key_id: u64,
        val_id: u64,
        seed: i64,
        key_writer: SegmentWriter,
        val_writer: SegmentWriter,
    ) -> Self {
        let mut key_index = Index::new(key_id, seed);
        let mut val_index = Index::new(val_id, seed);

        // explicitly record the default namespace in both indexes
        // during init to ensure the namespace is always recorded.
        // this is because the namespace is only recorded when it
        // changes, but if we start with the default namespace, it
        // will never be seen to change. this creates an extra byte
        // of space in the index, but there isn't a way to say "this
        // is the first namespace" without doing this. so even if the
        // first seen namespace isn't the default one, it will still
        // be a pointer to the same record.
        key_index.add_ns_offset(DEFAULT_NS);
        val_index.add_ns_offset(DEFAULT_NS);

        Self {
            key_writer,
            key_block_count: AtomicU64::new(0),
            val_writer,
            val_block_count: AtomicU64::new(0),
            key_index,
            current_key_block: Block::new(),
            current_val_block: Block::new(),
            val_index,
            current_ns: AtomicU64::new(DEFAULT_NS),
        }
    }

    pub fn write(&mut self, key: &[u8], val: &[u8]) -> Result<(), SegmentError> {
        // set the namespace
        let ns = u64::from_le_bytes(key[0..8].as_ref().try_into().unwrap());
        if ns != self.current_ns.load(Relaxed) {
            self.current_ns.store(ns, Relaxed);
            self.key_index.add_ns_offset(ns);
            self.val_index.add_ns_offset(ns);
        }

        // Write key to key block
        match self.current_key_block.add_entry(key, Complete) {
            | Ok(()) => {},
            | Err(be) => match be {
                | TooLargeForBlock => {
                    match self.split_across_blocks(key, &Key) {
                        | Ok(_) => {},
                        | Err(e) => return Err(e),
                    };
                },
                | BlockFull => {
                    match self.write_block(&Key) {
                        | Ok(_) => {},
                        | Err(e) => return Err(e),
                    };
                    match self.current_key_block.add_entry(key, Complete) {
                        | Ok(_) => {},
                        | Err(rbe) => match rbe {
                            | TooLargeForBlock => {
                                match self.split_across_blocks(key, &Key) {
                                    | Ok(_) => {},
                                    | Err(e) => return Err(e),
                                };
                            },
                            | _ => {
                                unreachable!("unexpected key block error, no idea how we got here")
                            },
                        },
                    };
                },
            },
        };
        self.key_index.add_item(key);

        // Write value to value block - FIXED
        match self.current_val_block.add_entry(val, Complete) {
            | Ok(()) => {},
            | Err(be) => match be {
                | TooLargeForBlock => {
                    match self.split_across_blocks(val, &Value) {
                        | Ok(_) => {},
                        | Err(e) => return Err(e),
                    };
                },
                | BlockFull => {
                    match self.write_block(&Value) {
                        | Ok(_) => {},
                        | Err(e) => return Err(e),
                    };
                    match self.current_val_block.add_entry(val, Complete) {
                        | Ok(_) => {},
                        | Err(rbe) => match rbe {
                            | TooLargeForBlock => {
                                match self.split_across_blocks(val, &Value) {
                                    | Ok(_) => {},
                                    | Err(e) => return Err(e),
                                };
                            },
                            | _ => {
                                unreachable!("unexpected val block error, no idea how we got here")
                            },
                        },
                    };
                },
            },
        };
        self.val_index.add_item(key);

        Ok(())
    }

    pub fn new_reader(&self) -> SegmentReader {
        let key_blocks = self.key_block_count.load(Relaxed) as usize;
        let val_blocks = self.val_block_count.load(Relaxed) as usize;

        match SegmentReader::with_visibility(
            self.key_writer.map.clone(),
            self.val_writer.map.clone(),
            &self.key_index,
            &self.val_index,
            key_blocks,
            val_blocks,
            ReadConfig::default(),
        ) {
            | Ok(v) => v,
            // fallback to default if there's an error
            | Err(_) => SegmentReader::new(
                self.key_writer.map.clone(),
                self.val_writer.map.clone(),
                &self.key_index,
                &self.val_index,
            )
            .unwrap(),
        }
    }

    /// Split a payload across multiple blocks.
    fn split_across_blocks(&mut self, data: &[u8], r#type: &BlockType) -> Result<(), SegmentError> {
        let mut remaining = data;

        // First, determine the maximum chunk size that can fit in a block
        // Account for the entry flag byte and other overhead
        let max_chunk_size = MAX_ENTRY_SIZE - 1;

        // Process the first chunk (START flag)
        if !remaining.is_empty() {
            let chunk_size = std::cmp::min(max_chunk_size, remaining.len());
            let chunk = &remaining[..chunk_size];

            match r#type {
                | Key => {
                    // Always start with a new block
                    if !self.current_key_block.is_empty() {
                        match self.write_block(&Key) {
                            | Ok(_) => {},
                            | Err(e) => return Err(e),
                        };
                    }

                    match self.current_key_block.add_entry(chunk, Start) {
                        | Ok(_) => {
                            self.key_index.add_block(data);
                            match self.write_block(&Key) {
                                | Ok(_) => {},
                                | Err(e) => return Err(e),
                            };
                        },
                        | Err(e) => {
                            eprintln!(
                                "Error adding START chunk to key block: {:?}, chunk size: {}",
                                e,
                                chunk.len()
                            );
                            return Err(SegmentError::InsufficientSpace);
                        },
                    }
                },
                | Value => {
                    // Always start with a new block
                    if !self.current_val_block.is_empty() {
                        match self.write_block(&Value) {
                            | Ok(_) => {},
                            | Err(e) => return Err(e),
                        };
                    }

                    match self.current_val_block.add_entry(chunk, Start) {
                        | Ok(_) => {
                            self.val_index.add_block(data);
                            match self.write_block(&Value) {
                                | Ok(_) => {},
                                | Err(e) => return Err(e),
                            };
                        },
                        | Err(e) => {
                            eprintln!(
                                "Error adding START chunk to value block: {:?}, chunk size: {}",
                                e,
                                chunk.len()
                            );
                            return Err(SegmentError::InsufficientSpace);
                        },
                    }
                },
            }

            remaining = &remaining[chunk_size..];
        }

        // Process middle chunks (MIDDLE flag)
        while remaining.len() > max_chunk_size {
            let chunk = &remaining[..max_chunk_size];

            match r#type {
                | Key => {
                    let mut block = Block::new();
                    match block.add_entry(chunk, Middle) {
                        | Ok(_) => {
                            match self
                                .key_writer
                                .write_block(block)
                                .map(|_| self.key_block_count.fetch_add(1, Relaxed))
                            {
                                | Ok(_) => {},
                                | Err(e) => return Err(e),
                            };
                        },
                        | Err(e) => {
                            eprintln!(
                                "Error adding MIDDLE chunk to key block: {:?}, chunk size: {}",
                                e,
                                chunk.len()
                            );
                            return Err(SegmentError::InsufficientSpace);
                        },
                    }
                },
                | Value => {
                    let mut block = Block::new();
                    match block.add_entry(chunk, Middle) {
                        | Ok(_) => {
                            match self
                                .val_writer
                                .write_block(block)
                                .map(|_| self.val_block_count.fetch_add(1, Relaxed))
                            {
                                | Ok(_) => {},
                                | Err(e) => return Err(e),
                            };
                        },
                        | Err(e) => {
                            eprintln!(
                                "Error adding MIDDLE chunk to value block: {:?}, chunk size: {}",
                                e,
                                chunk.len()
                            );
                            return Err(SegmentError::InsufficientSpace);
                        },
                    }
                },
            }

            remaining = &remaining[max_chunk_size..];
        }

        // Process the last chunk (END flag) if there's anything left
        if !remaining.is_empty() {
            match r#type {
                | Key => {
                    let mut block = Block::new();
                    match block.add_entry(remaining, End) {
                        | Ok(_) => {
                            match self
                                .key_writer
                                .write_block(block)
                                .map(|_| self.key_block_count.fetch_add(1, Relaxed))
                            {
                                | Ok(_) => {},
                                | Err(e) => return Err(e),
                            };
                        },
                        | Err(e) => {
                            eprintln!(
                                "Error adding END chunk to key block: {:?}, chunk size: {}",
                                e,
                                remaining.len()
                            );
                            return Err(SegmentError::InsufficientSpace);
                        },
                    }
                },
                | Value => {
                    let mut block = Block::new();
                    match block.add_entry(remaining, End) {
                        | Ok(_) => {
                            match self
                                .val_writer
                                .write_block(block)
                                .map(|_| self.val_block_count.fetch_add(1, Relaxed))
                            {
                                | Ok(_) => {},
                                | Err(e) => return Err(e),
                            };
                        },
                        | Err(e) => {
                            eprintln!(
                                "Error adding END chunk to value block: {:?}, chunk size: {}",
                                e,
                                remaining.len()
                            );
                            return Err(SegmentError::InsufficientSpace);
                        },
                    }
                },
            }
        }

        Ok(())
    }

    fn write_block(&mut self, r#type: &BlockType) -> Result<(), SegmentError> {
        match r#type {
            | Key => {
                if self.current_key_block.is_empty() {
                    return Ok(()); // Nothing to write
                }

                // Extract the first key if it exists
                let starting_key_data = match self.current_key_block.get(0) {
                    | Some((_flag, key_data)) => Some(key_data.to_vec()),
                    | None => None,
                };

                // Now swap the block
                let block = mem::replace(&mut self.current_key_block, Block::new());

                // Write block and increment counter
                let result = self.key_writer.write_block(block).map(|_| {
                    self.key_block_count.fetch_add(1, Relaxed);
                });

                // Add to index if we have a starting key
                if result.is_ok() {
                    if let Some(key_data) = starting_key_data {
                        self.key_index.add_block(&key_data);
                    }
                }

                result
            },
            | Value => {
                if self.current_val_block.is_empty() {
                    return Ok(()); // Nothing to write
                }

                // Extract the first key if it exists
                let starting_key_data = match self.current_val_block.get(0) {
                    | Some((_flag, key_data)) => Some(key_data.to_vec()),
                    | None => None,
                };

                // Now swap the block
                let block = mem::replace(&mut self.current_val_block, Block::new());

                // Write block and increment counter
                let result = self.val_writer.write_block(block).map(|_| {
                    self.val_block_count.fetch_add(1, Relaxed);
                });

                // Add to index if we have a starting key
                if result.is_ok() {
                    if let Some(key_data) = starting_key_data {
                        self.val_index.add_block(&key_data);
                    }
                }

                result
            },
        }
    }

    /// Flush any pending blocks
    pub fn flush(&mut self) -> Result<(), SegmentError> {
        // Flush key block if it has entries
        if !self.current_key_block.is_empty() {
            match self.write_block(&Key) {
                | Ok(_) => {},
                | Err(e) => return Err(e),
            };
        }

        // Flush value block if it has entries
        if !self.current_val_block.is_empty() {
            match self.write_block(&Value) {
                | Ok(_) => {},
                | Err(e) => return Err(e),
            };
        }

        Ok(())
    }

    pub fn sync(&mut self) -> Result<(), SegmentError> {
        match self.flush() {
            | Ok(_) => Ok(()),
            | Err(e) => Err(e),
        }
    }
}

impl Drop for Segment {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

#[cfg(test)]
#[allow(clippy::question_mark_used)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::undocumented_unsafe_blocks)]
mod tests {
    use std::{
        collections::HashMap,
        sync::Arc,
    };

    use bytes::Bytes;
    use rand::{
        Rng,
        RngCore,
        prelude::SliceRandom,
        rng,
        thread_rng,
    };
    use tempfile::tempdir;

    use super::*;
    use crate::{
        block::Block,
        hlc::{
            HLC,
            HybridLogicalClock,
        },
        keypair::{
            DEFAULT_NS,
            KeyBytes,
            ValueBytes,
        },
        map::Map,
        memtable::Memtable,
        segment_reader::SegmentReader,
        segment_writer::SegmentWriter,
    };

    // helper function to create temporary segment components
    fn create_test_segment() -> (Arc<Segment>, tempfile::TempDir) {
        let dir = tempdir().expect("failed to create temp dir");

        // Add a random component to filenames to ensure uniqueness even if temp dir is
        // reused
        let random_id: u64 = rand::random();

        // Create key map and writer
        let key_path = dir.path().join(format!("test-key-segment-{}", random_id));
        let key_map = Arc::new(Map::new(key_path, 4096 * 10).expect("failed to create key map"));
        let key_writer = SegmentWriter::new(key_map.clone()).expect("failed to create key writer");

        // Create value map and writer
        let val_path = dir.path().join(format!("test-val-segment-{}", random_id));
        let val_map = Arc::new(Map::new(val_path, 4096 * 10).expect("failed to create val map"));
        let val_writer = SegmentWriter::new(val_map.clone()).expect("failed to create val writer");

        // Create segment
        let seed = 42i64; // Fixed seed for reproducibility
        let segment = Arc::new(Segment::new(1, 2, seed, key_writer, val_writer));

        (segment, dir)
    }

    // helper function to create test key-value pair
    fn create_kv(key: &str, value: &str, clock: &HybridLogicalClock) -> (KeyBytes, ValueBytes) {
        (
            KeyBytes::new(DEFAULT_NS, Bytes::from(key.to_string()), clock.time()),
            ValueBytes::new(DEFAULT_NS, Bytes::from(value.to_string())),
        )
    }

    #[test]
    fn test_segment_basic_write() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Simple key-value pair
        let key = [0u8, 0, 0, 0, 0, 0, 0, 0, b'a', b'b', b'c'];
        let val = [0u8, 0, 0, 0, 0, 0, 0, 0, b'1', b'2', b'3'];

        // Write to segment
        let result = segment.write(&key, &val);
        assert!(result.is_ok(), "Failed to write to segment: {:?}", result);
    }

    #[test]
    fn test_segment_multiple_writes() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Write multiple key-value pairs
        for i in 0u32..10 {
            let mut key = vec![0u8; 8]; // namespace
            key.extend_from_slice(&i.to_le_bytes());

            let mut val = vec![0u8; 8]; // namespace
            val.extend_from_slice(&(i * 10).to_le_bytes());

            let result = segment.write(&key, &val);
            assert!(result.is_ok(), "Failed to write entry {}: {:?}", i, result);
        }
    }

    #[test]
    fn test_segment_namespace_handling() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Write entries with different namespaces
        for ns in &[1u64, 2u64, 1u64, 3u64, 2u64] {
            let mut key = ns.to_le_bytes().to_vec();
            key.extend_from_slice(b"testkey");

            let mut val = ns.to_le_bytes().to_vec();
            val.extend_from_slice(b"testvalue");

            let result = segment.write(&key, &val);
            assert!(
                result.is_ok(),
                "Failed to write entry for ns {}: {:?}",
                ns,
                result
            );
        }

        // Check that the namespaces were recorded in the indexes
        // Since we wrote 3 different namespaces with some repetition
        // We need to make sure the ns tracking works correctly
        assert!(
            segment.key_index.ns_offset_count() >= 3,
            "Key index should track at least 3 namespace changes"
        );
        assert!(
            segment.val_index.ns_offset_count() >= 3,
            "Value index should track at least 3 namespace changes"
        );
    }

    #[test]
    fn test_segment_large_entry() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Create a key that fits in one block
        let mut key = vec![0u8; 8]; // namespace
        key.extend_from_slice(b"large_entry_key");

        // Create a large value that spans multiple blocks
        let mut val = vec![0u8; 8]; // namespace
        let large_data_size = 8192; // 2 blocks worth of data
        let mut large_data = vec![0u8; large_data_size];
        thread_rng().fill_bytes(&mut large_data);
        val.extend_from_slice(&large_data);

        let result = segment.write(&key, &val);
        assert!(result.is_ok(), "Failed to write large entry: {:?}", result);
    }

    #[test]
    fn test_segment_mixed_entry_sizes() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        let mut rng = thread_rng();

        // Write entries with varying sizes
        for i in 0u32..20 {
            let mut key = vec![0u8; 8]; // namespace
            key.extend_from_slice(&i.to_le_bytes());

            // Generate random-sized values
            let size = match i % 4 {
                | 0 => 10,   // Small
                | 1 => 1000, // Medium
                | 2 => 4000, // Almost a block
                | 3 => 6000, // Multi-block
                | _ => unreachable!(),
            };

            let mut val = vec![0u8; 8]; // namespace
            let mut data = vec![0u8; size];
            rng.fill_bytes(&mut data);
            val.extend_from_slice(&data);

            let result = segment.write(&key, &val);
            assert!(
                result.is_ok(),
                "Failed to write entry with size {}: {:?}",
                size,
                result
            );
        }
    }

    #[test]
    fn test_segment_reader_creation() {
        let (segment, _dir) = create_test_segment();

        // Get a new reader
        let reader = segment.new_reader();
        assert!(
            reader.num_blocks() > 0,
            "Should be able to get block count from reader"
        );
    }

    #[test]
    fn test_segment_with_many_small_entries() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Write many small entries to test block packing
        for i in 0u32..1000 {
            let mut key = vec![0u8; 8]; // namespace
            key.extend_from_slice(&i.to_le_bytes());

            let mut val = vec![0u8; 8]; // namespace
            val.extend_from_slice(format!("Value{}", i).as_bytes());

            let result = segment.write(&key, &val);
            assert!(
                result.is_ok(),
                "Failed to write small entry {}: {:?}",
                i,
                result
            );
        }

        // Check blocks were created
        assert!(
            segment.key_block_count.load(Relaxed) > 0,
            "Should have created some key blocks"
        );
        assert!(
            segment.val_block_count.load(Relaxed) > 0,
            "Should have created some value blocks"
        );
    }

    #[test]
    fn test_segment_sequential_keys() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Test with lexicographically ordered keys
        let mut data = vec![];
        for c in 'a'..='z' {
            let key = format!("key_{}", c);
            let value = format!("value_{}", c);

            let mut key_data = DEFAULT_NS.to_le_bytes().to_vec();
            key_data.extend_from_slice(key.as_bytes());

            let mut val_data = DEFAULT_NS.to_le_bytes().to_vec();
            val_data.extend_from_slice(value.as_bytes());

            data.push((key_data, val_data));
        }

        // Write in order
        for (key, val) in data {
            let result = segment.write(&key, &val);
            assert!(
                result.is_ok(),
                "Failed to write sequential key: {:?}",
                result
            );
        }
    }

    #[test]
    fn test_segment_random_order_keys() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Create entries with random ordering
        let mut data = vec![];
        for i in 0..100 {
            let key = format!("random_key_{:03}", i);
            let value = format!("random_value_{:03}", i);

            let mut key_data = DEFAULT_NS.to_le_bytes().to_vec();
            key_data.extend_from_slice(key.as_bytes());

            let mut val_data = DEFAULT_NS.to_le_bytes().to_vec();
            val_data.extend_from_slice(value.as_bytes());

            data.push((key_data, val_data));
        }

        // Shuffle the data
        let mut rng = rand::thread_rng();
        data.shuffle(&mut rng);

        // Write in random order
        for (key, val) in data {
            let result = segment.write(&key, &val);
            assert!(
                result.is_ok(),
                "Failed to write random-order key: {:?}",
                result
            );
        }
    }

    #[test]
    fn test_segment_index_building() {
        let (mut segment, _dir) = create_test_segment();
        let segment = Arc::get_mut(&mut segment).unwrap();

        // Write entries and check index growth
        let initial_key_blocks = segment.key_index.block_count();

        for i in 0u32..512 {
            let mut key = vec![0u8; 8]; // namespace
            key.extend_from_slice(&i.to_le_bytes());

            let mut val = vec![0u8; 8]; // namespace
            val.extend_from_slice(&(i * 10).to_le_bytes());

            let result = segment.write(&key, &val);
            assert!(result.is_ok(), "Failed to write entry {}: {:?}", i, result);
        }

        segment.flush().expect("failed to flush segment");

        let final_key_blocks = segment.key_index.block_count();

        assert_eq!(
            final_key_blocks, 4,
            "there should be 4 blocks in the key index, found: {}",
            final_key_blocks
        );
    }
}
