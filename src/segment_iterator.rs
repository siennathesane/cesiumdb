use std::collections::Bound;
use bytes::{Bytes, BytesMut};
use crate::block::{Block, EntryFlag};
use crate::errs::SegmentError;
use crate::errs::SegmentError::ReadOutOfBounds;
use crate::keypair::{KeyBytes, ValueBytes};
use crate::segment_reader::SegmentReader;
use crate::utils::Deserializer;

/// Helper function to convert a bound of &[u8] to a bound of Bytes
pub(crate)fn convert_bound_to_bytes(bound: Bound<&[u8]>) -> Bound<Bytes> {
    match bound {
        Bound::Included(data) => Bound::Included(Bytes::copy_from_slice(data)),
        Bound::Excluded(data) => Bound::Excluded(Bytes::copy_from_slice(data)),
        Bound::Unbounded => Bound::Unbounded,
    }
}

pub(crate) struct SegmentBlockIterator<'a> {
    reader: &'a mut SegmentReader<'a>,
    current_block: usize,
}

impl SegmentBlockIterator<'_> {
    pub(crate) fn new<'a>(reader: &'a mut SegmentReader<'a>) -> SegmentBlockIterator<'a> {
        SegmentBlockIterator {
            reader,
            current_block: 0,
        }
    }
}

impl<'a> Iterator for SegmentBlockIterator<'a> {
    type Item = Result<Block, SegmentError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_block >= self.reader.num_blocks {
            return None;
        }

        let result = self.reader.read_key_block(self.current_block);
        self.current_block += 1;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.reader.num_blocks - self.current_block;
        (remaining, Some(remaining))
    }
}

pub(crate) struct SeekingBlockIterator<'a> {
    start: usize,
    end: usize,
    current: usize,
    reader: &'a mut SegmentReader<'a>,
}

impl<'a> SeekingBlockIterator<'a> {
    pub(crate) fn new<'b>(reader: &'b mut SegmentReader<'b>, start: usize, end: usize) -> SeekingBlockIterator<'b> {
        SeekingBlockIterator {
            start,
            end,
            current: start,
            reader,
        }
    }
    pub(crate) fn seek(&mut self, block_index: usize) -> Result<(), SegmentError> {
        if block_index >= self.end {
            return Err(ReadOutOfBounds);
        }
        self.reader.clear_cache();
        self.current = block_index;
        Ok(())
    }

    pub(crate) fn current_position(&self) -> usize {
        self.current
    }

    pub(crate) fn blocks_remaining(&self) -> usize {
        self.end - self.current
    }
}

impl<'a> Iterator for SeekingBlockIterator<'a> {
    type Item = Result<Block, SegmentError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.end {
            return None;
        }
        let result = self.reader.read_key_block(self.current);
        self.current += 1;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.current;
        (remaining, Some(remaining))
    }
}

/// Iterator for scanning a range of keys in a segment.
pub(crate) struct SegmentScanIterator<'a> {
    reader: &'a SegmentReader<'a>,
    current_block_index: usize,
    current_key_block: Option<Block>,
    current_key_index: usize,
    lower_bound: Bound<Bytes>,
    upper_bound: Bound<Bytes>,
    is_upper_inclusive: bool,
    is_lower_inclusive: bool,
}

impl<'a> Iterator for SegmentScanIterator<'a> {
    type Item = Result<(KeyBytes, ValueBytes), SegmentError>;

    fn next(&mut self) -> Option<Self::Item> {
        // Keep trying until we find a valid entry or exhaust all blocks
        loop {
            // If we don't have a current block or have reached the end of the current block,
            // try to load the next block
            if self.current_key_block.is_none() ||
                self.current_key_index >= self.current_key_block.as_ref().unwrap().num_entries() as usize {
                match self.load_next_block() {
                    Ok(false) => return None, // No more blocks
                    Ok(true) => {}, // Successfully loaded next block
                    Err(e) => return Some(Err(e)), // Error loading block
                }
            }

            // Get the current entry
            let key_block = self.current_key_block.as_ref().unwrap();
            match key_block.get(self.current_key_index) {
                Some((flag, data)) => {
                    // Increment the index for the next iteration
                    self.current_key_index += 1;

                    // Process the entry based on the flag
                    let key_bytes = match flag {
                        EntryFlag::Complete => Bytes::copy_from_slice(data),
                        EntryFlag::Start => {
                            // For multi-block keys, we need to read the full key
                            match self.read_full_key(flag, data) {
                                Ok(bytes) => bytes,
                                Err(e) => return Some(Err(e)),
                            }
                        },
                        _ => continue, // Skip middle or end entries
                    };

                    // Check if the key is within our range
                    if !self.is_in_range(&key_bytes) {
                        // If we're past the upper bound, we can stop scanning
                        if self.is_past_upper_bound(&key_bytes) {
                            return None;
                        }
                        continue; // Skip this key
                    }

                    // Parse the key
                    let key = KeyBytes::deserialize_from_memory(key_bytes.clone());

                    // Use val_index to find the value block for this key
                    let val_bytes = match self.read_value_for_key(&key_bytes) {
                        Ok(Some(bytes)) => bytes,
                        Ok(None) => continue, // No value found, skip this key
                        Err(e) => return Some(Err(e)),
                    };

                    // Parse the value
                    let value = ValueBytes::deserialize_from_memory(val_bytes);

                    return Some(Ok((key, value)));
                },
                None => {
                    // No more entries in this block, try the next block
                    self.current_key_block = None;
                }
            }
        }
    }
}

impl<'a> SegmentScanIterator<'a> {
    /// Creates a new segment scan iterator for the given reader and key range.
    ///
    /// # Arguments
    /// * `reader` - The segment reader to scan
    /// * `range` - Range of keys to scan
    pub fn new(reader: &'a SegmentReader<'a>, range: (Bound<&[u8]>, Bound<&[u8]>)) -> Self {
        let lower_bound = convert_bound_to_bytes(range.0);
        let upper_bound = convert_bound_to_bytes(range.1);

        let is_lower_inclusive = matches!(lower_bound, Bound::Included(_));
        let is_upper_inclusive = matches!(upper_bound, Bound::Included(_));

        Self {
            reader,
            current_block_index: 0,
            current_key_block: None,
            current_key_index: 0,
            lower_bound,
            upper_bound,
            is_upper_inclusive,
            is_lower_inclusive,
        }
    }
    
    /// Loads the next block for scanning.
    ///
    /// Returns:
    /// - `Ok(true)` if a block was successfully loaded
    /// - `Ok(false)` if there are no more blocks
    /// - `Err(...)` if an error occurred
    fn load_next_block(&mut self) -> Result<bool, SegmentError> {
        // If we've reached the end of visible blocks, stop
        if self.current_block_index >= self.reader.visible_key_blocks {
            return Ok(false);
        }

        // Read the next block
        match self.reader.read_key_block(self.current_block_index) {
            Ok(block) => {
                self.current_key_block = Some(block);
                self.current_key_index = 0;
                self.current_block_index += 1;
                Ok(true)
            },
            Err(e) => {
                // In case of error, try to move to the next block
                self.current_block_index += 1;
                Err(e)
            }
        }
    }

    /// Reads a multi-block key.
    /// This is a simplified implementation and should be expanded for real use.
    fn read_full_key(&self, flag: EntryFlag, initial_data: &[u8]) -> Result<Bytes, SegmentError> {
        // For complete entries, just copy the data
        if flag == EntryFlag::Complete {
            return Ok(Bytes::copy_from_slice(initial_data));
        }

        // For multi-block keys (Start flag), we need to read subsequent blocks
        let mut buffer = BytesMut::with_capacity(initial_data.len() * 2);
        buffer.extend_from_slice(initial_data);

        let mut current_block_index = self.current_block_index - 1; // We're already past this block
        let mut found_end = false;

        while current_block_index < self.reader.visible_key_blocks && !found_end {
            let next_block = match self.reader.read_key_block(current_block_index) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };

            if next_block.num_entries() == 0 {
                current_block_index += 1;
                continue;
            }

            let (next_flag, next_data) = match next_block.get(0) {
                Some(v) => v,
                None => return Err(SegmentError::CorruptedBlock),
            };

            match next_flag {
                EntryFlag::Middle => {
                    buffer.extend_from_slice(next_data);
                    current_block_index += 1;
                },
                EntryFlag::End => {
                    buffer.extend_from_slice(next_data);
                    found_end = true;
                },
                _ => {
                    return Err(SegmentError::CorruptedBlock);
                },
            }
        }

        if !found_end {
            return Err(SegmentError::CorruptedBlock);
        }

        Ok(buffer.freeze())
    }

    /// Checks if a key is within the scan range.
    fn is_in_range(&self, key: &Bytes) -> bool {
        // Check lower bound
        let satisfies_lower = match &self.lower_bound {
            Bound::Included(lower) => key.as_ref() >= lower.as_ref(),
            Bound::Excluded(lower) => key.as_ref() > lower.as_ref(),
            Bound::Unbounded => true,
        };

        // Check upper bound
        let satisfies_upper = match &self.upper_bound {
            Bound::Included(upper) => key.as_ref() <= upper.as_ref(),
            Bound::Excluded(upper) => key.as_ref() < upper.as_ref(),
            Bound::Unbounded => true,
        };

        satisfies_lower && satisfies_upper
    }

    /// Checks if a key is past the upper bound of the scan range.
    fn is_past_upper_bound(&self, key: &Bytes) -> bool {
        match &self.upper_bound {
            Bound::Included(upper) => key.as_ref() > upper.as_ref(),
            Bound::Excluded(upper) => key.as_ref() >= upper.as_ref(),
            Bound::Unbounded => false,
        }
    }

    /// Reads the value for a key.
    fn read_value_for_key(&self, key: &Bytes) -> Result<Option<Bytes>, SegmentError> {
        // Use val_index to find the value block for this key
        let val_block_offset = match self.reader.val_index.find_block(key) {
            Some(offset) => offset,
            None => return Ok(None), // No value block found for this key
        };

        // Read the value from the found block
        match self.reader.read_value(val_block_offset as usize, 0) {
            Ok(value) => Ok(Some(value)),
            Err(e) => Err(e),
        }
    }
}