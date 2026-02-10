use std::ptr;

use bytes::{
    BufMut,
    Bytes,
    BytesMut,
};

use crate::{
    errs::{
        BlockError,
        BlockError::{
            BlockFull,
            TooLargeForBlock,
        },
    },
    utils::Deserializer,
};

const OFFSET_SIZE: usize = size_of::<u16>();
const MAX_ENTRIES: usize = BLOCK_SIZE / ENTRY_SIZE;
/// The size of a block in bytes. This is the most common page size for memory
/// and NVMe devices.
pub const BLOCK_SIZE: usize = 4096;
/// The size of an entry in a block. An entry consists of a 2-byte offset and a
/// byte flag for the entry type.
pub(crate) const ENTRY_SIZE: usize = size_of::<u16>() + size_of::<u8>();
/// The overhead of a block, which is the space taken up by the offsets and
/// flags.
pub(crate) const BLOCK_OVERHEAD: usize = BLOCK_SIZE - MAX_ENTRIES;
/// The maximum entry size that can fit into an empty block.
pub(crate) const MAX_ENTRY_SIZE: usize = BLOCK_SIZE - OFFSET_SIZE - ENTRY_SIZE;

/// Flags to mark entry types in a block
#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum EntryFlag {
    Complete = 0, // Regular complete entry
    Start    = 1, // Start of a multi-block entry
    Middle   = 2, // Middle of a multi-block entry
    End      = 3, // End of a multi-block entry
}

/// A mutable block used during segment writing. Uses BytesMut staging buffers.
#[derive(Debug)]
pub(crate) struct Block {
    /// The number of entries in the block.
    num_entries: u16,
    /// The entry offsets (BytesMut for now, will be replaced with direct mmap
    /// writes)
    offsets: BytesMut,
    /// The actual entries (BytesMut for now, will be replaced with direct mmap
    /// writes)
    entries: BytesMut,
}

/// A block builder that writes directly to a mmap buffer (zero-copy writes)
pub(crate) struct BlockBuilder<'a> {
    /// Direct pointer to mmap buffer (BLOCK_SIZE bytes)
    buffer: &'a mut [u8],
    /// Number of entries written
    num_entries: u16,
    /// Temporary storage for offsets (will be written to buffer during
    /// finalize)
    offsets: Vec<u16>,
    /// Temporary storage for entries (will be written to buffer during
    /// finalize)
    entries: Vec<u8>,
}

/// A read-only block used after deserialization. Uses Bytes for cheap cloning
/// (just Arc bumps).
#[derive(Debug, Clone)]
pub(crate) struct ReadOnlyBlock {
    /// The number of entries in the block.
    num_entries: u16,
    /// The entry offsets, it's just a [u16]. Uses Bytes for zero-copy cloning.
    offsets: Bytes,
    /// The actual entries, it's just a single flag byte followed by the data.
    /// Uses Bytes for zero-copy cloning.
    entries: Bytes,
}

impl Block {
    /// Create a new block with BytesMut staging buffers
    pub(crate) fn new() -> Self {
        Block {
            num_entries: 0,
            offsets: BytesMut::new(),
            entries: BytesMut::new(),
        }
    }

    /// Add an entry to the block. If the block is full, an error will be
    /// returned.
    pub(crate) fn add_entry(&mut self, entry: &[u8], flag: EntryFlag) -> Result<(), BlockError> {
        // entry + offset size + byte flag
        let entry_size = entry.len() + size_of::<u8>();
        if !self.will_fit(entry_size) {
            return if self.is_empty() {
                // notify the caller to try again with a smaller entry
                Err(TooLargeForBlock)
            } else {
                // notify the caller that the block is full
                Err(BlockFull)
            };
        }

        // calculate the next offset
        let mut current_offset = 0;
        if self.num_entries > 0 {
            let offset = &self.offsets[self.offsets.len() - 2..];
            current_offset = u16::from_le_bytes([offset[0], offset[1]]);
        }
        let next_offset = current_offset + (entry_size as u16);

        // add the entry and update the offsets
        self.offsets.put_u16_le(next_offset);
        self.entries.put_u8(flag as u8); // flag for complete entry
        self.entries.put_slice(entry);
        self.num_entries += 1;

        Ok(())
    }

    /// Finalize the block by writing directly to the provided memory location.
    ///
    /// # Safety
    /// - dst must be valid for BLOCK_SIZE bytes
    /// - dst must be properly aligned for u16 writes (2-byte alignment)
    /// - dst must not overlap with any source data
    /// - Caller must ensure exclusive access to the dst memory region
    pub(crate) unsafe fn finalize(&self, dst: *mut u8) {
        // SAFETY: Verify alignment invariants in debug builds
        debug_assert!(!dst.is_null(), "Destination pointer must not be null");
        debug_assert!(
            dst as usize % std::mem::align_of::<u16>() == 0,
            "Destination pointer must be 2-byte aligned for u16 writes"
        );

        // SAFETY: All writes stay within BLOCK_SIZE bytes.
        // Each write uses non-overlapping offsets.
        unsafe {
            // write num_entries
            ptr::copy_nonoverlapping(
                self.num_entries.to_le_bytes().as_ptr(),
                dst,
                size_of::<u16>(),
            );

            // write offsets
            ptr::copy_nonoverlapping(
                self.offsets.as_ptr(),
                dst.add(size_of::<u16>()),
                self.offsets.len(),
            );

            // write entries
            ptr::copy_nonoverlapping(
                self.entries.as_ptr(),
                dst.add(size_of::<u16>() + self.offsets.len()),
                self.entries.len(),
            );

            // zero remaining space
            let written = size_of::<u16>() + self.offsets.len() + self.entries.len();
            if written < BLOCK_SIZE {
                ptr::write_bytes(dst.add(written), 0, BLOCK_SIZE - written);
            }
        }
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<(EntryFlag, &[u8])> {
        if index >= self.num_entries as usize {
            return None;
        }

        // Get the offsets directly without going through iterator state
        let start_offset = if index == 0 {
            0
        } else {
            let offset_idx = (index - 1) * 2;
            u16::from_le_bytes([self.offsets[offset_idx], self.offsets[offset_idx + 1]]) as usize
        };

        let end_offset = if index < self.num_entries as usize - 1 {
            let offset_idx = index * 2;
            u16::from_le_bytes([self.offsets[offset_idx], self.offsets[offset_idx + 1]]) as usize
        } else {
            self.entries.len()
        };

        let entry_data = &self.entries[start_offset..end_offset];
        let flag = match entry_data[0] {
            | 0 => EntryFlag::Complete,
            | 1 => EntryFlag::Start,
            | 2 => EntryFlag::Middle,
            | 3 => EntryFlag::End,
            | _ => unreachable!("invalid entry flag"),
        };

        Some((flag, &entry_data[1..]))
    }

    /// Add an entry that is part of a single block.
    pub(crate) fn add_complete_entry(&mut self, entry: &[u8]) -> Result<(), BlockError> {
        self.add_entry(entry, EntryFlag::Complete)
    }

    /// Returns an iterator over the entries in the block.
    #[inline]
    pub fn iter(&self) -> BlockIterator<'_> {
        BlockIterator {
            entries: self.entries.as_ref(),
            offsets: self.offsets.as_ref(),
            current: 0,
            num_entries: self.num_entries,
        }
    }
}

/// Helper methods.
impl Block {
    #[inline]
    pub(crate) fn offsets(&self) -> &[u8] {
        self.offsets.as_ref()
    }

    #[inline]
    pub(crate) fn entries(&self) -> &[u8] {
        self.entries.as_ref()
    }

    #[inline]
    pub fn remaining_space(&self) -> usize {
        BLOCK_SIZE - (size_of::<u16>() + self.offsets.len() + self.entries.len())
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.remaining_space() < ENTRY_SIZE + self.offsets.len() + self.entries.len()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len() + self.entries.len() + size_of::<u16>()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == size_of::<u16>()
    }

    #[inline]
    pub fn num_entries(&self) -> u16 {
        self.num_entries
    }

    /// Check if an entry of given size will fit in this block
    #[inline]
    pub fn will_fit(&self, entry_size: usize) -> bool {
        // account for:
        // 1. the entry data itself
        // 2. the flag byte
        // 3. the offset entry (u16)
        // 4. existing data (offsets + entries + num_entries)
        let required_space = entry_size + 1 + size_of::<u16>();

        let available = BLOCK_SIZE - self.len();

        available >= required_space
    }
}

// BlockBuilder - writes directly to mmap (zero-copy)
impl<'a> BlockBuilder<'a> {
    /// Create a new builder from a mmap slice (must be BLOCK_SIZE bytes)
    pub(crate) fn new(buffer: &'a mut [u8]) -> Self {
        assert_eq!(
            buffer.len(),
            BLOCK_SIZE,
            "Buffer must be exactly BLOCK_SIZE"
        );

        // Zero the buffer
        buffer.fill(0);

        Self {
            buffer,
            num_entries: 0,
            offsets: Vec::new(),
            entries: Vec::new(),
        }
    }

    /// Create a BlockBuilder from pre-built vecs (zero-copy construction)
    /// This avoids copying from PendingEntry -> BlockBuilder
    pub(crate) fn from_parts(buffer: &'a mut [u8], offsets: Vec<u16>, entries: Vec<u8>) -> Self {
        assert_eq!(
            buffer.len(),
            BLOCK_SIZE,
            "Buffer must be exactly BLOCK_SIZE"
        );

        // Zero the buffer
        buffer.fill(0);

        let num_entries = offsets.len() as u16;

        Self {
            buffer,
            num_entries,
            offsets,
            entries,
        }
    }

    /// Add an entry to the block (stored in Vec, written during finalize)
    /// This matches Block::add_entry() behavior but uses Vec instead of
    /// BytesMut
    pub(crate) fn add_entry(&mut self, entry: &[u8], flag: EntryFlag) -> Result<(), BlockError> {
        let entry_size = entry.len() + size_of::<u8>(); // data + flag byte
        let required_space = entry_size + size_of::<u16>(); // entry + offset

        // Calculate current space used
        let current_used = size_of::<u16>() // num_entries header
            + (self.offsets.len() * size_of::<u16>())
            + self.entries.len();

        // Check if entry will fit
        if current_used + required_space > BLOCK_SIZE {
            return if self.num_entries == 0 {
                Err(TooLargeForBlock)
            } else {
                Err(BlockFull)
            };
        }

        // Calculate cumulative offset (same as Block does)
        let current_offset = if self.num_entries > 0 {
            self.offsets[self.offsets.len() - 1]
        } else {
            0
        };
        let next_offset = current_offset + (entry_size as u16);

        // Append offset (matches Block::add_entry)
        self.offsets.push(next_offset);

        // Append flag + data (matches Block::add_entry)
        self.entries.push(flag as u8);
        self.entries.extend_from_slice(entry);

        self.num_entries += 1;
        Ok(())
    }

    /// Finalize the block by writing to the mmap buffer
    /// Layout: [num_entries:2][offsets...][entries...]
    pub(crate) fn finalize(self) {
        // Write num_entries header
        let num_entries_bytes = self.num_entries.to_le_bytes();
        self.buffer[0] = num_entries_bytes[0];
        self.buffer[1] = num_entries_bytes[1];

        // Write all offsets
        let mut offset_pos = 2;
        for offset in &self.offsets {
            let offset_bytes = offset.to_le_bytes();
            self.buffer[offset_pos] = offset_bytes[0];
            self.buffer[offset_pos + 1] = offset_bytes[1];
            offset_pos += 2;
        }

        // Write all entries
        let entries_start = 2 + (self.offsets.len() * 2);
        self.buffer[entries_start..entries_start + self.entries.len()]
            .copy_from_slice(&self.entries);
    }

    #[inline]
    pub(crate) fn num_entries(&self) -> u16 {
        self.num_entries
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.num_entries == 0
    }
}

// ReadOnlyBlock methods - same interface as Block for reading
impl ReadOnlyBlock {
    #[inline]
    pub fn get(&self, index: usize) -> Option<(EntryFlag, &[u8])> {
        if index >= self.num_entries as usize {
            return None;
        }

        let start_offset = if index == 0 {
            0
        } else {
            let offset_idx = (index - 1) * 2;
            u16::from_le_bytes([self.offsets[offset_idx], self.offsets[offset_idx + 1]]) as usize
        };

        let end_offset = if index < self.num_entries as usize - 1 {
            let offset_idx = index * 2;
            u16::from_le_bytes([self.offsets[offset_idx], self.offsets[offset_idx + 1]]) as usize
        } else {
            self.entries.len()
        };

        let entry_data = &self.entries[start_offset..end_offset];
        let flag = match entry_data[0] {
            | 0 => EntryFlag::Complete,
            | 1 => EntryFlag::Start,
            | 2 => EntryFlag::Middle,
            | 3 => EntryFlag::End,
            | _ => unreachable!("invalid entry flag"),
        };

        Some((flag, &entry_data[1..]))
    }

    #[inline]
    pub fn num_entries(&self) -> u16 {
        self.num_entries
    }

    #[inline]
    pub fn iter(&self) -> BlockIterator<'_> {
        BlockIterator {
            entries: self.entries.as_ref(),
            offsets: self.offsets.as_ref(),
            current: 0,
            num_entries: self.num_entries,
        }
    }

    #[inline]
    pub(crate) fn offsets(&self) -> &[u8] {
        self.offsets.as_ref()
    }

    #[inline]
    pub(crate) fn entries(&self) -> &[u8] {
        self.entries.as_ref()
    }
}

impl Deserializer for ReadOnlyBlock {
    fn deserialize(payload: Bytes) -> Self {
        // First two bytes are num_entries
        let num_entries = u16::from_le_bytes([payload[0], payload[1]]);

        // Use zero-copy slices from the payload - just Arc increments!
        let (offsets, entries) = if num_entries > 0 {
            // Read all offsets first
            let offsets_end = size_of::<u16>() + (num_entries as usize * size_of::<u16>());
            let offsets_slice = payload.slice(size_of::<u16>()..offsets_end);

            // Calculate entries size using last offset
            let last_offset = u16::from_le_bytes([
                offsets_slice[offsets_slice.len() - 2],
                offsets_slice[offsets_slice.len() - 1],
            ]) as usize;

            // Zero-copy entries slice - just Arc increments!
            let entries_slice = payload.slice(offsets_end..offsets_end + last_offset);

            (offsets_slice, entries_slice)
        } else {
            (Bytes::new(), Bytes::new())
        };

        ReadOnlyBlock {
            num_entries,
            offsets,
            entries,
        }
    }
}

/// An iterator over the entries in a block.
pub struct BlockIterator<'a> {
    /// Reference to the entries data
    entries: &'a [u8],
    /// Reference to the offsets data
    offsets: &'a [u8],
    /// Current entry index
    current: usize,
    /// Total number of entries
    num_entries: u16,
}

impl<'a> Iterator for BlockIterator<'a> {
    type Item = (EntryFlag, &'a [u8]);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.num_entries as usize {
            return None;
        }

        // Get the current entry's start offset
        let start_offset = if self.current == 0 {
            0
        } else {
            let offset_idx = (self.current - 1) * 2;
            u16::from_le_bytes([self.offsets[offset_idx], self.offsets[offset_idx + 1]]) as usize
        };

        // Get the end offset (either from next entry or end of entries)
        let end_offset = if self.current < self.num_entries as usize {
            let offset_idx = self.current * 2;
            u16::from_le_bytes([self.offsets[offset_idx], self.offsets[offset_idx + 1]]) as usize
        } else {
            self.entries.len()
        };

        let entry_data = &self.entries[start_offset..end_offset];
        let flag = match entry_data[0] {
            | 0 => EntryFlag::Complete,
            | 1 => EntryFlag::Start,
            | 2 => EntryFlag::Middle,
            | 3 => EntryFlag::End,
            | _ => unreachable!("invalid entry flag"),
        };

        self.current += 1;
        Some((flag, &entry_data[1..]))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.num_entries as usize - self.current;
        (remaining, Some(remaining))
    }
}

#[cfg(test)]
#[allow(clippy::question_mark_used)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::undocumented_unsafe_blocks)]
mod tests {
    use super::*;

    #[test]
    fn test_new_block() {
        let block = Block::new();
        assert_eq!(block.num_entries, 0);
        assert_eq!(block.len(), 2);
        assert!(block.is_empty());
        let expected_remaining = BLOCK_SIZE - size_of::<u16>();
        assert_eq!(block.remaining_space(), expected_remaining);
    }

    #[test]
    fn test_add_entry_success() {
        let mut block = Block::new();
        let entry = [1, 2, 3, 4];
        assert!(block.add_entry(&entry, EntryFlag::Complete).is_ok());
        assert_eq!(block.num_entries, 1);
        assert_eq!(block.entries(), [0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_add_entry_block_full() {
        let mut block = Block::new();
        let entry = [0u8; BLOCK_SIZE];
        assert!(matches!(
            block.add_entry(&entry, EntryFlag::Complete),
            Err(TooLargeForBlock)
        ));
    }

    #[test]
    fn test_add_entry_too_large_for_block() {
        let mut block = Block::new();
        let entry = vec![0u8; BLOCK_SIZE - size_of::<u16>() + 1];
        assert!(matches!(
            block.add_entry(&entry, EntryFlag::Complete),
            Err(TooLargeForBlock)
        ));
    }

    #[test]
    fn test_add_entry_multiple_entries() {
        let mut block = Block::new();
        let entry1 = [1, 2, 3, 4];
        let entry2 = [5, 6, 7, 8];
        assert!(block.add_entry(&entry1, EntryFlag::Complete).is_ok());
        assert!(block.add_entry(&entry2, EntryFlag::Complete).is_ok());
        assert_eq!(block.num_entries, 2);
        assert_eq!(block.entries(), &[0, 1, 2, 3, 4, 0, 5, 6, 7, 8]);
    }

    #[test]
    fn test_add_entry_remaining_space() {
        let mut block = Block::new();
        let entry = [1, 2, 3, 4];
        block.add_entry(&entry, EntryFlag::Complete).unwrap();
        let expected_remaining =
            BLOCK_SIZE - (size_of::<u16>() + block.offsets.len() + block.entries.len());
        assert_eq!(block.remaining_space(), expected_remaining);
    }

    #[test]
    fn test_add_entry_is_full() {
        let mut block = Block::new();
        let entry = [0u8; BLOCK_SIZE - MAX_ENTRIES];
        block.add_entry(&entry, EntryFlag::Complete).unwrap();
        assert!(block.is_full());
    }

    #[test]
    fn test_finalize_empty_block() {
        let block = Block::new();
        let mut buffer = vec![0u8; BLOCK_SIZE];
        unsafe {
            block.finalize(buffer.as_mut_ptr());
        }
        assert_eq!(buffer.len(), BLOCK_SIZE);
        assert_eq!(&buffer[2..], &[0u8; BLOCK_SIZE - 2]);
    }

    #[test]
    fn test_finalize_single_entry() {
        let mut block = Block::new();
        let entry = [1, 2, 3, 4];
        block.add_entry(&entry, EntryFlag::Complete).unwrap();
        let mut buffer = vec![0u8; BLOCK_SIZE];
        unsafe {
            block.finalize(buffer.as_mut_ptr());
        }
        assert_eq!(buffer.len(), BLOCK_SIZE);
        assert_eq!(buffer[0..2], (1u16).to_le_bytes());
        assert_eq!(buffer[2..4], (5u16).to_le_bytes());
        assert_eq!(buffer[5..9], entry); // skip the entry byte
        assert_eq!(&buffer[9..], &[0u8; BLOCK_SIZE - 9]);
    }

    #[test]
    fn test_finalize_multiple_entries() {
        let mut block = Block::new();
        let entry1 = [1, 2, 3, 4];
        let entry2 = [5, 6, 7, 8];
        block.add_entry(&entry1, EntryFlag::Complete).unwrap();
        block.add_entry(&entry2, EntryFlag::Complete).unwrap();
        let mut buffer = vec![0u8; BLOCK_SIZE];
        unsafe {
            block.finalize(buffer.as_mut_ptr());
        }

        assert_eq!(buffer.len(), BLOCK_SIZE);
        assert_eq!(buffer[0..2], (2u16).to_le_bytes());
        assert_eq!(buffer[2..4], (5u16).to_le_bytes());
        assert_eq!(buffer[4..6], (10u16).to_le_bytes());
        assert_eq!(buffer[7..11], entry1);
        assert_eq!(buffer[12..16], entry2);
        assert_eq!(&buffer[16..], &[0u8; BLOCK_SIZE - 16]);
    }

    #[test]
    fn test_finalize_full_block() {
        let mut block = Block::new();

        // account for the offsets and entry bytes
        let entry = vec![0u8; block.remaining_space() - 4];
        block.add_entry(&entry, EntryFlag::Complete).unwrap();

        let mut buffer = vec![0u8; BLOCK_SIZE];
        unsafe {
            block.finalize(buffer.as_mut_ptr());
        }

        assert_eq!(buffer.len(), BLOCK_SIZE);
        assert_eq!(buffer[0..2], (1u16).to_le_bytes());

        // entry plus the entry byte
        assert_eq!(
            u16::from_le_bytes(buffer[2..4].try_into().unwrap()),
            1 + entry.len() as u16
        );

        //
        assert_eq!(buffer[5], EntryFlag::Complete as u8);
        // assert_eq!(buffer[4..BLOCK_SIZE], entry);
    }

    #[test]
    fn test_finalize_partial_block() {
        let mut block = Block::new();
        let entry = [1, 2, 3, 4];
        block.add_entry(&entry, EntryFlag::Complete).unwrap();
        let entry2 = [5, 6, 7, 8, 9, 10];
        block.add_entry(&entry2, EntryFlag::Complete).unwrap();
        let mut buffer = vec![0u8; BLOCK_SIZE];
        unsafe {
            block.finalize(buffer.as_mut_ptr());
        }
        assert_eq!(buffer.len(), BLOCK_SIZE);
        assert_eq!(buffer[0..2], (2u16).to_le_bytes());
        assert_eq!(buffer[2..4], (5u16).to_le_bytes());
        assert_eq!(buffer[4..6], (12u16).to_le_bytes());
        assert_eq!(buffer[7..11], entry);
        assert_eq!(buffer[12..18], entry2);
        assert_eq!(&buffer[18..], vec![0u8; 4078]);
    }

    #[test]
    fn test_iterator_empty_block() {
        let block = Block::new();
        let mut iter = block.iter();
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iterator_single_entry() {
        let mut block = Block::new();
        let entry = [1, 2, 3, 4];
        block.add_entry(&entry, EntryFlag::Complete).unwrap();

        let mut iter = block.iter();
        assert_eq!(iter.next(), Some((EntryFlag::Complete, &entry[..])));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iterator_multiple_entries() {
        let mut block = Block::new();
        let entry1 = [1, 2, 3, 4];
        let entry2 = [5, 6, 7, 8];
        block.add_entry(&entry1, EntryFlag::Complete).unwrap();
        block.add_entry(&entry2, EntryFlag::Complete).unwrap();

        let mut iter = block.iter();
        assert_eq!(iter.next(), Some((EntryFlag::Complete, &entry1[..])));
        assert_eq!(iter.next(), Some((EntryFlag::Complete, &entry2[..])));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iterator_size_hint() {
        let mut block = Block::new();
        block.add_entry(&[1, 2, 3, 4], EntryFlag::Complete).unwrap();
        block.add_entry(&[5, 6, 7, 8], EntryFlag::Complete).unwrap();

        let mut iter = block.iter();
        assert_eq!(iter.size_hint(), (2, Some(2)));
        iter.next();
        assert_eq!(iter.size_hint(), (1, Some(1)));
        iter.next();
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }

    #[test]
    fn test_deserialize_empty_block() {
        let mut data = BytesMut::with_capacity(BLOCK_SIZE);
        data.put_u16_le(0); // num_entries = 0
        data.resize(BLOCK_SIZE, 0);

        let block = ReadOnlyBlock::deserialize(data.freeze());
        assert_eq!(block.num_entries, 0);
        assert!(block.offsets().is_empty());
        assert!(block.entries().is_empty());
    }

    #[test]
    fn test_deserialize_single_entry() {
        let mut data = BytesMut::with_capacity(BLOCK_SIZE);
        data.put_u16_le(1); // num_entries = 1
        data.put_u16_le(5); // offset to end of first entry

        // Entry data
        data.put_slice(b"hello");
        data.resize(BLOCK_SIZE, 0);

        let block = ReadOnlyBlock::deserialize(data.freeze());
        assert_eq!(block.num_entries, 1);
        assert_eq!(block.offsets().len(), 2); // one u16 offset
        assert_eq!(block.entries().len(), 5); // "hello"
    }

    #[test]
    fn test_deserialize_multiple_entries() {
        let mut data = BytesMut::with_capacity(BLOCK_SIZE);
        data.put_u16_le(2); // num_entries = 2
        data.put_u16_le(5); // offset to end of first entry
        data.put_u16_le(8); // offset to end of second entry

        // Entry data
        data.put_slice(b"hello"); // first entry
        data.put_slice(b"123"); // second entry
        data.resize(BLOCK_SIZE, 0);

        let block = ReadOnlyBlock::deserialize(data.freeze());
        assert_eq!(block.num_entries, 2);
        assert_eq!(block.offsets().len(), 4); // two u16 offsets
        assert_eq!(block.entries().len(), 8); // "hello123"
    }

    #[test]
    fn test_get_empty_block() {
        let block = Block::new();
        assert_eq!(block.get(0), None);
        assert_eq!(block.get(1), None);
    }

    #[test]
    fn test_get_single_entry() {
        let mut block = Block::new();
        let entry = [1, 2, 3, 4];
        block.add_entry(&entry, EntryFlag::Complete).unwrap();

        assert_eq!(block.get(0), Some((EntryFlag::Complete, &entry[..])));
        assert_eq!(block.get(1), None);
    }

    #[test]
    fn test_get_multiple_entries() {
        let mut block = Block::new();
        let entry1 = [1, 2, 3, 4];
        let entry2 = [5, 6, 7, 8];
        let entry3 = [9, 10];

        block.add_entry(&entry1, EntryFlag::Complete).unwrap();
        block.add_entry(&entry2, EntryFlag::Complete).unwrap();
        block.add_entry(&entry3, EntryFlag::Complete).unwrap();

        assert_eq!(block.get(0), Some((EntryFlag::Complete, &entry1[..])));
        assert_eq!(block.get(1), Some((EntryFlag::Complete, &entry2[..])));
        assert_eq!(block.get(2), Some((EntryFlag::Complete, &entry3[..])));
        assert_eq!(block.get(3), None);
    }

    #[test]
    fn test_get_varying_sizes() {
        let mut block = Block::new();
        let entry1 = [1];
        let entry2 = [2, 3, 4, 5, 6];
        let entry3 = [7, 8, 9];

        block.add_entry(&entry1, EntryFlag::Complete).unwrap();
        block.add_entry(&entry2, EntryFlag::Complete).unwrap();
        block.add_entry(&entry3, EntryFlag::Complete).unwrap();

        assert_eq!(block.get(0), Some((EntryFlag::Complete, &entry1[..])));
        assert_eq!(block.get(1), Some((EntryFlag::Complete, &entry2[..])));
        assert_eq!(block.get(2), Some((EntryFlag::Complete, &entry3[..])));
    }

    #[test]
    fn test_get_out_of_bounds() {
        let mut block = Block::new();
        block.add_entry(&[1, 2, 3], EntryFlag::Complete).unwrap();

        assert_eq!(block.get(1), None);
        assert_eq!(block.get(usize::MAX), None);
    }

    #[test]
    fn test_get_matches_iterator() {
        let mut block = Block::new();
        let entries = vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8, 9]];

        for entry in &entries {
            block.add_entry(entry, EntryFlag::Complete).unwrap();
        }

        // Verify get() matches iterator results
        for (i, entry) in entries.iter().enumerate() {
            assert_eq!(block.get(i), Some((EntryFlag::Complete, entry.as_slice())));
        }
    }

    #[test]
    fn test_blockbuilder_matches_block_output() {
        // Test that BlockBuilder produces the same binary output as Block
        let data1 = b"hello";
        let data2 = b"world";
        let data3 = b"test";

        // Build with Block
        let mut block = Block::new();
        block.add_entry(data1, EntryFlag::Complete).unwrap();
        block.add_entry(data2, EntryFlag::Complete).unwrap();
        block.add_entry(data3, EntryFlag::Complete).unwrap();

        let mut block_buffer = vec![0u8; BLOCK_SIZE];
        unsafe {
            block.finalize(block_buffer.as_mut_ptr());
        }

        // Build with BlockBuilder
        let mut builder_buffer = vec![0u8; BLOCK_SIZE];
        {
            let mut builder = BlockBuilder::new(&mut builder_buffer);
            builder.add_entry(data1, EntryFlag::Complete).unwrap();
            builder.add_entry(data2, EntryFlag::Complete).unwrap();
            builder.add_entry(data3, EntryFlag::Complete).unwrap();
            builder.finalize();
        }

        // Compare the outputs - they should be identical
        assert_eq!(
            &block_buffer[..],
            &builder_buffer[..],
            "BlockBuilder and Block should produce identical output"
        );
    }
}
