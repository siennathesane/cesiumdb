use bytes::{BufMut, BytesMut};

use crate::{
    errs::CesiumError,
};

pub(crate) const BLOCK_SIZE: usize = 4096;
pub(crate) const ENTRY_SIZE: usize = size_of::<u16>();
const MAX_ENTRIES: usize = BLOCK_SIZE / ENTRY_SIZE;

/// A single block of data in the table. The block is a fixed size and is
/// divided into two parts:
/// 1. The offsets: a list of 4-byte integers that point to the start of each entry in the block.
/// 2. The entries: the actual data stored in the block.
pub(crate) struct Block {
    /// The number of entries in the block.
    num_entries: u16,
    /// The entry offsets, it's just a [u16].
    offsets: BytesMut,
    /// The actual entries, it's just a [[Bytes]].
    entries: BytesMut,
}

impl Block {
    /// Create a new block. Creating the block will allocate the necessary
    /// memory upfront.
    pub(crate) fn new() -> Self {
        Block {
            num_entries: 0,
            offsets: BytesMut::with_capacity(MAX_ENTRIES),
            entries: BytesMut::with_capacity(BLOCK_SIZE - MAX_ENTRIES),
        }
    }

    /// Add an entry to the block. If the block is full, an error will be returned.
    pub(crate) fn add_entry(&mut self, entry: &[u8]) -> Result<(), CesiumError> {
        // check if the entry itself is too large to fit in an empty block
        if entry.len() + size_of::<u16>() > BLOCK_SIZE {
            return Err(CesiumError::TooLargeForBlock);
        }

        // check if the block is full
        if self.is_full() {
            return Err(CesiumError::BlockFull);
        }

        // check if the entry can fit in the remaining space of the block
        if entry.len() + self.entries.len() + self.offsets.len() + size_of::<u16>() > BLOCK_SIZE {
            return Err(CesiumError::TooLargeForBlock);
        }

        // calculate the next offset
        let mut current_offset = 0;
        if self.num_entries > 0 {
            let offset = &self.offsets[self.offsets.len() - 2..];
            current_offset = u16::from_le_bytes([offset[0], offset[1]]);
        }
        let next_offset = current_offset + (entry.len() as u16);

        // add the entry and update the offsets
        self.offsets.put_u16_le(next_offset);
        self.entries.put_slice(entry);
        self.num_entries += 1;

        Ok(())
    }

    /// Finalize the block by writing directly to the provided memory location.
    ///
    /// # Safety
    /// - dst must be valid for BLOCK_SIZE bytes
    /// - dst must be properly aligned
    /// - dst must not overlap with any source data
    pub(crate) unsafe fn finalize(&self, dst: *mut u8) {
        // write num_entries
        std::ptr::copy_nonoverlapping(
            self.num_entries.to_le_bytes().as_ptr(),
            dst,
            size_of::<u16>()
        );

        // write offsets
        std::ptr::copy_nonoverlapping(
            self.offsets.as_ptr(),
            dst.add(size_of::<u16>()),
            self.offsets.len()
        );

        // write entries
        std::ptr::copy_nonoverlapping(
            self.entries.as_ptr(),
            dst.add(size_of::<u16>() + self.offsets.len()),
            self.entries.len()
        );

        // zero remaining space
        let written = size_of::<u16>() + self.offsets.len() + self.entries.len();
        if written < BLOCK_SIZE {
            std::ptr::write_bytes(
                dst.add(written),
                0,
                BLOCK_SIZE - written
            );
        }
    }

    /// Returns an iterator over the entries in the block.
    #[inline]
    pub fn iter(&self) -> BlockIterator {
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
    pub(crate) fn offsets(&self) -> &[u8] {
        self.offsets.as_ref()
    }

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
    type Item = &'a [u8];

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
            u16::from_le_bytes([
                self.offsets[offset_idx],
                self.offsets[offset_idx + 1]
            ]) as usize
        };

        // Get the end offset (either from next entry or end of entries)
        let end_offset = if self.current < self.num_entries as usize {
            let offset_idx = self.current * 2;
            u16::from_le_bytes([
                self.offsets[offset_idx],
                self.offsets[offset_idx + 1]
            ]) as usize
        } else {
            self.entries.len()
        };

        self.current += 1;
        Some(&self.entries[start_offset..end_offset])
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.num_entries as usize - self.current;
        (remaining, Some(remaining))
    }
}

#[cfg(test)]
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
        assert!(block.add_entry(&entry).is_ok());
        assert_eq!(block.num_entries, 1);
        assert_eq!(block.entries(), &entry);
    }

    #[test]
    fn test_add_entry_block_full() {
        let mut block = Block::new();
        let entry = [0u8; BLOCK_SIZE];
        assert!(matches!(block.add_entry(&entry), Err(CesiumError::TooLargeForBlock)));
    }

    #[test]
    fn test_add_entry_too_large_for_block() {
        let mut block = Block::new();
        let entry = vec![0u8; BLOCK_SIZE - size_of::<u16>() + 1];
        assert!(matches!(block.add_entry(&entry), Err(CesiumError::TooLargeForBlock)));
    }

    #[test]
    fn test_add_entry_multiple_entries() {
        let mut block = Block::new();
        let entry1 = [1, 2, 3, 4];
        let entry2 = [5, 6, 7, 8];
        assert!(block.add_entry(&entry1).is_ok());
        assert!(block.add_entry(&entry2).is_ok());
        assert_eq!(block.num_entries, 2);
        assert_eq!(block.entries(), &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_add_entry_remaining_space() {
        let mut block = Block::new();
        let entry = [1, 2, 3, 4];
        block.add_entry(&entry).unwrap();
        let expected_remaining = BLOCK_SIZE - (size_of::<u16>() + block.offsets.len() + block.entries.len());
        assert_eq!(block.remaining_space(), expected_remaining);
    }

    #[test]
    fn test_add_entry_is_full() {
        let mut block = Block::new();
        let entry = [0u8; BLOCK_SIZE - MAX_ENTRIES];
        block.add_entry(&entry).unwrap();
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
        block.add_entry(&entry).unwrap();
        let mut buffer = vec![0u8; BLOCK_SIZE];
        unsafe {
            block.finalize(buffer.as_mut_ptr());
        }
        assert_eq!(buffer.len(), BLOCK_SIZE);
        assert_eq!(buffer[0..2], (1u16).to_le_bytes());
        assert_eq!(buffer[2..4], (4u16).to_le_bytes());
        assert_eq!(buffer[4..8], entry);
        assert_eq!(&buffer[8..], &[0u8; BLOCK_SIZE - 8]);
    }

    #[test]
    fn test_finalize_multiple_entries() {
        let mut block = Block::new();
        let entry1 = [1, 2, 3, 4];
        let entry2 = [5, 6, 7, 8];
        block.add_entry(&entry1).unwrap();
        block.add_entry(&entry2).unwrap();
        let mut buffer = vec![0u8; BLOCK_SIZE];
        unsafe {
            block.finalize(buffer.as_mut_ptr());
        }
        assert_eq!(buffer.len(), BLOCK_SIZE);
        assert_eq!(buffer[0..2], (2u16).to_le_bytes());
        assert_eq!(buffer[2..4], (4u16).to_le_bytes());
        assert_eq!(buffer[4..6], (8u16).to_le_bytes());
        assert_eq!(buffer[6..10], entry1);
        assert_eq!(buffer[10..14], entry2);
        assert_eq!(&buffer[14..], &[0u8; BLOCK_SIZE - 14]);
    }

    #[test]
    fn test_finalize_full_block() {
        let mut block = Block::new();
        let entry = [0u8; BLOCK_SIZE - MAX_ENTRIES];
        block.add_entry(&entry).unwrap();
        let mut buffer = vec![0u8; BLOCK_SIZE];
        unsafe {
            block.finalize(buffer.as_mut_ptr());
        }
        assert_eq!(buffer.len(), BLOCK_SIZE);
        assert_eq!(buffer[0..2], (1u16).to_le_bytes());
        assert_eq!(buffer[2..4], ((BLOCK_SIZE - MAX_ENTRIES) as u16).to_le_bytes());
        // assert_eq!(buffer[4..BLOCK_SIZE], entry);
    }

    #[test]
    fn test_finalize_partial_block() {
        let mut block = Block::new();
        let entry = [1, 2, 3, 4];
        block.add_entry(&entry).unwrap();
        let entry2 = [5, 6, 7, 8, 9, 10];
        block.add_entry(&entry2).unwrap();
        let mut buffer = vec![0u8; BLOCK_SIZE];
        unsafe {
            block.finalize(buffer.as_mut_ptr());
        }
        assert_eq!(buffer.len(), BLOCK_SIZE);
        assert_eq!(buffer[0..2], (2u16).to_le_bytes());
        assert_eq!(buffer[2..4], (4u16).to_le_bytes());
        assert_eq!(buffer[4..6], (10u16).to_le_bytes());
        assert_eq!(buffer[6..10], entry);
        assert_eq!(buffer[10..16], entry2);
        assert_eq!(&buffer[16..], &[0u8; BLOCK_SIZE - 16]);
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
        block.add_entry(&entry).unwrap();

        let mut iter = block.iter();
        assert_eq!(iter.next(), Some(&entry[..]));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iterator_multiple_entries() {
        let mut block = Block::new();
        let entry1 = [1, 2, 3, 4];
        let entry2 = [5, 6, 7, 8];
        block.add_entry(&entry1).unwrap();
        block.add_entry(&entry2).unwrap();

        let mut iter = block.iter();
        assert_eq!(iter.next(), Some(&entry1[..]));
        assert_eq!(iter.next(), Some(&entry2[..]));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iterator_size_hint() {
        let mut block = Block::new();
        block.add_entry(&[1, 2, 3, 4]).unwrap();
        block.add_entry(&[5, 6, 7, 8]).unwrap();

        let mut iter = block.iter();
        assert_eq!(iter.size_hint(), (2, Some(2)));
        iter.next();
        assert_eq!(iter.size_hint(), (1, Some(1)));
        iter.next();
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }
}