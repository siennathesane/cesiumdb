use bytes::{Buf, BufMut, Bytes, BytesMut};

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

    /// Finalize the block by writing the number of entries, the offsets, and the entries
    pub(crate) fn finalize(&mut self) -> Bytes {
        let mut block = BytesMut::with_capacity(BLOCK_SIZE);
        block.put_u16_le(self.num_entries);
        block.put_slice(self.offsets.as_ref());
        block.put_slice(self.entries.as_ref());

        // zero out the remaining space in the block
        let remaining = BLOCK_SIZE - block.len();
        for _ in 0..remaining {
            block.put_u8(0);
        }

        block.freeze()
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
        let mut block = Block::new();
        let finalized = block.finalize();
        assert_eq!(finalized.len(), BLOCK_SIZE);
        assert_eq!(&finalized[2..], &[0u8; BLOCK_SIZE - 2]);
    }

    #[test]
    fn test_finalize_single_entry() {
        let mut block = Block::new();
        let entry = [1, 2, 3, 4];
        block.add_entry(&entry).unwrap();
        let finalized = block.finalize();
        assert_eq!(finalized.len(), BLOCK_SIZE);
        assert_eq!(finalized[0..2], (1u16).to_le_bytes());
        assert_eq!(finalized[2..4], (4u16).to_le_bytes());
        assert_eq!(finalized[4..8], entry);
        assert_eq!(&finalized[8..], &[0u8; BLOCK_SIZE - 8]);
    }

    #[test]
    fn test_finalize_multiple_entries() {
        let mut block = Block::new();
        let entry1 = [1, 2, 3, 4];
        let entry2 = [5, 6, 7, 8];
        block.add_entry(&entry1).unwrap();
        block.add_entry(&entry2).unwrap();
        let finalized = block.finalize();
        assert_eq!(finalized.len(), BLOCK_SIZE);
        assert_eq!(finalized[0..2], (2u16).to_le_bytes());
        assert_eq!(finalized[2..4], (4u16).to_le_bytes());
        assert_eq!(finalized[4..6], (8u16).to_le_bytes());
        assert_eq!(finalized[6..10], entry1);
        assert_eq!(finalized[10..14], entry2);
        assert_eq!(&finalized[14..], &[0u8; BLOCK_SIZE - 14]);
    }

    #[test]
    fn test_finalize_full_block() {
        let mut block = Block::new();
        let entry = [0u8; BLOCK_SIZE - MAX_ENTRIES];
        block.add_entry(&entry).unwrap();
        let finalized = block.finalize();
        assert_eq!(finalized.len(), BLOCK_SIZE);
        assert_eq!(finalized[0..2], (1u16).to_le_bytes());
        assert_eq!(finalized[2..4], ((BLOCK_SIZE - MAX_ENTRIES) as u16).to_le_bytes());
        // assert_eq!(finalized[4..BLOCK_SIZE], entry);
    }

    #[test]
    fn test_finalize_partial_block() {
        let mut block = Block::new();
        let entry = [1, 2, 3, 4];
        block.add_entry(&entry).unwrap();
        let entry2 = [5, 6, 7, 8, 9, 10];
        block.add_entry(&entry2).unwrap();
        let finalized = block.finalize();
        assert_eq!(finalized.len(), BLOCK_SIZE);
        assert_eq!(finalized[0..2], (2u16).to_le_bytes());
        assert_eq!(finalized[2..4], (4u16).to_le_bytes());
        assert_eq!(finalized[4..6], (10u16).to_le_bytes());
        assert_eq!(finalized[6..10], entry);
        assert_eq!(finalized[10..16], entry2);
        assert_eq!(&finalized[16..], &[0u8; BLOCK_SIZE - 16]);
    }
}