// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

use std::io;

use thiserror::Error;

use crate::segment::BlockType;

#[derive(Error, Debug)]
pub enum ManifestError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("corrupted manifest header")]
    CorruptedHeader,
    #[error("invalid magic number: expected 0x43455349, got {0:#x}")]
    InvalidMagic(u32),
    #[error("unsupported manifest version: {0}")]
    UnsupportedVersion(u32),
    #[error("CRC mismatch: expected {expected:#x}, got {actual:#x}")]
    CrcMismatch { expected: u32, actual: u32 },
    #[error("invalid VersionEdit type: {0:#x}")]
    InvalidEditType(u8),
    #[error("segment error: {0}")]
    SegmentError(#[from] SegmentError),
}

#[derive(Error, Debug)]
pub enum CesiumError {
    #[error("memtable error")]
    MemtableError(MemtableError),
    #[error("fs error")]
    FsError(FsError),
    #[error("block error")]
    BlockError(BlockError),
    #[error("segment error")]
    SegmentError(SegmentError),
    #[error("journal error")]
    JournalError(JournalError),
    #[error("compaction error: {0}")]
    CompactionError(#[from] CompactionError),
    #[error("manifest error: {0}")]
    ManifestError(#[from] ManifestError),
    #[error("initialization error: {0}")]
    InitializationError(String),
}

#[derive(Error, Debug)]
pub enum CompactionError {
    #[error("compaction not initialized")]
    NotInitialized,
    #[error("compaction job failed: {0}")]
    JobFailed(String),
    #[error("version changed during compaction")]
    VersionChanged,
    #[error("segment error: {0}")]
    SegmentError(#[from] SegmentError),
    #[error("I/O error: {0}")]
    IoError(#[from] io::Error),
    #[error("shutting down")]
    ShuttingDown,
}

#[derive(Error, Debug)]
pub enum MemtableError {
    #[error("data insertion would exceed maximum capacity")]
    DataExceedsMaximum,
    #[error("memtable is frozen")]
    MemtableIsFrozen,
}

#[derive(Error, Debug)]
pub enum BlockError {
    #[error("block is corrupted")]
    CorruptedBlock,
    #[error("block is full")]
    BlockFull,
    #[error("entry is too large for block")]
    TooLargeForBlock,
}

#[derive(Error, Debug)]
pub enum JournalError {
    #[error("journal entry too short")]
    EntryTooShort,
    #[error("invalid journal entry type")]
    InvalidEntryType,
}

#[derive(Error, Debug)]
pub enum SegmentError {
    #[error("segment is full")]
    InsufficientSpace,
    #[error("segment must be multiple of 4096")]
    InvalidSize,
    #[error("read-ahead cache size is invalid")]
    InvalidReadAheadSize,
    #[error("can't create writer for {0} block id {1}")]
    CantCreateWriter(BlockType, u64),
    #[error("can't create reader")]
    CantCreateReader,
    #[error("read out of bounds")]
    ReadOutOfBounds,
    #[error("write out of bounds")]
    WriteOutOfBounds,
    #[error("missing key")]
    MissingKey,
    #[error("corrupted block")]
    CorruptedBlock,
    #[error("segment is closing, no more blocks can be written")]
    Closing,
    #[error("segment is not closing")]
    NotClosing,
    #[error("segment is read-only")]
    ReadOnly,
    #[error("invalid key format")]
    InvalidKey,
    #[error("io error")]
    IoError(io::Error),
}

#[derive(Error, Debug)]
pub enum FsError {
    #[error("os i/o error")]
    IoError(io::Error),
    #[error("invalid header format")]
    InvalidHeaderFormat(String),
    #[error("no contiguous space available, can't find a contiguous block big enough")]
    NoContiguousSpace,
    #[error("block is too fragmented")]
    FragmentationLimit,
    #[error("filesystem is full")]
    StorageExhausted,
    #[error("frange is already open")]
    FRangeAlreadyOpen,
    #[error("frange is already closed")]
    FRangeAlreadyClosed,
    #[error("frange not found")]
    FRangeNotFound,
    #[error("frange is still open")]
    FRangeStillOpen,
    #[error("read out of bounds")]
    ReadOutOfBounds,
    #[error("write out of bounds")]
    WriteOutOfBounds,
    #[error("block index out of bounds")]
    BlockIndexOutOfBounds,
    #[error("no adjacent space available, can't find a free range after metadata")]
    NoAdjacentSpace,
    #[error("adjacent space isn't big enough")]
    InsufficientSpace,
    #[error("metadata too large")]
    MetadataTooLarge,
    #[error("no free space available")]
    NoFreeSpace,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cesium_error_display() {
        let err = CesiumError::MemtableError(MemtableError::DataExceedsMaximum);
        assert_eq!(err.to_string(), "memtable error");

        let err = CesiumError::BlockError(BlockError::BlockFull);
        assert_eq!(err.to_string(), "block error");

        let err = CesiumError::SegmentError(SegmentError::InsufficientSpace);
        assert_eq!(err.to_string(), "segment error");
    }

    #[test]
    fn test_memtable_error_display() {
        let err = MemtableError::DataExceedsMaximum;
        assert_eq!(
            err.to_string(),
            "data insertion would exceed maximum capacity"
        );

        let err = MemtableError::MemtableIsFrozen;
        assert_eq!(err.to_string(), "memtable is frozen");
    }

    #[test]
    fn test_block_error_display() {
        let err = BlockError::CorruptedBlock;
        assert_eq!(err.to_string(), "block is corrupted");

        let err = BlockError::BlockFull;
        assert_eq!(err.to_string(), "block is full");

        let err = BlockError::TooLargeForBlock;
        assert_eq!(err.to_string(), "entry is too large for block");
    }

    #[test]
    fn test_journal_error_display() {
        let err = JournalError::EntryTooShort;
        assert_eq!(err.to_string(), "journal entry too short");

        let err = JournalError::InvalidEntryType;
        assert_eq!(err.to_string(), "invalid journal entry type");
    }

    #[test]
    fn test_segment_error_display() {
        let err = SegmentError::InsufficientSpace;
        assert_eq!(err.to_string(), "segment is full");

        let err = SegmentError::InvalidSize;
        assert_eq!(err.to_string(), "segment must be multiple of 4096");

        let err = SegmentError::InvalidReadAheadSize;
        assert_eq!(err.to_string(), "read-ahead cache size is invalid");

        let err = SegmentError::ReadOutOfBounds;
        assert_eq!(err.to_string(), "read out of bounds");

        let err = SegmentError::WriteOutOfBounds;
        assert_eq!(err.to_string(), "write out of bounds");

        let err = SegmentError::MissingKey;
        assert_eq!(err.to_string(), "missing key");

        let err = SegmentError::CorruptedBlock;
        assert_eq!(err.to_string(), "corrupted block");

        let err = SegmentError::Closing;
        assert_eq!(
            err.to_string(),
            "segment is closing, no more blocks can be written"
        );

        let err = SegmentError::NotClosing;
        assert_eq!(err.to_string(), "segment is not closing");

        let err = SegmentError::ReadOnly;
        assert_eq!(err.to_string(), "segment is read-only");
    }

    #[test]
    fn test_segment_error_with_block_type() {
        use crate::segment::BlockType;

        let err = SegmentError::CantCreateWriter(BlockType::Key, 42);
        assert!(err.to_string().contains("key"));
        assert!(err.to_string().contains("42"));
    }

    #[test]
    fn test_fs_error_display() {
        let err = FsError::InvalidHeaderFormat("bad header".to_string());
        let err_str = err.to_string();
        assert!(
            err_str.contains("invalid header format"),
            "Error string: {}",
            err_str
        );
        // Note: The error format doesn't include the string parameter in display

        let err = FsError::NoContiguousSpace;
        assert_eq!(
            err.to_string(),
            "no contiguous space available, can't find a contiguous block big enough"
        );

        let err = FsError::FragmentationLimit;
        assert_eq!(err.to_string(), "block is too fragmented");

        let err = FsError::StorageExhausted;
        assert_eq!(err.to_string(), "filesystem is full");

        let err = FsError::FRangeAlreadyOpen;
        assert_eq!(err.to_string(), "frange is already open");

        let err = FsError::FRangeAlreadyClosed;
        assert_eq!(err.to_string(), "frange is already closed");

        let err = FsError::FRangeNotFound;
        assert_eq!(err.to_string(), "frange not found");

        let err = FsError::FRangeStillOpen;
        assert_eq!(err.to_string(), "frange is still open");

        let err = FsError::ReadOutOfBounds;
        assert_eq!(err.to_string(), "read out of bounds");

        let err = FsError::WriteOutOfBounds;
        assert_eq!(err.to_string(), "write out of bounds");

        let err = FsError::BlockIndexOutOfBounds;
        assert_eq!(err.to_string(), "block index out of bounds");

        let err = FsError::NoAdjacentSpace;
        assert_eq!(
            err.to_string(),
            "no adjacent space available, can't find a free range after metadata"
        );

        let err = FsError::InsufficientSpace;
        assert_eq!(err.to_string(), "adjacent space isn't big enough");

        let err = FsError::MetadataTooLarge;
        assert_eq!(err.to_string(), "metadata too large");

        let err = FsError::NoFreeSpace;
        assert_eq!(err.to_string(), "no free space available");
    }

    #[test]
    fn test_error_wrapping() {
        let memtable_err = MemtableError::DataExceedsMaximum;
        let cesium_err = CesiumError::MemtableError(memtable_err);

        match cesium_err {
            | CesiumError::MemtableError(MemtableError::DataExceedsMaximum) => {},
            | _ => panic!("expected DataExceedsMaximum"),
        }
    }

    #[test]
    fn test_error_debug() {
        let err = MemtableError::DataExceedsMaximum;
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("DataExceedsMaximum"));

        let err = BlockError::BlockFull;
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("BlockFull"));
    }

    #[test]
    fn test_io_error_wrapping() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let segment_err = SegmentError::IoError(io_err);

        match segment_err {
            | SegmentError::IoError(_) => {},
            | _ => panic!("expected IoError variant"),
        }
    }

    #[test]
    fn test_fs_io_error_wrapping() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let fs_err = FsError::IoError(io_err);

        match fs_err {
            | FsError::IoError(_) => {},
            | _ => panic!("expected IoError variant"),
        }
    }

    #[test]
    fn test_all_memtable_error_variants() {
        let errors = vec![
            MemtableError::DataExceedsMaximum,
            MemtableError::MemtableIsFrozen,
        ];

        for err in errors {
            // just verify they can all be created and displayed
            let _ = err.to_string();
        }
    }

    #[test]
    fn test_all_block_error_variants() {
        let errors = vec![
            BlockError::CorruptedBlock,
            BlockError::BlockFull,
            BlockError::TooLargeForBlock,
        ];

        for err in errors {
            let _ = err.to_string();
        }
    }

    #[test]
    fn test_all_journal_error_variants() {
        let errors = vec![JournalError::EntryTooShort, JournalError::InvalidEntryType];

        for err in errors {
            let _ = err.to_string();
        }
    }

    #[test]
    fn test_cesium_error_from_memtable() {
        let mem_err = MemtableError::MemtableIsFrozen;
        let cesium_err = CesiumError::MemtableError(mem_err);

        assert!(matches!(
            cesium_err,
            CesiumError::MemtableError(MemtableError::MemtableIsFrozen)
        ));
    }

    #[test]
    fn test_cesium_error_from_block() {
        let block_err = BlockError::BlockFull;
        let cesium_err = CesiumError::BlockError(block_err);

        assert!(matches!(
            cesium_err,
            CesiumError::BlockError(BlockError::BlockFull)
        ));
    }

    #[test]
    fn test_cesium_error_from_segment() {
        let segment_err = SegmentError::ReadOnly;
        let cesium_err = CesiumError::SegmentError(segment_err);

        assert!(matches!(
            cesium_err,
            CesiumError::SegmentError(SegmentError::ReadOnly)
        ));
    }
}
