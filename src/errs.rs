// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

use std::io;

use thiserror::Error;
use crate::segment::BlockType;

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
    #[error("can't create frange for {0} block id {1}")]
    CantCreateFRange(BlockType, u64, FsError),
    #[error("can't open frange for {0} block id {1}")]
    CantOpenFRange(BlockType, u64, FsError),
    #[error("can't read frange for {0} block id {1}")]
    CantReadFRange(BlockType, u64, FsError),
    #[error("can't create writer for {0} block id {1}")]
    CantCreateWriter(BlockType, u64),
    #[error("read out of bounds")]
    ReadOutOfBounds,
    #[error("write out of bounds")]
    WriteOutOfBounds,
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
