// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

use std::io;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CesiumError {
    #[error("os i/o error")]
    IoError(io::Error),
    #[error("data insertion would exceed maximum capacity")]
    DataExceedsMaximum,
    #[error("memtable is frozen")]
    MemtableIsFrozen,
    #[error("block is full")]
    BlockFull,
    #[error("entry is too large for block")]
    TooLargeForBlock,
    #[error("segment is full")]
    SegmentFull,
    #[error("segment must be multiple of 4096")]
    SegmentSizeInvalid,
    #[error("no free space available")]
    NoFreeSpace,
    #[error("invalid header format")]
    InvalidHeaderFormat(String),
    #[error("metadata growth error")]
    MetadataGrowthError(MetadataGrowthError),
    #[error("fs error")]
    FsError(FsError),
}

#[derive(Error, Debug)]
pub enum MetadataGrowthError {
    #[error("no adjacent space available, can't find a free range after metadata")]
    NoAdjacentSpace,
    #[error("adjacent space isn't big enough")]
    InsufficientSpace,
    #[error("metadata too large")]
    MetadataTooLarge,
}

#[derive(Error, Debug)]
pub enum FsError {
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
}
