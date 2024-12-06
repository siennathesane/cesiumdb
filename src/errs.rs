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
}
