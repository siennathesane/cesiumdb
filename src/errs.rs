use std::io;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CesiumError {
    #[error("os i/o error")]
    IoError(io::Error),
    #[error("data insertion would exceed maximum capacity")]
    DataExceedsMaximum,
    #[error("memtable is frozen")]
    MemtableIsFrozen
}
