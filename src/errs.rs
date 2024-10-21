use std::io;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("general i/o error")]
    Io(IoError),
}

#[derive(Error, Debug)]
pub enum IoError {
    #[error("os i/o error")]
    Error(io::Error),
}
