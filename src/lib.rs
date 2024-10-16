#![allow(dead_code)]
#![allow(unused_variables)]

mod config;
mod db;
mod level;
pub(crate) mod segment;
mod tombstone;

#[cfg(not(unix))]
compile_error!("only linux & macos are unsupported");
