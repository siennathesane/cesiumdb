#![allow(dead_code)]
#![allow(unused_variables)]

mod sstable;
mod tombstone;

// TODO(@siennathesane): this in theory shouldn't be a requirement but it will
// take time to figure it out
#[cfg(not(target_pointer_width = "64"))]
compile_error!("this crate will not work on 32-bit targets");

#[cfg(not(unix))]
compile_error!("only linux is unsupported");
