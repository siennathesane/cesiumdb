use std::{
    fmt::Display,
    mem,
    sync::{
        atomic::{
            AtomicU64,
            Ordering::Relaxed,
        },
        Arc,
    },
};

use bytes::Bytes;
use parking_lot::RwLock;

use crate::{
    block::{
        Block,
        EntryFlag,
        EntryFlag::{
            Complete,
            End,
            Middle,
            Start,
        },
        BLOCK_SIZE,
        MAX_ENTRY_SIZE,
    },
    errs::{
        BlockError,
        SegmentError,
    },
    fs::Fs,
    index::SegmentIndex,
    keypair::{
        KeyBytes,
        ValueBytes,
    },
    segment::BlockType::{
        Key,
        Value,
    },
    segment_writer::SegmentWriter,
};

#[derive(Debug)]
pub enum BlockType {
    Key,
    Value,
}

impl Display for BlockType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            | Key => write!(f, "key"),
            | Value => write!(f, "value"),
        }
    }
}

pub(crate) struct Segment {
    // keys
    key_writer: SegmentWriter,
    key_block_count: AtomicU64,
    key_index: SegmentIndex,
    current_key_block: Block,

    // values
    val_writer: SegmentWriter,
    val_block_count: AtomicU64,
    current_val_block: Block,
    val_index: SegmentIndex,

    // shared
    current_ns: AtomicU64,
}

impl Segment {
    pub(crate) fn new(
        key_id: u64,
        val_id: u64,
        seed: i64,
        key_writer: SegmentWriter,
        val_writer: SegmentWriter,
    ) -> Self {
        Self {
            key_writer,
            key_block_count: AtomicU64::new(0),
            val_writer,
            val_block_count: AtomicU64::new(0),
            key_index: SegmentIndex::new(key_id, seed),
            current_key_block: Block::new(),
            current_val_block: Block::new(),
            val_index: SegmentIndex::new(val_id, seed),
            current_ns: AtomicU64::new(0),
        }
    }

    pub(crate) fn write(&mut self, key: &[u8], val: &[u8]) -> Result<(), SegmentError> {
        // set the namespace
        let ns = u64::from_le_bytes(key[0..8].as_ref().try_into().unwrap());
        if ns != self.current_ns.load(Relaxed) {
            self.current_ns.store(ns, Relaxed);
            self.key_index.add_ns_offset(ns);
            self.val_index.add_ns_offset(ns);
        }

        // NB(@siennathesane): we are tightly packing the keys and values into blocks
        // and sometimes the payload will span multiple blocks. this sometimes means
        // the underlying "file" (re: FRange) will be fragmented. this isn't
        // inherently a problem because when a full compaction is run, the
        // filesystem will be defragmented and the data will be contiguous.

        match self.current_key_block.add_entry(key, Complete) {
            | Ok(()) => {},
            | Err(be) => match be {
                | entry_too_large => {
                    match self.split_across_blocks(key, &Key) {
                        | Ok(_) => {},
                        | Err(e) => return Err(e),
                    };
                },
                | block_full => {
                    self.write_block(&Key);
                    match self.current_key_block.add_entry(key, Complete) {
                        | Ok(_) => {},
                        | Err(rbe) => match rbe {
                            | too_large_for_block => {
                                match self.split_across_blocks(key, &Key) {
                                    | Ok(_) => {},
                                    | Err(e) => return Err(e),
                                };
                            },
                            | _ => {
                                unreachable!("unexpected key block error, no idea how we got here")
                            },
                        },
                    };
                },
            },
        };
        self.key_index.add_item(key);

        match self.current_val_block.add_entry(key, Complete) {
            | Ok(()) => {},
            | Err(be) => match be {
                | entry_too_large => {
                    match self.split_across_blocks(val, &Value) {
                        | Ok(_) => {},
                        | Err(e) => return Err(e),
                    };
                },
                | block_full => {
                    self.write_block(&Value);
                    match self.current_key_block.add_entry(key, Complete) {
                        | Ok(_) => {},
                        | Err(rbe) => match rbe {
                            | too_large_for_block => {
                                match self.split_across_blocks(val, &Value) {
                                    | Ok(_) => {},
                                    | Err(e) => return Err(e),
                                };
                            },
                            | _ => {
                                unreachable!("unexpected val block error, no idea how we got here")
                            },
                        },
                    };
                },
            },
        };
        self.val_index.add_item(key);

        Ok(())
    }

    /// Split a payload across multiple blocks.
    fn split_across_blocks(&mut self, data: &[u8], r#type: &BlockType) -> Result<(), SegmentError> {
        let mut remaining = data;

        // Write start block
        let available = self.current_key_block.remaining_space() - 1; // -1 for flag
        if !remaining.is_empty() {
            match &r#type {
                | key => {
                    match self
                        .current_key_block
                        .add_entry(&remaining[..available], Start)
                    {
                        | Ok(_) => {},
                        | Err(_) => {
                            unreachable!("key block is properly sized, this should never happen")
                        },
                    };
                    self.key_index.add_block(data);
                    match self.write_block(key) {
                        | Ok(_) => {},
                        | Err(e) => return Err(e),
                    };
                },
                | value => {
                    match self
                        .current_val_block
                        .add_entry(&remaining[..available], Start)
                    {
                        | Ok(_) => {},
                        | Err(_) => {
                            unreachable!("val block is properly sized, this should never happen")
                        },
                    };
                    match self.write_block(value) {
                        | Ok(_) => {},
                        | Err(e) => return Err(e),
                    };
                },
            };
            remaining = &remaining[available..];
        }

        // write middle blocks
        while remaining.len() > MAX_ENTRY_SIZE {
            match r#type {
                | key => {
                    let mut block = mem::replace(&mut self.current_key_block, Block::new());
                    match block.add_entry(&remaining[..MAX_ENTRY_SIZE], Middle) {
                        | Ok(_) => {},
                        | Err(_) => {
                            unreachable!("middle key block is properly sized, this should never happen")
                        },
                    };
                    match self.write_block(key) {
                        | Ok(_) => {},
                        | Err(e) => return Err(e),
                    };
                },
                | value => {
                    let mut block = mem::replace(&mut self.current_key_block, Block::new());
                    match block.add_entry(&remaining[..MAX_ENTRY_SIZE], Middle) {
                        | Ok(_) => {},
                        | Err(_) => {
                            unreachable!("middle val block is properly sized, this should never happen")
                        },
                    };
                    match self.write_block(value) {
                        | Ok(_) => {},
                        | Err(e) => return Err(e),
                    };
                },
            };
            remaining = &remaining[MAX_ENTRY_SIZE..];
        }

        // Write end block
        match r#type {
            | key => {
                match self.current_key_block.add_entry(remaining, End) {
                    | Ok(_) => {},
                    | Err(_) => {
                        unreachable!("last key block is properly sized, this should never happen")
                    },
                };
            },
            | value => {
                match self.current_val_block.add_entry(remaining, End) {
                    | Ok(_) => {},
                    | Err(_) => {
                        unreachable!("last val block is properly sized, this should never happen")
                    },
                };
            },
        }

        Ok(())
    }

    fn write_block(&mut self, r#type: &BlockType) -> Result<(), SegmentError> {
        // swap the blocks to prepare to write it to disk
        let block = match &r#type {
            | key => mem::replace(&mut self.current_key_block, Block::new()),
            | value => mem::replace(&mut self.current_val_block, Block::new()),
        };

        // add the starting key to the index. this will never be `None` because the
        // block is always full but the API needs to be an Option
        match r#type {
            | key => {
                match self.current_key_block.get(0) {
                    | None => {},
                    | Some(.., v) => {
                        self.key_index.add_block(v.1);
                    },
                };
            },
            | value => {
                match self.current_val_block.get(0) {
                    | None => {},
                    | Some(.., v) => {
                        self.val_index.add_block(v.1);
                    },
                };
            },
        };

        // send it to the queue to be written to disk
        match r#type {
            | key => {
                match self.key_writer.write_block(block) {
                    | Ok(()) => {
                        self.key_block_count.fetch_add(1, Relaxed);
                    },
                    | Err(e) => {
                        return Err(e);
                    },
                };
                self.key_block_count.fetch_add(1, Relaxed);
                Ok(())
            },
            | value => {
                match self.val_writer.write_block(block) {
                    | Ok(()) => {
                        self.val_block_count.fetch_add(1, Relaxed);
                    },
                    | Err(e) => return Err(e),
                };
                self.val_block_count.fetch_add(1, Relaxed);
                Ok(())
            },
        }
    }
}
