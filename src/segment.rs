use std::{
    fmt::Display,
    mem,
    sync::{
        Arc,
        atomic::{
            AtomicU64,
            Ordering::Relaxed,
        },
    },
};

use crate::{
    block::{
        Block,
        EntryFlag::{
            Complete,
            End,
            Middle,
            Start,
        },
        MAX_ENTRY_SIZE,
    },
    errs::SegmentError,
    index::Index,
    segment::BlockType::{
        Key,
        Value,
    },
    segment_reader::SegmentReader,
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
    key_index: Index,
    current_key_block: Block,

    // values
    val_writer: SegmentWriter,
    val_block_count: AtomicU64,
    current_val_block: Block,
    val_index: Index,

    // shared
    current_ns: AtomicU64,

    // readers
    reader: Arc<SegmentReader>,
}

impl Segment {
    pub(crate) fn new(
        key_id: u64,
        val_id: u64,
        seed: i64,
        key_writer: SegmentWriter,
        val_writer: SegmentWriter,
        reader: SegmentReader,
    ) -> Self {
        Self {
            key_writer,
            key_block_count: AtomicU64::new(0),
            val_writer,
            val_block_count: AtomicU64::new(0),
            key_index: Index::new(key_id, seed),
            current_key_block: Block::new(),
            current_val_block: Block::new(),
            val_index: Index::new(val_id, seed),
            current_ns: AtomicU64::new(0),
            reader: Arc::new(reader),
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

    pub(crate) fn new_reader(&self) -> Arc<SegmentReader> {
        self.reader.clone()
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
                            unreachable!(
                                "middle key block is properly sized, this should never happen"
                            )
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
                            unreachable!(
                                "middle val block is properly sized, this should never happen"
                            )
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
