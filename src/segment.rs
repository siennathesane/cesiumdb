use std::{
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
        EntryFlag::{
            Middle,
            Start,
        },
        BLOCK_SIZE,
    },
    errs::{
        CesiumError,
        CesiumError::IoError,
    },
    fs::Fs,
    index::SegmentIndex,
    keypair::{
        KeyBytes,
        ValueBytes,
    },
    segment_writer::SegmentWriter,
};
use crate::block::EntryFlag;
use crate::block::EntryFlag::{Complete, End};

pub enum BlockType {
    Key,
    Value,
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
        id: u64,
        seed: i64,
        key_writer: SegmentWriter,
        val_writer: SegmentWriter,
    ) -> Self {
        Self {
            key_writer,
            key_block_count: AtomicU64::new(0),
            val_writer,
            val_block_count: AtomicU64::new(0),
            key_index: SegmentIndex::new(id, seed),
            current_key_block: Block::new(),
            current_val_block: Block::new(),
            val_index: SegmentIndex::new(id, seed),
            current_ns: AtomicU64::new(0),
        }
    }

    pub(crate) fn write(&mut self, key: &[u8], val: &[u8]) -> Result<(), CesiumError> {
        
        // set the namespace
        let ns = u64::from_le_bytes(key[0..8].as_ref().try_into().unwrap());
        if ns != self.current_ns.load(Relaxed) {
            self.current_ns.store(ns, Relaxed);
            self.key_index.add_ns_offset(ns);
            self.val_index.add_ns_offset(ns);
        }
        
        // NB(@siennathesane): we are tightly packing the keys and values into blocks
        //    and sometimes the payload will span multiple blocks. this sometimes means
        //    the underlying "file" (re: FRange) will be fragmented. this isn't inherently
        //    a problem because when a full compaction is run, the filesystem will be
        //    defragmented and the data will be contiguous.
        
        if self.current_key_block.will_fit(key.len()) {
            self.current_key_block.add_entry(&key, Complete)?;
            self.key_index.add_item(&key);
        } else {
            self.key_index.add_item(&key);
            self.split_across_blocks(&key, &BlockType::Key)?;
        }
        
        if self.current_val_block.will_fit(val.len()) {
            self.current_val_block.add_entry(&val, Complete)?;
            self.val_index.add_item(&val);
        } else {
            self.val_index.add_item(&val);
            self.split_across_blocks(&val, &BlockType::Value)?;
        }

        Ok(())
    }

    fn split_across_blocks(&mut self, data: &[u8], r#type: &BlockType) -> Result<(), CesiumError> {
        let mut remaining = data;

        // Write start block
        let available = self.current_key_block.remaining_space() - 1; // -1 for flag
        if !remaining.is_empty() {
            match &r#type {
                | Key => {
                    self.current_key_block
                        .add_entry(&remaining[..available], Start)?;
                    self.key_index.add_block(&data);
                    self.write_block(Key)?;
                },
                | Value => {
                    self.current_val_block
                        .add_entry(&remaining[..available], Start)?;
                    self.write_block(Value)?;
                },
            };
            remaining = &remaining[available..];
        }

        // Write middle blocks
        while remaining.len() > BLOCK_SIZE - 1 {
            match r#type {
                | Key => {
                    let mut block = mem::replace(&mut self.current_key_block, Block::new());
                    block.add_entry(&remaining[..BLOCK_SIZE - 1], Middle)?;
                    self.write_block(Key)?;
                    remaining = &remaining[BLOCK_SIZE - 1..];
                },
                | Value => {
                    let mut block = mem::replace(&mut self.current_key_block, Block::new());
                    block.add_entry(&remaining[..BLOCK_SIZE - 1], Middle)?;
                    self.write_block(Value)?;
                },
            };
            remaining = &remaining[BLOCK_SIZE - 1..];
        }

        // Write end block
        if !remaining.is_empty() {
            match r#type {
                | Key => {
                    self.current_key_block
                        .add_entry(remaining, End)?;
                },
                | Value => {
                    self.current_val_block
                        .add_entry(remaining, End)?;
                },
            }
        };
        
        Ok(())
    }
    
    fn write_block(&mut self, r#type: &BlockType) -> Result<(), CesiumError> {
        // swap the blocks to prepare to write it to disk
        let block = match &r#type {
            | Key => mem::replace(&mut self.current_key_block, Block::new()),
            | Value => mem::replace(&mut self.current_val_block, Block::new()),
        };
        
        // add the starting key to disk. this will never be `None` because the block
        // is always full but the API needs to be an Option
        let mut index = match r#type {
            | Key => {
                let val =match self.current_key_block.get(0) {
                    None => {}
                    Some(.., v) => {
                        self.key_index.add_block(v.1);
                    },
                };
            },
            | Value => {
                let val =match self.current_val_block.get(0) {
                    None => {}
                    Some(.., v) => {
                        self.val_index.add_block(v.1);
                    },
                };
            },
        };
        
        match r#type {
            | Key => {
                match self.key_writer.write_block(block) {
                    Ok(()) => {
                        self.key_block_count.fetch_add(1, Relaxed);
                    },
                    Err(e) => {
                        return Err(e);
                    },
                };
                self.key_block_count.fetch_add(1, Relaxed);
                Ok(())
            },
            | Value => {
                match self.val_writer.write_block(block) {
                    Ok(()) => {
                        self.val_block_count.fetch_add(1, Relaxed);
                    }
                    Err(e) => return Err(e),
                };
                self.val_block_count.fetch_add(1, Relaxed);
                Ok(())
            },
        }
    }
}
