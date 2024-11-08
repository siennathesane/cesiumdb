// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

use std::{
    ptr::NonNull,
    sync::atomic::{
        AtomicU32,
        AtomicU64,
        Ordering::{
            Acquire,
            Relaxed,
            Release,
        },
    },
};
use std::sync::Arc;
use bitflags::bitflags;
use parking_lot::Mutex;
use crate::{cache_line_padding, CACHE_LINE};
use crate::disk::{
    page::{
        LoosePage,
        Page,
    },
    reader::Reader,
};

// compile-time verifications that things are cache aligned.
const _: () = assert!(align_of::<MainSection>() >= CACHE_LINE);
const _: () = assert!(align_of::<WriterSection>() >= CACHE_LINE);

pub(crate) struct Transaction {
    /// Parent transaction if this is a nested transaction
    parent: Option<Arc<Transaction>>,
    /// Child transaction if one exists
    child: Option<Arc<Transaction>>,
    /// Next unallocated page number
    next_page_num: u64,
    txn_id: u64,
    // env: Arc<Environment>,
    free_pages: Vec<NonNull<Page>>, // TODO(@siennathesane): maybe not right
    inner: TransactionInner,
    spill_pages: Vec<u64>,
}

bitflags! {
    pub(crate) struct TransactionFlags: u32 {
        const READ_ONLY = 0x01;
        const WRITE_MAP = 0x02;
        const FINISHED = 0x04;
        const ERROR = 0x08;
        const DIRTY = 0x10;
        const SPILLS = 0x20;
        const HAS_CHILD = 0x40;
        
        const BLOCKED = Self::FINISHED.bits() | Self::ERROR.bits() | Self::HAS_CHILD.bits();
    }
}

bitflags! {
    pub (crate) struct DbFlags: u8 {
        const DIRTY = 0x01;
        const STALE = 0x02;
        const NEW = 0x04;
        const VALID = 0x08;
        const USR_VALID = 0x10;
        const DUP_DATA = 0x20;
    }
}

impl TransactionBody {
    #[inline]
    pub(crate) fn magic(&self) -> u32 {
        self.magic
    }

    #[inline]
    pub(crate) fn set_magic(&mut self, magic: u32) {
        self.magic = magic
    }

    #[inline]
    pub(crate) fn format(&self) -> u32 {
        self.format
    }

    #[inline]
    pub(crate) fn set_format(&mut self, format: u32) {
        self.format = format
    }

    #[inline]
    pub(crate) fn txn_id(&self) -> u64 {
        self.txn_id.load(Acquire)
    }

    #[inline]
    pub(crate) fn set_txn_id(&self, id: u64) {
        self.txn_id.store(id, Release)
    }

    #[inline]
    pub(crate) fn num_readers(&self) -> u32 {
        self.num_readers.load(Acquire)
    }

    #[inline]
    pub(crate) fn set_num_readers(&self, num_readers: u32) {
        self.num_readers.store(num_readers, Release)
    }
}

/// Cache line aligned transaction info section containing the transaction body
#[repr(C, align(64))]
pub(crate) struct MainSection {
    /// The actual transaction body data
    pub(crate) body: TransactionBody,

    /// Padding to ensure cache line alignment
    _pad: [u8; cache_line_padding!(TransactionBody)],
}

/// Cache line aligned section containing the writer mutex
#[repr(C, align(64))]
pub(crate) struct WriterSection {
    /// Writer mutex for protecting write transactions
    pub(crate) writer_mutex: Mutex<()>,

    /// Padding to ensure cache line alignment
    _pad: [u8; cache_line_padding!(Mutex<()>)],
}

/// The complete reader table structure
#[repr(C)]
pub(crate) struct TransactionInfo {
    /// Main transaction information section
    pub(crate) main: MainSection,

    /// Writer mutex section
    pub(crate) writer: WriterSection,

    /// Array of readers - conceptually variable length
    pub(crate) readers: Vec<Reader>,
}

impl TransactionInfo {
    /// Get a slice of the readers array
    #[inline]
    pub unsafe fn readers(&self) -> &[Reader] {
        self.readers.as_slice()
    }

    /// Get a mutable slice of the readers array
    #[inline]
    pub unsafe fn readers_slice_mut(&mut self) -> &mut [Reader] {
        self.readers.as_mut_slice()
    }

    /// Get the transaction body magic number
    #[inline]
    pub(crate) fn magic(&self) -> u32 {
        self.main.body.magic()
    }

    /// Get the transaction body format
    #[inline]
    pub(crate) fn format(&self) -> u32 {
        self.main.body.format()
    }

    /// Get the transaction ID
    #[inline]
    pub(crate) fn txn_id(&self) -> u64 {
        self.main.body.txn_id()
    }

    /// Set the transaction ID
    #[inline]
    pub(crate) fn set_txn_id(&self, id: u64) {
        self.main.body.set_txn_id(id)
    }

    /// Get the number of readers
    #[inline]
    pub(crate) fn num_readers(&self) -> u32 {
        self.main.body.num_readers()
    }

    /// Set the number of readers
    #[inline]
    pub(crate) fn add_readers(&self, num: u32) {
        self.main.body.set_num_readers(num)
    }
}

/// The body of a transaction.
#[repr(C, align(64))]
pub(crate) struct TransactionBody {
    // encryption would break that guarantee.
    /// Magic number identifying this as an LMDB file.
    magic: u32,

    /// Format version of this lock file
    format: u32,

    /// Mutex protecting access to the reader table.
    /// This is the reader table lock.
    reader_mutex: Mutex<()>,

    /// ID of the last transaction committed to the database.
    /// This is stored here for convenience but can always be
    /// determined by reading the main database meta pages.
    txn_id: AtomicU64,

    /// Number of slots that have been used in the reader table.
    /// This tracks the maximum count and is never decremented
    /// when readers release their slots.
    num_readers: AtomicU32,
}

impl Default for TransactionBody {
    #[inline]
    fn default() -> Self {
        Self {
            magic: 0,
            format: 0,
            reader_mutex: Mutex::new(()),
            txn_id: AtomicU64::new(0),
            num_readers: AtomicU32::new(0),
        }
    }
}

pub(crate) struct TransactionInner {
    loose_head: Option<NonNull<Page>>,
    loose_count: AtomicU64,
}

impl TransactionInner {
    #[inline(always)]
    pub(crate) fn push_loose_page(&mut self, page: NonNull<Page>) {
        let mut loose = LoosePage::new(page);
        loose.set_next(self.loose_head.map(|mut ptr| unsafe { ptr.as_mut() }));
        self.loose_head = Some(loose.page());
        self.loose_count.fetch_add(1, Acquire);
    }

    #[inline(always)]
    pub(crate) fn pop_loose_page(&mut self) -> Option<NonNull<Page>> {
        if let Some(mut ptr) = self.loose_head {
            unsafe {
                let page = ptr.as_mut();
                let loose = LoosePage::new(page);
                self.loose_head = loose.next();
                self.loose_count.fetch_sub(1, Acquire);
                Some(page)
            }
        } else {
            None
        }
    }

    #[inline(always)]
    pub(crate) fn loose_count(&self) -> u64 {
        self.loose_count.load(Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_cache_alignment() {
        assert_eq!(
            align_of::<MainSection>(),
            CACHE_LINE,
            "MainSection is not cache aligned"
        );
        assert_eq!(
            size_of::<MainSection>() % CACHE_LINE,
            0,
            "MainSection is not a multiple of a cache line"
        );
        assert_eq!(
            align_of::<WriterSection>(),
            CACHE_LINE,
            "WriterSection is not cache aligned"
        );
        assert_eq!(
            size_of::<WriterSection>() % CACHE_LINE,
            0,
            "WriterSection is not a multiple of a cache line"
        );
    }
}
