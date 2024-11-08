// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

use std::{
    mem::transmute,
    sync::atomic::{
        AtomicU32,
        AtomicU64,
        Ordering::{
            Acquire,
            Release,
        },
    },
    thread::ThreadId,
};

use parking_lot::Mutex;

/// The size of a cache line in bytes.
pub(crate) const CACHE_LINE: usize = 64;

/// Calculate the padding needed to align a type to a cache line
macro_rules! cache_line_padding {
    ($t:ty) => {{
        const BODY_SIZE: usize = size_of::<$t>();
        const PADDING: usize = (BODY_SIZE + CACHE_LINE - 1) & !(CACHE_LINE - 1);
        PADDING - BODY_SIZE
    }};
}

// compile-time verifications that things are cache aligned.
const _: () = assert!(align_of::<Reader>() >= CACHE_LINE);
const _: () = assert!(align_of::<MainSection>() >= CACHE_LINE);
const _: () = assert!(align_of::<WriterSection>() >= CACHE_LINE);

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

// TODO(@siennathesane): not sure we'll need this since we have ARCs soooo?
// we'll find out
#[repr(C)]
pub(crate) struct ReadBody {
    /// Current Transaction ID when this transaction began, or (txnid_t)-1.
    /// Multiple readers that start at the same time will probably have the same
    /// ID here. Again, it's not important to exclude them from  anything; all
    /// we need to know is which version of the DB they  started from so we can
    /// avoid overwriting any data used in that particular version.
    tx_id: AtomicU64,
    /// The process ID of the process owning this reader transaction
    pid: AtomicU64,
    /// The thread ID of the thread owning this reader transaction
    thread_id: AtomicU64,
}

/// A reader for a transaction.
#[repr(C, align(64))]
pub(crate) struct Reader {
    /// The reader data
    inner: ReadBody,

    /// Padding to ensure that the reader is the size of a cache line
    _pad: [u8; {
        const BODY_SIZE: usize = size_of::<ReadBody>();
        const PADDING: usize = (BODY_SIZE + CACHE_LINE - 1) & !(CACHE_LINE - 1);
        PADDING - BODY_SIZE
    }],
}

impl Reader {
    #[inline]
    pub fn tx_id(&self) -> u64 {
        self.inner.tx_id.load(Acquire)
    }

    #[inline]
    pub fn set_tx_id(&self, id: u64) {
        self.inner.tx_id.store(id, Release)
    }

    #[inline]
    pub fn pid(&self) -> u64 {
        self.inner.pid.load(Acquire)
    }

    #[inline]
    pub fn set_pid(&self, pid: u64) {
        self.inner.pid.store(pid, Release)
    }

    #[inline]
    pub fn thread_id(&self) -> u64 {
        self.inner.thread_id.load(Acquire)
    }

    #[inline]
    pub fn set_thread_id(&self, thread_id: u64) {
        self.inner.thread_id.store(thread_id, Release)
    }
}

impl Default for Reader {
    #[inline]
    fn default() -> Self {
        Self {
            inner: ReadBody {
                tx_id: AtomicU64::new(u64::MAX),
                pid: AtomicU64::new(std::process::id() as u64),
                thread_id: AtomicU64::new(
                    // just the init, we'll set it later
                    unsafe { transmute::<ThreadId, u64>(std::thread::current().id()) },
                ),
            },
            _pad: [0u8; {
                const BODY_SIZE: usize = size_of::<ReadBody>();
                const PADDING: usize = (BODY_SIZE + CACHE_LINE - 1) & !(CACHE_LINE - 1);
                PADDING - BODY_SIZE
            }],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_cache_alignment() {
        assert_eq!(
            align_of::<Reader>(),
            CACHE_LINE,
            "Reader is not cache aligned"
        );
        assert_eq!(
            size_of::<Reader>() % CACHE_LINE,
            0,
            "Reader is not a multiple of a cache line"
        );
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
