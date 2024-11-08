// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

use std::{
    intrinsics::transmute,
    sync::atomic::{
        AtomicU64,
        Ordering::{
            Acquire,
            Release,
        },
    },
    thread::ThreadId,
};

use crate::{
    cache_line_padding,
    CACHE_LINE,
};

// compile-time verifications that things are cache aligned.
const _: () = assert!(align_of::<Reader>() >= CACHE_LINE);

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
    _pad: [u8; cache_line_padding!(ReadBody)],
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
    use crate::CACHE_LINE;

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
    }
}
