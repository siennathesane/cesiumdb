// Copyright (c) Dom Dwyer <dom@itsallbroken.com>
// SPDX-License-Identifier: BSD-3-Clause

//! Bitmap implementations for the backing storage of a
//! [`Bloom2`](crate::bloom::Bloom2).

mod bytes;
mod compressed_bitmap;

pub use bytes::*;
pub use compressed_bitmap::*;

#[inline(always)]
pub(crate) fn bitmask_for_key(key: usize) -> usize {
    1 << (key % (u64::BITS as usize))
}

#[inline(always)]
pub(crate) fn index_for_key(key: usize) -> usize {
    key / (u64::BITS as usize)
}
