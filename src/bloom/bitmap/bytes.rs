// Copyright (c) Dom Dwyer <dom@itsallbroken.com>
// SPDX-License-Identifier: BSD-3-Clause



use std::convert::TryInto;

use bytes::{BufMut, Bytes, BytesMut};

use crate::bloom::{
    bitmap::{bitmask_for_key, index_for_key},
    Bitmap,
};

/// A plain, heap-allocated, `O(1)` indexed bitmap using `bytes::BytesMut` for
/// storage.
///
/// This type provide fast O(1) read and write operations, but trades O(n) space
/// complexity for the additional performance.
///
/// The [BytesBitmap] representation is suitable for persistence without the
/// need for serialisation; the output of [BytesBitmap::freeze()] can be used to
/// construct a new instance. [Serde] serialisation is also implemented as a
/// conveinence to enable serialisation to various formats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BytesBitmap {
    max_key: usize,
    bitmap: BytesMut,
}

impl BytesBitmap {
    pub fn freeze(self) -> Bytes {
        self.bitmap.freeze()
    }

    pub fn max_key(&self) -> usize {
        self.max_key
    }

    pub fn from_bytes(bitmap: impl Into<Bytes>) -> Self {
        let bitmap = bitmap.into();
        Self {
            max_key: bitmap.len() * 8,
            bitmap: BytesMut::from(bitmap),
        }
    }
}

impl Bitmap for BytesBitmap {
    fn new_with_capacity(max_key: usize) -> Self {
        let size = (index_for_key(max_key) + 1) * size_of::<usize>();
        let bytes = BytesMut::zeroed(size);

        Self {
            bitmap: bytes,
            max_key,
        }
    }

    fn set(&mut self, key: usize, value: bool) {
        let offset = index_for_key(key);
        let byte_offset = offset * size_of::<usize>();

        let slice = &mut self.bitmap[byte_offset..byte_offset + size_of::<usize>()];
        let mut num = usize::from_ne_bytes(slice.try_into().unwrap());

        if value {
            num |= bitmask_for_key(key);
        } else {
            num &= !bitmask_for_key(key);
        }

        slice.copy_from_slice(&num.to_ne_bytes());
    }

    fn get(&self, key: usize) -> bool {
        let offset = index_for_key(key);
        let byte_offset = offset * size_of::<usize>();
        let slice = &self.bitmap[byte_offset..byte_offset + size_of::<usize>()];
        let num = usize::from_ne_bytes(slice.try_into().unwrap());
        num & bitmask_for_key(key) != 0
    }

    fn byte_size(&self) -> usize {
        self.bitmap.len()
    }

    fn or(&self, other: &Self) -> Self {
        assert_eq!(self.bitmap.len(), other.bitmap.len());

        let mut result = BytesMut::with_capacity(self.bitmap.len());
        let chunks = self
            .bitmap
            .chunks_exact(size_of::<usize>())
            .zip(other.bitmap.chunks_exact(size_of::<usize>()));

        for (a_chunk, b_chunk) in chunks {
            let a = usize::from_ne_bytes(a_chunk.try_into().unwrap());
            let b = usize::from_ne_bytes(b_chunk.try_into().unwrap());
            result.put_slice(&(a | b).to_ne_bytes());
        }

        Self {
            bitmap: result,
            max_key: self.max_key,
        }
    }
}

