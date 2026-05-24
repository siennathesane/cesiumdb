// Copyright (c) Dom Dwyer <dom@itsallbroken.com>
// SPDX-License-Identifier: BSD-3-Clause

use crate::bloom::Bitmap;

use super::{bitmask_for_key, index_for_key};

/// A plain, heap-allocated, `O(1)` indexed bitmap.
///
/// This bitmap requires `O(n)` space and can be read and wrote to in `O(1)`
/// time.
///
/// This type is fast for both read and writes, but trades additional space for
/// the additional performance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VecBitmap {
    bitmap: Vec<usize>,
    max_key: usize,
}

impl VecBitmap {
    pub(crate) fn into_parts(self) -> (Vec<usize>, usize) {
        (self.bitmap, self.max_key)
    }
}

impl Bitmap for VecBitmap {
    fn set(&mut self, key: usize, value: bool) {
        let offset = index_for_key(key);

        if value {
            self.bitmap[offset] |= bitmask_for_key(key);
        } else {
            self.bitmap[offset] &= !bitmask_for_key(key);
        }
    }

    fn get(&self, key: usize) -> bool {
        let offset = index_for_key(key);

        self.bitmap[offset] & bitmask_for_key(key) != 0
    }

    fn byte_size(&self) -> usize {
        self.bitmap.len() * std::mem::size_of::<usize>()
    }

    fn or(&self, other: &Self) -> Self {
        // Invariant: the block maps are of equal length, meaning the zipped
        // iters yield both sides to completion.
        assert_eq!(self.bitmap.len(), other.bitmap.len());

        let bitmap = self
            .bitmap
            .iter()
            .zip(&other.bitmap)
            .map(|(a, b)| a | b)
            .collect();

        Self {
            bitmap,
            max_key: self.max_key,
        }
    }

    fn new_with_capacity(max_key: usize) -> Self {
        let bitmap = vec![0; index_for_key(max_key) + 1];
        Self { bitmap, max_key }
    }
}

