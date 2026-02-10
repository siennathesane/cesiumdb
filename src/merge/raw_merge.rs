// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

//! Raw merge iterator for zero-copy compaction.
//!
//! Specialized merge iterator that operates on `RawEntry` (raw serialized
//! bytes) instead of `(KeyBytes, ValueBytes)`, eliminating deserialization
//! overhead.

use std::{
    cmp::{
        Ordering,
        Reverse,
    },
    collections::BinaryHeap,
};

use crate::{
    errs::SegmentError,
    raw_entry::RawEntry,
};

/// A merge iterator that operates on raw serialized entries.
///
/// Uses a min-heap to merge multiple sorted iterators of `RawEntry`,
/// maintaining the same ordering semantics as `MergeIterator` but
/// without deserializing keys into `KeyBytes`.
pub(crate) struct RawMergeIterator<I>
where
    I: Iterator<Item = Result<RawEntry, SegmentError>>, {
    iters: BinaryHeap<RawHeapItem<I>>,
}

impl<I> RawMergeIterator<I>
where
    I: Iterator<Item = Result<RawEntry, SegmentError>>,
{
    pub fn new(mut iters: Vec<I>) -> Self {
        iters.sort_by_key(|iter| Reverse(iter.size_hint().0));

        let heap = iters
            .into_iter()
            .enumerate()
            .map(|(idx, mut iter)| {
                let peeked = iter.next();
                RawHeapItem {
                    iter,
                    peeked,
                    index: idx as u64,
                }
            })
            .collect();
        Self { iters: heap }
    }
}

impl<I> Iterator for RawMergeIterator<I>
where
    I: Iterator<Item = Result<RawEntry, SegmentError>>,
{
    type Item = Result<RawEntry, SegmentError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut smallest = match self.iters.pop() {
                | None => return None,
                | Some(v) => v,
            };

            // Take the peeked value
            let current_item = smallest.peeked.take();

            // Peek the next value
            smallest.peeked = smallest.iter.next();
            let has_more = smallest.peeked.is_some();

            match current_item {
                | Some(Ok(entry)) => {
                    // Skip pointer entries (ts == 0) unless it's the last item
                    if entry.ts() == 0 {
                        if has_more {
                            self.iters.push(smallest);
                        }
                        continue;
                    }

                    if has_more {
                        self.iters.push(smallest);
                    }
                    return Some(Ok(entry));
                },
                | Some(Err(e)) => {
                    // Propagate errors but keep iterating other sources
                    if has_more {
                        self.iters.push(smallest);
                    }
                    return Some(Err(e));
                },
                | None => {
                    // Don't push back empty iterators
                    continue;
                },
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let mut size = 0;
        for item in self.iters.iter() {
            size += item.iter.size_hint().0;
            if item.peeked.is_some() {
                size += 1;
            }
        }
        (size, None)
    }
}

struct RawHeapItem<I>
where
    I: Iterator<Item = Result<RawEntry, SegmentError>>, {
    iter: I,
    peeked: Option<Result<RawEntry, SegmentError>>,
    index: u64,
}

impl<I> PartialEq for RawHeapItem<I>
where
    I: Iterator<Item = Result<RawEntry, SegmentError>>,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<I> Eq for RawHeapItem<I> where I: Iterator<Item = Result<RawEntry, SegmentError>> {}

impl<I> PartialOrd for RawHeapItem<I>
where
    I: Iterator<Item = Result<RawEntry, SegmentError>>,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<I> Ord for RawHeapItem<I>
where
    I: Iterator<Item = Result<RawEntry, SegmentError>>,
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match (&self.peeked, &other.peeked) {
            | (Some(Ok(left)), Some(Ok(right))) => {
                // Reversed for min-heap behavior (BinaryHeap is a max-heap)
                right.cmp_key(left)
            },
            | (Some(Ok(_)), Some(Err(_))) => Ordering::Less, // Ok < Err (prefer Ok)
            | (Some(Err(_)), Some(Ok(_))) => Ordering::Greater,
            | (Some(_), None) => Ordering::Less,
            | (None, Some(_)) => Ordering::Greater,
            | (Some(Err(_)), Some(Err(_))) => other.index.cmp(&self.index),
            | (None, None) => other.index.cmp(&self.index),
        }
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::*;
    use crate::{
        keypair::{
            DEFAULT_NS,
            KeyBytes,
            ValueBytes,
        },
        utils::Serializer,
    };

    fn make_raw(ns: u64, key: &str, ts: u128, value: &str) -> RawEntry {
        let k = KeyBytes::new(ns, Bytes::from(key.to_string()), ts);
        let v = ValueBytes::new(ns, Bytes::from(value.to_string()));
        RawEntry::new(k.serialize(), v.serialize())
    }

    #[test]
    fn test_raw_merge_empty() {
        let empty_vec: Vec<Vec<Result<RawEntry, SegmentError>>> = vec![];
        let iters = empty_vec.into_iter().map(IntoIterator::into_iter).collect();
        let mut merge_iter = RawMergeIterator::new(iters);
        assert!(merge_iter.next().is_none());
    }

    #[test]
    fn test_raw_merge_single_iterator() {
        let data: Vec<Result<RawEntry, SegmentError>> = vec![
            Ok(make_raw(DEFAULT_NS, "key1", 1, "value1")),
            Ok(make_raw(DEFAULT_NS, "key2", 2, "value2")),
        ];
        let iters = vec![data.into_iter()];
        let mut merge_iter = RawMergeIterator::new(iters);

        let first = merge_iter.next().unwrap().unwrap();
        assert_eq!(first.user_key(), b"key1");
        let second = merge_iter.next().unwrap().unwrap();
        assert_eq!(second.user_key(), b"key2");
        assert!(merge_iter.next().is_none());
    }

    #[test]
    fn test_raw_merge_version_ordering() {
        let data1: Vec<Result<RawEntry, SegmentError>> =
            vec![Ok(make_raw(DEFAULT_NS, "key1", 3, "v3"))];
        let data2: Vec<Result<RawEntry, SegmentError>> =
            vec![Ok(make_raw(DEFAULT_NS, "key1", 2, "v2"))];
        let data3: Vec<Result<RawEntry, SegmentError>> =
            vec![Ok(make_raw(DEFAULT_NS, "key1", 1, "v1"))];

        let iters = vec![data1.into_iter(), data2.into_iter(), data3.into_iter()];
        let merge_iter = RawMergeIterator::new(iters);

        let results: Vec<_> = merge_iter.filter_map(|r| r.ok()).collect();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].ts(), 3);
        assert_eq!(results[1].ts(), 2);
        assert_eq!(results[2].ts(), 1);
    }

    #[test]
    fn test_raw_merge_namespace_ordering() {
        let data1: Vec<Result<RawEntry, SegmentError>> = vec![Ok(make_raw(2, "key1", 1, "ns2_v"))];
        let data2: Vec<Result<RawEntry, SegmentError>> = vec![Ok(make_raw(1, "key1", 1, "ns1_v"))];

        let iters = vec![data1.into_iter(), data2.into_iter()];
        let merge_iter = RawMergeIterator::new(iters);

        let results: Vec<_> = merge_iter.filter_map(|r| r.ok()).collect();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].ns(), 1);
        assert_eq!(results[1].ns(), 2);
    }

    #[test]
    fn test_raw_merge_error_propagation() {
        let data: Vec<Result<RawEntry, SegmentError>> = vec![
            Ok(make_raw(DEFAULT_NS, "key1", 1, "v1")),
            Err(SegmentError::CorruptedBlock),
        ];
        let iters = vec![data.into_iter()];
        let merge_iter = RawMergeIterator::new(iters);

        let results: Vec<_> = merge_iter.collect();
        assert_eq!(results.len(), 2);
        assert!(results[0].is_ok());
        assert!(results[1].is_err());
    }

    #[test]
    fn test_raw_merge_size_hint() {
        let data1: Vec<Result<RawEntry, SegmentError>> = vec![
            Ok(make_raw(DEFAULT_NS, "key1", 1, "v1")),
            Ok(make_raw(DEFAULT_NS, "key2", 1, "v2")),
        ];
        let data2: Vec<Result<RawEntry, SegmentError>> = vec![
            Ok(make_raw(DEFAULT_NS, "key3", 1, "v3")),
            Ok(make_raw(DEFAULT_NS, "key4", 1, "v4")),
        ];

        let iters = vec![data1.into_iter(), data2.into_iter()];
        let merge_iter = RawMergeIterator::new(iters);

        let (lower, upper) = merge_iter.size_hint();
        assert_eq!(lower, 4);
        assert_eq!(upper, None);
    }
}
