use std::{
    cell::RefCell,
    cmp::Ordering::{
        self,
        Equal,
    },
    collections::BinaryHeap,
};

use crate::{
    keypair::{
        KeyBytes,
        ValueBytes,
    },
    peek::Peekable,
};

/// MergeIterator is a merge iterator that merges multiple iterators into one.
/// It maintains the ordering of keys across all iterators, preserves stable
/// ordering for equal keys, handles exhausted iterators, and uses a min-heap
/// for efficient next-smallest-key lookup.
pub struct MergeIterator<I>
where
    I: Iterator<Item = (KeyBytes, ValueBytes)>, {
    iters: BinaryHeap<HeapItem<I>>,
}

impl<I> MergeIterator<I>
where
    I: Iterator<Item = (KeyBytes, ValueBytes)>,
{
    pub fn new(iters: Vec<I>) -> Self {
        let heap = iters
            .into_iter()
            .enumerate()
            .map(|(idx, iter)| HeapItem {
                iter: RefCell::new(Peekable::new(iter)),
                index: idx as u64,
            })
            .collect();
        Self { iters: heap }
    }
}

impl<I> Iterator for MergeIterator<I>
where
    I: Iterator<Item = (KeyBytes, ValueBytes)>,
{
    type Item = (KeyBytes, ValueBytes);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let smallest = self.iters.pop()?;

            // Get next item and check if we have more items
            let has_more = {
                let mut iter = smallest.iter.borrow_mut();
                iter.peek().is_some()
            };

            let next_item = smallest.iter.borrow_mut().next();

            // Decide whether to push back the iterator based on conditions
            match next_item {
                | Some((key, value)) => {
                    // Skip pointer entries (ts == 0) unless it's the last item
                    if key.ts() == 0 {
                        if has_more {
                            self.iters.push(smallest);
                        }
                        continue;
                    }

                    // For regular entries, always push back if there are more items
                    if has_more {
                        self.iters.push(smallest);
                    }
                    return Some((key, value));
                },
                | None => {
                    // Don't push back empty iterators
                    continue;
                },
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let mut size = 0;
        for iter in self.iters.iter() {
            size += iter.iter.borrow().size_hint().0;
        }
        (size, None)
    }
}

pub(crate) struct HeapItem<I>
where
    I: Iterator<Item = (KeyBytes, ValueBytes)>, {
    iter: RefCell<Peekable<I>>,
    index: u64,
}

impl<I> PartialEq for HeapItem<I>
where
    I: Iterator<Item = (KeyBytes, ValueBytes)>,
{
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<I> Eq for HeapItem<I> where I: Iterator<Item = (KeyBytes, ValueBytes)> {}

impl<I> PartialOrd for HeapItem<I>
where
    I: Iterator<Item = (KeyBytes, ValueBytes)>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<I> Ord for HeapItem<I>
where
    I: Iterator<Item = (KeyBytes, ValueBytes)>,
{
    fn cmp(&self, other: &Self) -> Ordering {
        match (
            self.iter.borrow_mut().peek(),
            other.iter.borrow_mut().peek(),
        ) {
            | (Some((left, _)), Some((right, _))) => right.cmp(left),
            | (Some(_), None) => Ordering::Less,
            | (None, Some(_)) => Ordering::Greater,
            | (None, None) => self.index.cmp(&other.index).reverse(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::Bound;

    use bytes::Bytes;

    use super::*;
    use crate::{
        keypair::{
            KeyBytes,
            ValueBytes,
            DEFAULT_NS,
        },
        memtable::Memtable,
    };

    // Helper function to create a test key-value pair
    fn create_kv<S: Into<String>>(ns: u64, key: S, ts: u128, value: S) -> (KeyBytes, ValueBytes) {
        (
            KeyBytes::new(ns, Bytes::from(key.into()), ts),
            ValueBytes::new(ns, Bytes::from(value.into())),
        )
    }

    #[test]
    fn test_empty_iterator() {
        let empty_vec: Vec<Vec<(KeyBytes, ValueBytes)>> = vec![];
        let iters = empty_vec.into_iter().map(IntoIterator::into_iter).collect();
        let mut merge_iter = MergeIterator::new(iters);
        assert!(merge_iter.next().is_none());
    }

    #[test]
    fn test_single_iterator() {
        let data = vec![
            create_kv(DEFAULT_NS, "key1", 1, "value1"),
            create_kv(DEFAULT_NS, "key2", 2, "value2"),
        ];
        let iters = vec![data.clone().into_iter()];
        let mut merge_iter = MergeIterator::new(iters);

        for expected in data {
            let actual = merge_iter.next();
            assert_eq!(Some(expected), actual);
        }
        assert!(merge_iter.next().is_none());
    }

    #[test]
    fn test_version_ordering() {
        // Test ordering with same key, different versions
        let data1 = vec![create_kv(DEFAULT_NS, "key1", 3, "value1_v3")];
        let data2 = vec![create_kv(DEFAULT_NS, "key1", 2, "value1_v2")];
        let data3 = vec![create_kv(DEFAULT_NS, "key1", 1, "value1_v1")];

        let iters = vec![data1.into_iter(), data2.into_iter(), data3.into_iter()];
        let merge_iter = MergeIterator::new(iters);

        let results: Vec<_> = merge_iter.collect();
        assert_eq!(results.len(), 3);

        // Verify timestamps are in descending order
        assert_eq!(results[0].0.ts(), 3); // newest
        assert_eq!(results[1].0.ts(), 2);
        assert_eq!(results[2].0.ts(), 1); // oldest
    }

    #[test]
    fn test_namespace_ordering() {
        // Test ordering with same key, different namespaces
        let data1 = vec![create_kv(2, "key1", 1, "ns2_value1")];
        let data2 = vec![create_kv(1, "key1", 1, "ns1_value1")];

        let iters = vec![data1.into_iter(), data2.into_iter()];
        let merge_iter = MergeIterator::new(iters);

        let results: Vec<_> = merge_iter.collect();
        assert_eq!(results.len(), 2);

        // Verify namespaces are in ascending order
        assert_eq!(results[0].0.ns(), 1);
        assert_eq!(results[1].0.ns(), 2);
    }

    #[test]
    fn test_complex_merge() {
        let iter1 = vec![
            create_kv(1, "key1", 3, "ns1_key1_v3"),
            create_kv(2, "key1", 2, "ns2_key1_v2"),
        ]
        .into_iter();
        let iter2 = vec![
            create_kv(1, "key1", 2, "ns1_key1_v2"),
            create_kv(2, "key1", 3, "ns2_key1_v3"),
        ]
        .into_iter();
        let iter3 = vec![
            create_kv(1, "key1", 1, "ns1_key1_v1"),
            create_kv(2, "key1", 1, "ns2_key1_v1"),
        ]
        .into_iter();

        let iters = vec![iter1, iter2, iter3];
        let mut merge_iter = MergeIterator::new(iters);

        let results: Vec<_> = merge_iter.collect();
        assert_eq!(results.len(), 6);

        assert_eq!(results[0].0.ns(), 1);
        assert_eq!(results[0].0.ts(), 3);
        assert_eq!(results[1].0.ns(), 1);
        assert_eq!(results[1].0.ts(), 2);
        assert_eq!(results[2].0.ns(), 1);
        assert_eq!(results[2].0.ts(), 1);
        assert_eq!(results[3].0.ns(), 2);
        assert_eq!(results[3].0.ts(), 3);
        assert_eq!(results[4].0.ns(), 2);
        assert_eq!(results[4].0.ts(), 2);
        assert_eq!(results[5].0.ns(), 2);
        assert_eq!(results[5].0.ts(), 1);
    }

    #[test]
    fn test_iterator_exhaustion() {
        let iter1 = vec![create_kv(DEFAULT_NS, "key1", 1, "value1")].into_iter();
        let iter2 = vec![create_kv(DEFAULT_NS, "key2", 1, "value2")].into_iter();
        let iter3 = Vec::<(KeyBytes, ValueBytes)>::new().into_iter();

        let mut results = Vec::new();
        let mut merge_iter = MergeIterator::new(vec![iter1, iter2, iter3]);
        while let Some(item) = merge_iter.next() {
            results.push(item);
        }

        assert_eq!(results.len(), 2);
        assert_eq!(String::from_utf8_lossy(&results[0].1.value), "value1");
        assert_eq!(String::from_utf8_lossy(&results[1].1.value), "value2");
    }

    #[test]
    fn test_with_memtable_iterators() {
        let memtable1 = Memtable::new(1, 1024 * 1024);
        let memtable2 = Memtable::new(2, 1024 * 1024);

        memtable1
            .put(
                KeyBytes::new(DEFAULT_NS, Bytes::from("key1"), 3),
                ValueBytes::new(DEFAULT_NS, Bytes::from("value1_v3")),
            )
            .unwrap();
        memtable1
            .put(
                KeyBytes::new(DEFAULT_NS, Bytes::from("key2"), 1),
                ValueBytes::new(DEFAULT_NS, Bytes::from("value2_v1")),
            )
            .unwrap();

        memtable2
            .put(
                KeyBytes::new(DEFAULT_NS, Bytes::from("key1"), 2),
                ValueBytes::new(DEFAULT_NS, Bytes::from("value1_v2")),
            )
            .unwrap();
        memtable2
            .put(
                KeyBytes::new(DEFAULT_NS, Bytes::from("key2"), 2),
                ValueBytes::new(DEFAULT_NS, Bytes::from("value2_v2")),
            )
            .unwrap();

        let iter1 = memtable1.scan(Bound::Unbounded, Bound::Unbounded);
        let iter2 = memtable2.scan(Bound::Unbounded, Bound::Unbounded);
        let merge_iter = MergeIterator::new(vec![iter1, iter2]);

        let results: Vec<_> = merge_iter.collect();

        assert_eq!(
            results.len(),
            4,
            "Should have exactly 4 entries after filtering"
        );
        assert_eq!(String::from_utf8_lossy(&results[0].1.value), "value1_v3");
        assert_eq!(String::from_utf8_lossy(&results[1].1.value), "value1_v2");
        assert_eq!(String::from_utf8_lossy(&results[2].1.value), "value2_v2");
        assert_eq!(String::from_utf8_lossy(&results[3].1.value), "value2_v1");
    }

    #[test]
    fn test_size_hint() {
        let iter1 = vec![
            create_kv(DEFAULT_NS, "key1", 1, "value1"),
            create_kv(DEFAULT_NS, "key2", 1, "value2"),
        ]
        .into_iter();
        let iter2 = vec![
            create_kv(DEFAULT_NS, "key3", 1, "value3"),
            create_kv(DEFAULT_NS, "key4", 1, "value4"),
        ]
        .into_iter();

        let iters = vec![iter1, iter2];
        let merge_iter = MergeIterator::new(iters);

        let (lower, upper) = merge_iter.size_hint();
        assert_eq!(lower, 4);
        assert_eq!(upper, None);
    }
}
