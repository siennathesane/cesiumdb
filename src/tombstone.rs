use std::{
    alloc::{
        alloc_zeroed,
        Layout,
    },
    collections::BTreeMap,
    ops::Bound::{
        Excluded,
        Unbounded,
    },
};

pub trait Tombstone {
    fn marker(&self) -> u64;
}

impl Tombstone for u64 {
    fn marker(&self) -> u64 {
        *self
    }
}

// nb (sienna): use `Tombstone.marker` as the namespace separators. so we sort
// the array first by tombstone then by whatever their comparators are, then
// each tombstone is the index value of when the next tombstone starts. this
// lets us optimize a binary search into a small subset of the entire array at
// the cost of space but the space is logrithmic and only grows based on the
// amount of namespaces. this is sorted small on the left, big on the right
// TODO(@siennathesane): this needs to be statically sized
pub struct TombstonedVec<T> {
    buf: Vec<T>,
    tombstones: BTreeMap<u64, u64>, // val: start idx
}

impl<T> TombstonedVec<T>
where
    T: Tombstone + Ord + Clone,
{
    pub fn new() -> Self {
        TombstonedVec {
            // TODO(@siennathesane): add the tombstone vec to the total memory capacity
            buf: Vec::with_capacity(1_000_000),
            tombstones: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, val: T) {
        let marker = val.marker();

        let insert_pos = match self.tombstones.range(..=marker).next_back() {
            | Some((&k, &v)) if k == marker => {
                // existing tombstone, find position within the bucket
                let end = self.next_idx(k);
                v + self.buf[v as usize..end as usize]
                    .binary_search(&val)
                    .unwrap_or_else(|e| e) as u64
            },
            | Some((&k, &v)) => {
                // new tombstone, insert at the end of the previous bucket
                let next_idx = self.next_idx(k);
                self.tombstones.insert(marker, next_idx);
                next_idx
            },
            | None => {
                // First tombstone
                self.tombstones.insert(marker, 0);
                0
            },
        };

        // insert the value
        self.buf.insert(insert_pos as usize, val);

        // update subsequent tombstones
        for (_, idx) in self.tombstones.range_mut((Excluded(marker), Unbounded)) {
            *idx += 1;
        }
    }

    /// Determines if a tombstone marker exists in this specific Vec.
    pub fn has_marker(&self, marker: u64) -> bool {
        self.tombstones.contains_key(&marker)
    }

    pub fn get(&self, marker: u64, idx: u64) -> Option<&T> {
        if !self.tombstones.contains_key(&marker) {
            return None;
        }

        Some(self.buf.get(idx as usize).unwrap())
    }

    fn next_idx(&self, marker: u64) -> u64 {
        self.tombstones
            .range((Excluded(marker), Unbounded))
            .next()
            .map(|(_, &idx)| idx)
            .unwrap_or(self.buf.len() as u64)
    }
}

fn zeroed_box<T>() -> Box<T> {
    try_zeroed_box().unwrap()
}

fn try_zeroed_box<T>() -> Result<Box<T>, ()> {
    if size_of::<T>() == 0 {
        // This will not allocate but simply create an arbitrary non-null
        // aligned pointer, valid for Box for a zero-sized pointee.
        let ptr = core::ptr::NonNull::dangling().as_ptr();
        return Ok(unsafe { Box::from_raw(ptr) });
    }
    let layout = Layout::new::<T>();
    let ptr = unsafe { alloc_zeroed(layout) };
    if ptr.is_null() {
        // we don't know what the error is because `alloc_zeroed` is a dumb API
        Err(())
    } else {
        Ok(unsafe { Box::<T>::from_raw(ptr as *mut T) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
    struct TestItem {
        marker: u64,
        value: u64,
    }

    impl Tombstone for TestItem {
        fn marker(&self) -> u64 {
            self.marker
        }
    }

    #[test]
    fn test_new() {
        let vec: TombstonedVec<TestItem> = TombstonedVec::new();
        assert!(vec.buf.is_empty());
        assert!(vec.tombstones.is_empty());
    }

    #[test]
    fn test_insert_single_tombstone() {
        let mut vec = TombstonedVec::new();
        vec.insert(TestItem {
            marker: 1,
            value: 10,
        });
        assert_eq!(
            vec.buf,
            vec![TestItem {
                marker: 1,
                value: 10
            }]
        );
        assert_eq!(vec.tombstones, [(1, 0)].into_iter().collect());
    }

    #[test]
    fn test_insert_multiple_tombstones() {
        let mut vec = TombstonedVec::new();
        vec.insert(TestItem {
            marker: 1,
            value: 10,
        });
        vec.insert(TestItem {
            marker: 2,
            value: 20,
        });
        vec.insert(TestItem {
            marker: 3,
            value: 30,
        });
        assert_eq!(
            vec.buf,
            vec![
                TestItem {
                    marker: 1,
                    value: 10
                },
                TestItem {
                    marker: 2,
                    value: 20
                },
                TestItem {
                    marker: 3,
                    value: 30
                },
            ]
        );
        assert_eq!(
            vec.tombstones,
            [(1, 0), (2, 1), (3, 2)].into_iter().collect()
        );
    }

    #[test]
    fn test_insert_same_tombstone() {
        let mut vec = TombstonedVec::new();
        vec.insert(TestItem {
            marker: 1,
            value: 10,
        });
        vec.insert(TestItem {
            marker: 1,
            value: 5,
        });
        vec.insert(TestItem {
            marker: 1,
            value: 15,
        });
        assert_eq!(
            vec.buf,
            vec![
                TestItem {
                    marker: 1,
                    value: 5
                },
                TestItem {
                    marker: 1,
                    value: 10
                },
                TestItem {
                    marker: 1,
                    value: 15
                },
            ]
        );
        assert_eq!(vec.tombstones, [(1, 0)].into_iter().collect());
    }

    #[test]
    fn test_insert_mixed_order() {
        let mut vec = TombstonedVec::new();
        vec.insert(TestItem {
            marker: 2,
            value: 20,
        });
        vec.insert(TestItem {
            marker: 1,
            value: 10,
        });
        vec.insert(TestItem {
            marker: 3,
            value: 30,
        });
        vec.insert(TestItem {
            marker: 2,
            value: 25,
        });
        vec.insert(TestItem {
            marker: 1,
            value: 15,
        });
        assert_eq!(
            vec.buf,
            vec![
                TestItem {
                    marker: 1,
                    value: 10
                },
                TestItem {
                    marker: 1,
                    value: 15
                },
                TestItem {
                    marker: 2,
                    value: 20
                },
                TestItem {
                    marker: 2,
                    value: 25
                },
                TestItem {
                    marker: 3,
                    value: 30
                },
            ]
        );
        assert_eq!(
            vec.tombstones,
            [(1, 0), (2, 2), (3, 4)].into_iter().collect()
        );
    }

    #[test]
    fn test_insert_large_gap_between_tombstones() {
        let mut vec = TombstonedVec::new();
        vec.insert(TestItem {
            marker: 1,
            value: 10,
        });
        vec.insert(TestItem {
            marker: 1000,
            value: 1000,
        });
        vec.insert(TestItem {
            marker: 500,
            value: 500,
        });
        assert_eq!(
            vec.buf,
            vec![
                TestItem {
                    marker: 1,
                    value: 10
                },
                TestItem {
                    marker: 500,
                    value: 500
                },
                TestItem {
                    marker: 1000,
                    value: 1000
                },
            ]
        );
        assert_eq!(
            vec.tombstones,
            [(1, 0), (500, 1), (1000, 2)].into_iter().collect()
        );
    }

    #[test]
    fn test_insert_duplicate_values() {
        let mut vec = TombstonedVec::new();
        vec.insert(TestItem {
            marker: 1,
            value: 10,
        });
        vec.insert(TestItem {
            marker: 1,
            value: 10,
        });
        vec.insert(TestItem {
            marker: 2,
            value: 20,
        });
        vec.insert(TestItem {
            marker: 2,
            value: 20,
        });
        assert_eq!(
            vec.buf,
            vec![
                TestItem {
                    marker: 1,
                    value: 10
                },
                TestItem {
                    marker: 1,
                    value: 10
                },
                TestItem {
                    marker: 2,
                    value: 20
                },
                TestItem {
                    marker: 2,
                    value: 20
                },
            ]
        );
        assert_eq!(vec.tombstones, [(1, 0), (2, 2)].into_iter().collect());
    }

    #[test]
    fn test_insert_many_items() {
        let mut vec = TombstonedVec::new();
        for i in 0..1000 {
            vec.insert(TestItem {
                marker: i / 100,
                value: i,
            });
        }
        assert_eq!(vec.buf.len(), 1000);
        assert_eq!(vec.tombstones.len(), 10);
        for i in 0..10 {
            assert_eq!(vec.tombstones[&(i as u64)], (i * 100) as u64);
        }
    }

    #[test]
    fn test_insert_reverse_order() {
        let mut vec = TombstonedVec::new();
        for i in (0..10).rev() {
            vec.insert(TestItem {
                marker: i,
                value: i * 10,
            });
        }
        assert_eq!(
            vec.buf,
            (0..10)
                .map(|i| TestItem {
                    marker: i,
                    value: i * 10
                })
                .collect::<Vec<_>>()
        );
        assert_eq!(vec.tombstones, (0..10).map(|i| (i, i as u64)).collect());
    }
}
