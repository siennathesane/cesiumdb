[![Build & Test](https://github.com/siennathesane/cesiumdb/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/siennathesane/cesiumdb/actions/workflows/build-and-test.yml)
[![codecov](https://codecov.io/gh/siennathesane/cesiumdb/graph/badge.svg?token=D7RBD3OX2U)](https://codecov.io/gh/siennathesane/cesiumdb)

# CesiumDB

A key-value store focused on performance, security, and stability.

## License

CesiumDB is licensed under GPL v3.0 with the Class Path Exception. This means you can safely link to CesiumDB in your project. So it's safe for corporate consumption, just not closed-source modification :simple_smile:

If you would like a non-GPL license, please reach out :simple_smile:

## MVCC

CesiumDB doesn't contain MVCC semantics due to the use of a hybrid linear clock (HLC). This provides guaranteed operation ordering based on the database's view of the data after it enters the boundary; operations are linear and non-collidable. This removes a transaction API and pushes the responsibility of correct ordering to the application via "last write wins". This is a tradeoff between ergonomics & maintainability for everyone. Application owners know their application best, and it's easier to reason about the ordering of data operations in the application layer.

While the HLC is monotonic, it is also exceedingly performant with nanosecond precision. This allows for a high degree of concurrency and parallelism. As an example, on @siennathesane's Macbook Pro M1 Pro chipset, the clock has a general resolution of about 2 nanoseconds.

If you have your heart set on transactions, you can wrap the database in a `MutexGuard` or `RwLock` to provide transactional semantics. Like this:

```rust
use std::sync::{Mutex, MutexGuard};
let db = Mutex::new(CesiumDB::new());
{
    let mut tx: MutexGuard<CesiumDB> = db.lock().unwrap();
    tx.put("key", "value");
    tx.sync();
}
// other non-tx operations
```

### BYOHLC

CesiumDB does let you bring your own hybrid logical clock implementation. This is useful if you have a specific HLC implementation you want to use, or if you want to use a different clock entirely. This is done by implementing the `HLC` trait and passing it to the `CesiumDB` constructor. However, if you can provide a more precise clock than the provided one, please submit an issue or PR so we can all benefit from it.

## Unsafety: Or... How To Do Dangerous Things Safely

There is a non-trivial amount of `unsafe` code. Most of it is related to the internal filesystem implementation with `mmap` (which cannot be made safe).

Internally, the filesystem I built for CesiumDB is a lock-free, thread-safe portable filesystem since one of my use cases is an embedded system that doesn't have a filesystem, only a device driver. LMDB is a huge inspiration for this project, so I wanted to utilize a lot of the same methodologies around `mmap`, but to make it as safe as possible. The nifty part is that Linux doesn't distinguish between a file and a block device for `mmap`, so I can `mmap` a block device and treat it like a file. The perk is that we get native write speeds for the device, we have a bin-packing filesystem that is portable across devices, and if all else fails, we can just `fallocate` a file and use that. The downside is that writing directly to device memory is dangerous and is inherently "unsafe", so a lot of the optimizations are `unsafe` because of this.

There is :sparkles: __EXTENSIVE__ :sparkles: testing around the `unsafe` code, and I am confident in its correctness. My goal is to keep this project at a high degree of code coverage with tests to help continue to ensure said confidence. However, if you find a bug, please submit an issue or PR.

## Contributing

Contributions are welcome! Please submit a PR with your changes. If you're unsure about the changes, please submit an issue first.

I will not be accepting any pull requests which contain `async` code.

## To Do's

Things I'd like to actually do, preferably before other people consume this.

- [ ] Bloom filter size is currently hardcoded. I'd like to make it configurable.
- [ ] Add some kind of `fallocate` automation or growth strategy for the filesystem when it's not a block device.
- [ ] Write some kind of auto-configuration for the generalized configs.
- [ ] Investigate the point at which we can no longer `mmap` a physical device. Theoretically, even without swap space, I can `mmap` a 1TiB physical device to the filesystem implementation. But I feel like shit gets real weird.
- [ ] Figure out how hard it would be to support `no_std` for the embedded workloads. I suspect it would require a custom variation of `std::collections::BinaryHeap`, which would be... difficult lol
- [ ] Add `miri` integration tests.
- [ ] Add `loom` integration tests.
- [ ] Revisit the merge operator because it seems... slow? Idk. 115ms to do a full table scan of 800,000 keys across 8 memtables feels really slow.
- [ ] Add more granular `madvise` commands to the filesystem to give the kernel some hints.
- [ ]  Revisit the merge iterator. The benchmarks have it at ~120ms for a full scan of 8 memtables with 100,000 keys each. I have no idea if this is a mismatch of my expectations or a gross inability of mine to optimize it further. Every optimization I've tried is 5-20% slower (including my own cache-optimized min heap) than this.