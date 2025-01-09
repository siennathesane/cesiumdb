[![builds.sr.ht status](https://builds.sr.ht/~siennathesane/cesiumdb/commits/feat/builds/amd64.yml.svg)](https://builds.sr.ht/~siennathesane/cesiumdb/commits/feat/builds/amd64.yml?)
[![codecov](https://codecov.io/gh/siennathesane/cesiumdb/graph/badge.svg?token=D7RBD3OX2U)](https://codecov.io/gh/siennathesane/cesiumdb)

# CesiumDB

A key-value store focused on performance.

# Work In Progress

This project is an active work-in-progress.

It will likely compile, and most tests will likely pass, but it is not feature complete yet. The current state of work
is stabilizing the embedded filesystem implementation so the front end memtables can rely on the backend embedded
filesystem. Once that work is done, then it's just implementing levels (relatively easy) and compaction (easy enough).

## Inspiration

This project was heavily inspired and influenced by (in no particular order):

* Long compile times for Facebook's `rocksdb`
* Howard Chu's `lmdb`
* CockroachDB's `pebble`
* Ben Johnson's `boltdb`
* Google's `leveldb`
* Giorgos Xanthakis et al's `parallax`
* A burning desire to have a rust-native LSM-tree that has column family/namespace support

## Interesting Features

It's :sparkles: __FAST__ :sparkles: and has a few interesting features:

* A blazingly fast hybrid logical clock (HLC) for ordering operations instead of MVCC semantics
* A high-performance, lock-free, thread-safe, portable filesystem that works with block devices
* An insanely fast bloom filter for fast lookups

### How _Fast_ is Fast?

I'm glad you asked! Here are some benchmarks:

* Internal bloom filter lookups: ~860 _picoseconds_
* Merge operator: ~115ms for a full table scan of 800,000 keys across 8 memtables

## Usage

Add this to your `Cargo.toml`:

```toml

[dependencies]
cesiumdb = "1.0"
```

And use:

```rust
use cesiumdb::CesiumDB;

// use a temp file, most useful for testing
let db = CesiumDB::default ();

// no namespace
db.put(b"key", b"value");
db.get(b"key");

// with a namespace
db.put(1, b"key", b"value");
db.get(1, b"key");
```

See the [API documentation](https://docs.rs/cesiumdb) for more information.

## Namespaces are not Column Families

CesiumDB uses a construct I call "namespacing". It's a way for data of a similar type to be grouped together, but it is
not stored separately than other namespaced data. Namespaces are ultimately glorified range markers to ensure fast data
lookups across a large set of internal data, and a bit of a way to make it easy for users to manage their data. I would
argue namespaces are closer to tables than column families.

## MVCC is... Not Here

CesiumDB doesn't contain MVCC semantics due to the use of a hybrid linear clock (HLC). This provides guaranteed
operation ordering based on the database's view of the data after it enters the boundary; operations are linear and
non-collidable. This removes a transaction API and pushes the responsibility of correct ordering to the application
via "last write wins". This is a tradeoff between ergonomics & maintainability for everyone. Application owners know
their application best, and it's easier to reason about the ordering of data operations in the application layer.

While the HLC is monotonic, it is also exceedingly performant with nanosecond precision. This allows for a high degree
of concurrency and parallelism. As an example, on @siennathesane's Macbook Pro M1 Pro chipset, the clock has a general
resolution of about 2 nanoseconds.

If you have your heart set on transactions, you can wrap the database in a `MutexGuard` or `RwLock` to provide
transactional semantics. Like this:

```rust
use std::sync::{Mutex, MutexGuard};
let db = Mutex::new(CesiumDB::new());
{
let mut tx: MutexGuard < CesiumDB > = db.lock().unwrap();
tx.put("key", "value");
tx.sync();
}
// other non-tx operations
```

### BYOHLC

CesiumDB does let you bring your own hybrid logical clock implementation. This is useful if you have a specific HLC
implementation you want to use, or if you want to use a different clock entirely. This is done by implementing the `HLC`
trait and passing it to the `CesiumDB` constructor. However, if you can provide a more precise clock than the provided
one, please submit an issue or PR so we can all benefit from it.

## Unsafety: Or... How To Do Dangerous Things Safely

There is a non-trivial amount of `unsafe` code. Most of it is related to the internal filesystem implementation with
`mmap` (which cannot be made safe) and it's entrypoints (the handlers and such).

Internally, the filesystem I built for CesiumDB is a lock-free, thread-safe portable filesystem since one of my use
cases is an embedded system that doesn't have a filesystem, only a device driver. LMDB is a huge inspiration for this
project, so I wanted to utilize a lot of the same methodologies around `mmap`, but to make it as safe as possible. The
nifty part is that Linux doesn't distinguish between a file and a block device for `mmap`, so I can `mmap` a block
device and treat it like a file. The perk is that we get native write speeds for the device, we have a bin-packing
filesystem that is portable across devices, and if all else fails, we can just `fallocate` a file and use that. The
downside is that writing directly to device memory is dangerous and is inherently "unsafe", so a lot of the
optimizations are `unsafe` because of this.

There is :sparkles: __EXTENSIVE__ :sparkles: testing around the `unsafe` code, and I am confident in its correctness. My
goal is to keep this project at a high degree of code coverage with tests to help continue to ensure said confidence.
However, if you find a bug, please submit an issue or PR.

## Contributing

Contributions are welcome! Please submit a PR with your changes. If you're unsure about the changes, please submit an
issue first.

I will only accept `async` code if it is in the hot path for compaction or flushing, and it can't be handled with a
thread.

## To Do's

An alphabetical list of things I'd like to actually do for the long-term safety and stability of the project.

- [ ] Add `loom` integration tests.
- [ ] Add `miri` integration tests.
- [ ] Add more granular `madvise` commands to the filesystem to give the kernel some hints.
- [ ] Add some kind of `fallocate` automation or growth strategy for the filesystem when it's not a block device.
- [ ] Add some kind of `fsck` and block checksums since journaling is already present. There are basic unit tests for
  this but no supported tool for it.
- [ ] Bloom filter size is currently hardcoded. I'd like to make it configurable.
- [ ] Determine how to expose the untrustworthiness of the bloom filter.
- [ ] Figure out how hard it would be to support `no_std` for the embedded workloads. I suspect it would be... difficult
  lol
- [ ] Investigate the point at which we can no longer `mmap` a physical device. Theoretically, even without swap space,
  I can `mmap` a 1TiB physical device to the filesystem implementation. But I feel like shit gets real weird. Idk, it's
  a Linux-ism I want to investigate.
- [ ] Remove the question mark operator.
- [ ] Revisit the merge iterator. The benchmarks have it at ~115ms for a full scan of 8 memtables with 100,000 keys
  each. I have no idea if this is a mismatch of my expectations or a gross inability of mine to optimize it further.
  Every optimization I've tried is 5-20% slower (including my own cache-optimized min heap) than this.
- [ ] Write some kind of auto-configuration for the generalized configs.

## License

CesiumDB is licensed under GPL v3.0 with the Class Path Exception. This means you can safely link to CesiumDB in your
project. So it's safe for corporate consumption, just not closed-source modification :simple_smile:

If you would like a non-GPL license, please reach out :simple_smile:
