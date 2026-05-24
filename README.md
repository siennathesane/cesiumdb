[![codecov](https://codecov.io/gh/siennathesane/cesiumdb/graph/badge.svg?token=D7RBD3OX2U)](https://codecov.io/gh/siennathesane/cesiumdb)

# CesiumDB

A key-value store focused on performance.

# Usage Note: Beta Software

CesiumDB is in beta. The API is stable and the database itself is stable — the core LSM-tree (memtables, levels L0–L7, flushes, and compaction) is fully functional and extensively tested. The on-disk format is stable. The remaining work is tuning out the last performance kinks, particularly around compaction stall behaviour under heavy concurrent write load. It turns out Facebook as right, performance tuning an LSM-tree is hard 😒

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

* LSM-tree with tiered L0–L2 and leveled L3–L7 compaction
* Configurable per-level target segment sizes via `target_file_size_multiplier`
* Background compaction scheduler with autoconfiguration support
* Namespaces for logical key grouping within a single LSM-tree
* Hybrid Logical Clock (HLC) for deterministic versioning
* Memory-mapped segment files with bloom-filter-accelerated lookups

### How _Fast_ is Fast?

I'm glad you asked! Here are some benchmarks from the built-in `bench` binary (Apple Silicon M1, release build, 8 threads):

| Workload | Value Size | Ops/sec | µs/op | MB/s | P99.99 |
|----------|-----------|---------|-------|------|--------|
| fillrandom | 400 B | ~646K | 1.55 | 246 | 4.6 ms |

Internal micro-benchmarks:

* Bloom filter lookups: ~860 _picoseconds_
* Merge operator: ~115 ms for a full table scan of 800,000 keys across 8 memtables

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
cesiumdb = "0.1.0"
```

And use:

```rust
use cesiumdb::{Db, DbOptions};

let mut opts = DbOptions::default();
opts.data_dir(std::path::PathBuf::from("/var/lib/cesiumdb"));
let db = Db::open(opts);

// simple put/get
db.put(b"key", b"value").unwrap();
let value = db.get(b"key").unwrap();

// with a namespace
db.put_ns(1, b"key", b"value").unwrap();
let value = db.get_ns(1, b"key").unwrap();
```

See the [API documentation](https://docs.rs/cesiumdb) for more information.

## Namespaces are not Column Families

CesiumDB uses a construct I call "namespacing". It's a way for data of a similar type to be grouped together, but it is
not stored separately than other namespaced data. Namespaces are ultimately glorified range markers to ensure fast data
lookups across a large set of internal data, and a bit of a way to make it easy for users to manage their data. I would
argue namespaces are closer to tables than column families.

## Hybrid Logical Clocks

CesiumDB does let you bring your own hybrid logical clock implementation for key versioning. This is useful if you have
a specific HLC implementation you want to use, or if you want to use a different clock entirely. This is done by
implementing the `HLC` trait and passing it to the `DbOptions` clock setter. However, if you can provide a more precise
clock than the provided one, please submit an issue or PR so we can all benefit from it.

## Benchmarking

A `db_bench`-style benchmark binary is included:

```bash
cargo build --release --bin bench

# 1M random writes, 400 B values, 8 threads
./target/release/bench --benchmarks=fillrandom,stats --num=1000000 --threads=8

# Tune segment sizes
TARGET_SEGMENT_SIZE_MB=64 TARGET_FILE_SIZE_MULTIPLIER=2 ./benchmark.sh fillrandom
```

See `benchmark.sh` for available environment variables.

## Unsafety: Or... How To Do Dangerous Things Safely

There is a non-trivial amount of `unsafe` code. Most of it is related to the internal implementation with `mmap` (which
cannot be made safe) and its entrypoints (the handlers and such). I also make use of pointer arithmetic on
memory-mapped file locations. This is one of the areas where safety comes at the cost of performance. However, if you
can find a way to make it safe, please submit an issue or PR. I would love to see it!

There is :sparkles: __EXTENSIVE__ :sparkles: testing around the `unsafe` code, and I am confident in its correctness. My
goal is to keep this project at a high degree of code coverage with tests to help continue to ensure said confidence.
However, if you find a bug, please submit an issue or PR.

## Contributing

Contributions are welcome! Please submit a PR with your changes. If you're unsure about the changes, please submit an
issue first.

## To Do's

An alphabetical list of things I'd like to actually do for the long-term safety and stability of the project.

- [x] Write some kind of auto-configuration for the generalized configs.
- [ ] Add `loom` integration tests.
- [ ] Add `miri` integration tests.
- [ ] Add more granular `madvise` commands to the filesystem to give the kernel some hints.
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

## License

CesiumDB is licensed under GPL v3.0 with the Class Path Exception. This means you can safely link to CesiumDB in your
project. So it's safe for corporate consumption, just not closed-source modification :simple_smile:

If you would like a non-GPL license, please reach out :simple_smile:
