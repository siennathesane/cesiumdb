# CesiumDB

A key-value store focused on flexibility and performance. It is currently under active development and features are either missing or incomplete. Check on the [features](#features) section to see where things are.

## About

CesiumDB is a key-value store with time series support. Internally, it uses a log-structured merge-tree that is based off [Badger](https://github.com/dgraph-io/badger), [RocksDB](https://github.com/facebook/rocksdb), and [WiscKey](https://www.usenix.org/system/files/conference/fast16/fast16-papers-lu.pdf). CesiumDB is not disk-compatible with any of the aforementioned solutions due to different data formats, but using it should feel similar. The core architecture of CesiumDB - the LSM-tree and SSTables - is  influenced by Badger v4.

## Features

- [] Simple `Get` / `Put` / `Delete` APIs
- [] Prefix trees for segmenting data
- [] Built-in timing support with optional clock replacement
- [] Adaptive cuckoo filters
- [] SSTables
- [] SSTable block cache
- [] Key index cache
- [] Time series index cache
- [] Atomic transaction support
- [] Online compaction?
- [] Disk compression
- [] Disk encryption
