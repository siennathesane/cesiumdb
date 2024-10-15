# CesiumDB

A key-value store focused on flexibility and performance. It is currently under active development and features are either missing or incomplete. Check on the [features](#features) section to see where things are.

## About

CesiumDB is a key-value store with time series support. Internally, it uses a log-structured merge-tree that is based off [Badger](https://github.com/dgraph-io/badger), [RocksDB](https://github.com/facebook/rocksdb), and [WiscKey](https://www.usenix.org/system/files/conference/fast16/fast16-papers-lu.pdf). CesiumDB is not disk-compatible with any of the aforementioned solutions due to different data formats, but using it should feel similar. The core architecture of CesiumDB is a Rust-specific implementation of Parallax

## References

```doi
Giorgos Xanthakis, Giorgos Saloustros, Nikos Batsaras, Anastasios Papagiannis, and Angelos Bilas. 2021. Parallax: Hybrid Key-Value Placement in LSM-based Key-Value Stores. In Proceedings of the ACM Symposium on Cloud Computing (SoCC '21). Association for Computing Machinery, New York, NY, USA, 305–318.
DOI:https://doi.org/10.1145/3472883.3487012
```