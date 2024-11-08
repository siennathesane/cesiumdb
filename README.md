# CesiumDB

A key-value store focused on performance, security, and stability.

## Philosophy

CesiumDB is designed around the principle that the best database is one you can trust and maintain, not necessarily the one with the most theoretical optimizations. The architectural choices reflect this philosophy:

### Core Design Principles

1. **Simplicity Over Complexity**
   - Hybrid architecture combining proven data structures: skiplist memtables and B+tree disk storage
   - Minimal moving parts: no complex compaction strategies or level management
   - Each component is independently testable and auditable

2. **Pragmatic Performance**
   - Fast writes through in-memory skiplists
   - Efficient reads via memory-mapped B+tree (LMDB-inspired)
   - Handles both burst writes and read-heavy workloads without specialized tuning
   - High-precision hybrid logical clock for versioning without MVCC complexity

3. **Clear Guarantees**
   - Monotonic timestamps ensure consistent ordering
   - Last-write-wins semantics simplify concurrency
   - Single version per key on disk reduces complexity
   - Predictable performance characteristics

4. **Maintainable Architecture**
   - Each component does one thing well
   - Clear boundaries between subsystems
   - Inspired by battle-tested designs (LMDB's B+tree, skip lists)
   - Prioritizes code clarity and auditability

## Design Choices

- **Memtables**: Size-bounded skip lists for efficient concurrent writes
- **Disk Storage**: Simple B+tree implementation for reliable reads
- **Versioning**: Hybrid logical clock provides monotonic ordering without MVCC overhead
- **Transactions**: App-level locking when needed, avoiding complex MVCC

## Trade-offs

I explicitly chose to forgo certain optimizations in favor of maintainability:
- No LSM-style leveled compaction
- Single version per key on disk via last write wins
- No transactions

These trade-offs mean CesiumDB might not be the absolute fastest for specific workloads, but it will be:
- Easier to maintain
- Easier to audit
- More predictable
- More reliable

My philosophy is that a database should be a trustworthy foundation for building systems, not a constant source of operational complexity.

## License

CesiumDB is licensed under GPL v3.0 with the Class Path Exception. This means you can use CesiumDB in your project without having to open source your project. However, if you modify CesiumDB, you must open source your changes. If you would like a non-GPL license, please reach out :smile:

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

## Security

If enabled, CesiumDB can use user-provided encryption keys to secure your data the moment the database takes ownership of your data. As this is a non-trivial performance hit, somewhere between 10-30%, it is disabled by default.

Further, we can securely encrypt every internal allocation (including when we load your already encrypted data) so your data will be very safe.

- User-provided encryption key
- Encrypted in-memory the moment CesiumDB takes ownership
- Securely loaded from disk with checksumming
- Encrypted in-memory
