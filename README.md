# CesiumDB

A key-value store focused on performance and security.

## About

CesiumDB is a key-value store. Internally, it's an LSM-tree with a whack of optimizations. Some of these optimizations include:

- Key-value separations, keys are stored in different files than values
- Key-size separations (pending)
- Direct device access via raw syscalls
- Reference-only memcpys

## MVCC

CesiumDB doesn't contain MVCC semantics due to the use of a hybrid linear clock (HLC). This provides guaranteed operation ordering based on the database's view of the data after it enters the boundary; operations are linear and non-collidable. This removes a transaction API and pushes the responsibility of correct ordering to the application. This is a tradeoff between ergonomics & maintainability. Application owners know their application best, and it's easier to reason about the ordering of operations in the application layer.

While the HLC is monotonic, it is also exceedingly performant with nanosecond precision. This allows for a an exceedingly degree of concurrency and parallelism. As an example, on @siennathesane's Macbook Pro M1 Pro chipset, the clock has a general resolution of about 2 nanoseconds. So if you can write lock-free, highly concurrent code, you can achieve _very_ high throughput.

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
