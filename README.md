# CesiumDB

A key-value store focused on performance, security, and stability.

## License

CesiumDB is licensed under GPL v3.0 with the Class Path Exception. This means you can link to CesiumDB in your project without having to open source your project. So it's safe for corporate consumption, just not closed-source modification :simple_smile:

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

## To Do's

- [ ] Investigate the point at which we can no longer `mmap` a physical device. Theoretically, even without swap space, I can `mmap` a 1TiB physical device to the filesystem implementation. But I feel like shit gets real weird. Unfortunately I don't have the ability to test it so volunteers would be great!
- [ ] Figure out how hard it would be to support `no_std` for the embedded workloads. I suspect it would require a custom variation of `std::collections::BinaryHeap`, which would be... difficult lol
