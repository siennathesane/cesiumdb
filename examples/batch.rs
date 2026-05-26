// Copyright (c) Sienna Meridian Satterwhite
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

//! Batch write example.
//!
//! Batching multiple operations into a single call is significantly
//! faster than issuing individual puts.
//!
//! Run with:
//! ```bash
//! cargo run --example batch
//! ```

use std::path::PathBuf;

use cesiumdb::{
    Batch::{
        Put,
        PutNs,
    },
    Db,
    DbOptions,
    hlc::{
        HLC,
        HybridLogicalClock,
    },
};

fn main() {
    let mut opts = DbOptions::default();
    opts.data_dir(PathBuf::from("/tmp/cesiumdb_batch_example"));

    let db = Db::open(opts).unwrap();
    let clock = HybridLogicalClock::new();

    // Batch multiple operations atomically
    db.batch::<&[u8], &[u8]>(&[
        Put(b"key1", b"value1", clock.time()),
        Put(b"key2", b"value2", clock.time()),
        Put(b"key3", b"value3", clock.time()),
    ])
    .unwrap();

    // Verify
    assert_eq!(
        db.get(b"key1").unwrap().as_deref(),
        Some(b"value1".as_slice())
    );
    assert_eq!(
        db.get(b"key2").unwrap().as_deref(),
        Some(b"value2".as_slice())
    );
    assert_eq!(
        db.get(b"key3").unwrap().as_deref(),
        Some(b"value3".as_slice())
    );

    println!("batch write succeeded");

    // Batch with namespaces
    db.batch::<&[u8], &[u8]>(&[
        PutNs(1, b"config", b"debug", clock.time()),
        PutNs(2, b"config", b"release", clock.time()),
        PutNs(1, b"version", b"1.0.0", clock.time()),
    ])
    .unwrap();

    println!("ns1 config = {:?}", db.get_ns(1, b"config").unwrap());
    println!("ns2 config = {:?}", db.get_ns(2, b"config").unwrap());

    db.sync().unwrap();
    db.close().unwrap();
    println!("done");
}
