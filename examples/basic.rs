// Copyright (c) Sienna Meridian Satterwhite
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

//! Basic CesiumDB usage example.
//!
//! Run with:
//! ```bash
//! cargo run --example basic
//! ```

use std::path::PathBuf;

use cesiumdb::{
    Db,
    DbOptions,
};

fn main() {
    let mut opts = DbOptions::default();
    opts.data_dir(PathBuf::from("/tmp/cesiumdb_basic_example"));

    let db = Db::open(opts).unwrap();

    // Simple put/get
    db.put(b"hello", b"world").unwrap();
    let value = db.get(b"hello").unwrap();
    assert_eq!(value.as_deref(), Some(b"world".as_slice()));
    println!("put(hello, world) => get(hello) = {:?}", value);

    // Overwrite
    db.put(b"hello", b"cesium").unwrap();
    let value = db.get(b"hello").unwrap();
    assert_eq!(value.as_deref(), Some(b"cesium".as_slice()));
    println!("put(hello, cesium) => get(hello) = {:?}", value);

    // Delete
    db.delete(b"hello").unwrap();
    let value = db.get(b"hello").unwrap();
    assert!(value.is_none());
    println!("delete(hello) => get(hello) = {:?}", value);

    // Sync to disk
    db.sync().unwrap();
    println!("synced to disk");

    // Clean up
    db.close().unwrap();
    println!("closed successfully");
}
