// Copyright (c) Sienna Meridian Satterwhite
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

//! Namespace isolation example.
//!
//! Namespaces allow logical grouping of keys within a single database.
//! Keys in different namespaces are completely isolated.
//!
//! Run with:
//! ```bash
//! cargo run --example namespaces
//! ```

use std::path::PathBuf;

use cesiumdb::{
    Db,
    DbOptions,
};

fn main() {
    let mut opts = DbOptions::default();
    opts.data_dir(PathBuf::from("/tmp/cesiumdb_namespaces_example"));

    let db = Db::open(opts).unwrap();

    // Write the same key to two different namespaces
    db.put_ns(1, b"user:42", b"alice").unwrap();
    db.put_ns(2, b"user:42", b"bob").unwrap();

    // Each namespace sees only its own value
    let ns1_value = db.get_ns(1, b"user:42").unwrap();
    let ns2_value = db.get_ns(2, b"user:42").unwrap();

    assert_eq!(ns1_value.as_deref(), Some(b"alice".as_slice()));
    assert_eq!(ns2_value.as_deref(), Some(b"bob".as_slice()));

    println!("ns1 user:42 = {:?}", ns1_value);
    println!("ns2 user:42 = {:?}", ns2_value);

    // Delete from one namespace does not affect the other
    db.delete_ns(1, b"user:42").unwrap();
    assert!(db.get_ns(1, b"user:42").unwrap().is_none());
    assert!(db.get_ns(2, b"user:42").unwrap().is_some());

    println!("deleted from ns1; ns2 still has the key");

    db.sync().unwrap();
    db.close().unwrap();
    println!("done");
}
