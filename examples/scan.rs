// Copyright (c) Sienna Meridian Satterwhite
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

//! Range scan example.
//!
//! Demonstrates iterating over a range of keys.
//!
//! Run with:
//! ```bash
//! cargo run --example scan
//! ```

use std::{
    ops::Bound,
    path::PathBuf,
};

use cesiumdb::{
    Db,
    DbOptions,
};

fn main() {
    let mut opts = DbOptions::default();
    opts.data_dir(PathBuf::from("/tmp/cesiumdb_scan_example"));

    let db = Db::open(opts).unwrap();

    // Insert some ordered keys
    for i in 0..10 {
        let key = format!("key_{:03}", i);
        let value = format!("value_{}", i);
        db.put(key.as_bytes(), value.as_bytes()).unwrap();
    }

    // Scan a range: key_003 <= key < key_007
    let start = b"key_003".to_vec();
    let end = b"key_007".to_vec();

    println!("scanning key_003 .. key_007:");
    for (key, value) in db.scan(Bound::Included(&start), Bound::Excluded(&end)).unwrap() {
        println!(
            "  {} = {}",
            String::from_utf8_lossy(&key),
            String::from_utf8_lossy(&value)
        );
    }

    db.sync().unwrap();
    db.close().unwrap();
    println!("done");
}
