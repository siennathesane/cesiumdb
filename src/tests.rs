// Copyright (c) Sienna Meridian Satterwhite
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

//! Unit tests for the public [`Db`] API.

use std::sync::Arc;

use crate::{
    Batch::Put,
    Db,
    DbOptions,
};

const MAX_KEYS: u64 = 10_000;

fn db_builder() -> Arc<Db> {
    Db::open(DbOptions::default()).unwrap()
}

#[test]
fn test_db_put() {
    let db = db_builder();

    // initial insert
    let mut keypair_size = 0;
    for i in 0..MAX_KEYS {
        let key = format!("key-{}", i).into_bytes();
        let val = format!("value-{}", i).into_bytes();
        keypair_size += key.len() + val.len();
        assert!(db.put(key.as_ref(), val.as_ref()).is_ok());
    }

    {
        let guard = db.inner.state.lock();
        assert!(
            guard.current_memtable().size() > keypair_size as u64,
            "the memtable must be bigger than the keypair size to ensure the keys are actually stored"
        );
    }

    // re-insert the same keys but with new versions
    for i in 0..MAX_KEYS {
        let key = format!("key-{}", i).into_bytes();
        let val = format!("value-{}", i).into_bytes();
        keypair_size += key.len() + val.len();
        assert!(db.put(key.as_ref(), val.as_ref()).is_ok());
    }

    {
        let guard = db.inner.state.lock();
        assert!(
            guard.current_memtable().size() > (keypair_size * 2) as u64,
            "the memtable must be at least twice as big as before with the new versions"
        );
    }
}

#[test]
fn db_put_batch() {
    let db = db_builder();

    let mut keypair_size = 0;
    for batch_size in [1, 10, 100].iter() {
        let mut batch = Vec::with_capacity(*batch_size);

        for i in 0..(*batch_size * 100) {
            let key = format!("key-{}", i).into_bytes();
            let val = format!("value-{}", i).into_bytes();
            keypair_size += key.len() + val.len();

            let op = Put(key, val.clone(), db.time());
            batch.push(op)
        }

        assert!(db.batch(&batch).is_ok());

        {
            let guard = db.inner.state.lock();
            assert!(
                guard.current_memtable().size() > keypair_size as u64,
                "the memtable must be bigger than the keypair size to ensure the keys are actually stored"
            );
        }
    }
}

#[test]
fn test_db_get() {
    let db = db_builder();

    // test get on empty db
    let result = db.get(b"nonexistent");
    assert!(result.is_ok());
    assert!(
        result.unwrap().is_none(),
        "get on empty db should return None"
    );

    // insert and retrieve
    let key = b"test-key";
    let val = b"test-value";
    assert!(db.put(key, val).is_ok());

    let result = db.get(key);
    assert!(result.is_ok());
    let retrieved = result.unwrap();
    assert!(
        retrieved.is_some(),
        "get should return Some for existing key"
    );
    assert_eq!(&retrieved.unwrap()[..], val, "retrieved value should match");

    // test get on different key
    let result = db.get(b"different-key");
    assert!(result.is_ok());
    assert!(
        result.unwrap().is_none(),
        "get on non-existent key should return None"
    );
}

#[test]
fn test_db_get_latest_version() {
    let db = db_builder();

    let key = b"versioned-key";
    let val1 = b"value-1";
    let val2 = b"value-2";
    let val3 = b"value-3";

    // insert multiple versions
    assert!(db.put(key, val1).is_ok());
    assert!(db.put(key, val2).is_ok());
    assert!(db.put(key, val3).is_ok());

    // get should return the latest version
    let result = db.get(key);
    assert!(result.is_ok());
    let retrieved = result.unwrap();
    assert!(retrieved.is_some());
    assert_eq!(
        &retrieved.unwrap()[..],
        val3,
        "get should return the latest value"
    );
}

#[test]
fn test_db_delete() {
    let db = db_builder();

    let key = b"key-to-delete";
    let val = b"value";

    // insert key
    assert!(db.put(key, val).is_ok());

    // verify it exists
    let result = db.get(key);
    assert!(result.is_ok());
    assert!(result.unwrap().is_some());

    // delete the key
    assert!(db.delete(key).is_ok());

    // verify the key no longer exists (tombstone filters it out)
    let result = db.get(key);
    assert!(result.is_ok());
    assert!(result.unwrap().is_none(), "deleted key should return None");
}

#[test]
fn test_db_put_ns() {
    let db = db_builder();

    let ns1: u64 = 1;
    let ns2: u64 = 2;
    let key = b"same-key";
    let val1 = b"value-in-ns1";
    let val2 = b"value-in-ns2";

    // put same key in different namespaces
    assert!(db.put_ns(ns1, key, val1).is_ok());
    assert!(db.put_ns(ns2, key, val2).is_ok());

    // retrieve from each namespace
    let result1 = db.get_ns(ns1, key);
    assert!(result1.is_ok());
    let value1 = result1.unwrap();
    assert!(value1.is_some());
    assert_eq!(&value1.unwrap()[..], val1);

    let result2 = db.get_ns(ns2, key);
    assert!(result2.is_ok());
    let value2 = result2.unwrap();
    assert!(value2.is_some());
    assert_eq!(&value2.unwrap()[..], val2);
}

#[test]
fn test_db_get_ns() {
    let db = db_builder();

    let ns: u64 = 42;
    let key = b"namespaced-key";
    let val = b"namespaced-value";

    // get on empty namespace
    let result = db.get_ns(ns, key);
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());

    // insert into namespace
    assert!(db.put_ns(ns, key, val).is_ok());

    // retrieve from namespace
    let result = db.get_ns(ns, key);
    assert!(result.is_ok());
    let retrieved = result.unwrap();
    assert!(retrieved.is_some());
    assert_eq!(&retrieved.unwrap()[..], val);

    // verify key doesn't exist in default namespace
    let result = db.get(key);
    assert!(result.is_ok());
    assert!(
        result.unwrap().is_none(),
        "key should not exist in default namespace"
    );
}

#[test]
fn test_db_delete_ns() {
    let db = db_builder();

    let ns: u64 = 10;
    let key = b"key-to-delete";
    let val = b"value";

    // insert into namespace
    assert!(db.put_ns(ns, key, val).is_ok());

    // verify it exists
    let result = db.get_ns(ns, key);
    assert!(result.is_ok());
    assert!(result.unwrap().is_some());

    // delete from namespace
    assert!(db.delete_ns(ns, key).is_ok());

    // verify the key no longer exists (tombstone filters it out)
    let result = db.get_ns(ns, key);
    assert!(result.is_ok());
    assert!(
        result.unwrap().is_none(),
        "deleted key in namespace should return None"
    );
}

#[test]
fn test_db_options_default() {
    let opts = DbOptions::default();
    let db = Db::open(opts).unwrap();

    // basic operation to ensure default options work
    assert!(db.put(b"test", b"value").is_ok());
    let result = db.get(b"test");
    assert!(result.is_ok());
    assert!(result.unwrap().is_some());
}

#[test]
fn test_db_time() {
    let db = db_builder();

    let time1 = db.time();
    let time2 = db.time();

    // time should be monotonically increasing
    assert!(
        time2 >= time1,
        "clock should return monotonically increasing values"
    );
}

#[test]
fn test_db_batch_mixed_operations() {
    use crate::Batch::PutNs;

    let db = db_builder();

    let ns: u64 = 5;
    let batch = vec![
        Put(b"key1".to_vec(), b"val1".to_vec(), db.time()),
        PutNs(ns, b"key2".to_vec(), b"val2".to_vec(), db.time()),
        Put(b"key3".to_vec(), b"val3".to_vec(), db.time()),
    ];

    assert!(db.batch(&batch).is_ok());

    // verify all operations succeeded
    assert!(db.get(b"key1").unwrap().is_some());
    assert!(db.get_ns(ns, b"key2").unwrap().is_some());
    assert!(db.get(b"key3").unwrap().is_some());
}

#[test]
fn test_db_empty_key() {
    let db = db_builder();

    let key = b"";
    let val = b"empty-key-value";

    assert!(db.put(key, val).is_ok());
    let result = db.get(key);
    assert!(result.is_ok());
    assert!(result.unwrap().is_some());
}

#[test]
fn test_db_empty_value() {
    let db = db_builder();

    let key = b"key-with-empty-value";
    let val = b"";

    assert!(db.put(key, val).is_ok());
    let result = db.get(key);
    assert!(result.is_ok());
    let retrieved = result.unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().len(), 0);
}

#[test]
fn test_db_large_key_value() {
    let db = db_builder();

    let key = vec![b'k'; 1000];
    let val = vec![b'v'; 10000];

    assert!(db.put(&key, &val).is_ok());
    let result = db.get(&key);
    assert!(result.is_ok());
    let retrieved = result.unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().len(), val.len());
}

// ===== Scan API Tests =====

#[test]
fn test_scan_empty_db() {
    use std::ops::Bound;
    let db = db_builder();
    let results: Vec<_> = db
        .scan(Bound::Unbounded, Bound::Unbounded)
        .unwrap()
        .collect();
    assert_eq!(results.len(), 0);
}

#[test]
fn test_scan_single_memtable() {
    use std::ops::Bound;
    let db = db_builder();

    db.put(b"a", b"1").unwrap();
    db.put(b"b", b"2").unwrap();
    db.put(b"c", b"3").unwrap();

    let results: Vec<_> = db
        .scan(Bound::Unbounded, Bound::Unbounded)
        .unwrap()
        .collect();
    assert_eq!(results.len(), 3);
    assert_eq!(&results[0].0[..], b"a");
    assert_eq!(&results[0].1[..], b"1");
    assert_eq!(&results[1].0[..], b"b");
    assert_eq!(&results[1].1[..], b"2");
    assert_eq!(&results[2].0[..], b"c");
    assert_eq!(&results[2].1[..], b"3");
}

#[test]
fn test_scan_with_tombstones() {
    use std::ops::Bound;
    let db = db_builder();

    db.put(b"a", b"1").unwrap();
    db.put(b"b", b"2").unwrap();
    db.put(b"c", b"3").unwrap();
    db.delete(b"b").unwrap();

    let results: Vec<_> = db
        .scan(Bound::Unbounded, Bound::Unbounded)
        .unwrap()
        .collect();
    assert_eq!(results.len(), 2, "Should filter out tombstone for 'b'");
    assert_eq!(&results[0].0[..], b"a");
    assert_eq!(&results[1].0[..], b"c");
}

#[test]
fn test_scan_unbounded() {
    use std::ops::Bound;
    let db = db_builder();

    for i in 0..10 {
        let key = format!("key-{:02}", i);
        let val = format!("val-{:02}", i);
        db.put(key.as_bytes(), val.as_bytes()).unwrap();
    }

    let results: Vec<_> = db
        .scan(Bound::Unbounded, Bound::Unbounded)
        .unwrap()
        .collect();
    assert_eq!(results.len(), 10);
}

#[test]
fn test_scan_bounded() {
    use std::ops::Bound;
    let db = db_builder();

    db.put(b"a", b"1").unwrap();
    db.put(b"b", b"2").unwrap();
    db.put(b"c", b"3").unwrap();
    db.put(b"d", b"4").unwrap();
    db.put(b"e", b"5").unwrap();

    // Scan [b, d) should return b and c
    let results: Vec<_> = db
        .scan(Bound::Included(b"b"), Bound::Excluded(b"d"))
        .unwrap()
        .collect();

    assert_eq!(results.len(), 2);
    assert_eq!(&results[0].0[..], b"b");
    assert_eq!(&results[0].1[..], b"2");
    assert_eq!(&results[1].0[..], b"c");
    assert_eq!(&results[1].1[..], b"3");
}

#[test]
fn test_scan_namespace_isolation() {
    use std::ops::Bound;
    let db = db_builder();

    db.put_ns(1, b"key", b"ns1-value").unwrap();
    db.put_ns(2, b"key", b"ns2-value").unwrap();
    db.put_ns(1, b"key2", b"ns1-value2").unwrap();

    let results_ns1: Vec<_> = db
        .scan_ns(1, Bound::Unbounded, Bound::Unbounded)
        .unwrap()
        .collect();
    let results_ns2: Vec<_> = db
        .scan_ns(2, Bound::Unbounded, Bound::Unbounded)
        .unwrap()
        .collect();

    assert_eq!(results_ns1.len(), 2);
    assert_eq!(results_ns2.len(), 1);
    assert_eq!(&results_ns2[0].1[..], b"ns2-value");
}

#[test]
fn test_scan_across_frozen_memtables() {
    use std::ops::Bound;
    let db = db_builder();

    // Write some data
    db.put(b"a", b"1").unwrap();
    db.put(b"b", b"2").unwrap();

    // Force sync to freeze memtable
    db.sync().unwrap();

    // Write more data to new memtable
    db.put(b"c", b"3").unwrap();
    db.put(b"d", b"4").unwrap();

    let results: Vec<_> = db
        .scan(Bound::Unbounded, Bound::Unbounded)
        .unwrap()
        .collect();
    assert_eq!(
        results.len(),
        4,
        "Should scan across both frozen and current memtable"
    );
    assert_eq!(&results[0].0[..], b"a");
    assert_eq!(&results[1].0[..], b"b");
    assert_eq!(&results[2].0[..], b"c");
    assert_eq!(&results[3].0[..], b"d");
}

#[test]
fn test_scan_mvcc_ordering() {
    // Simplified test to check iteration order
    use std::ops::Bound;
    let db = db_builder();

    // Write in explicit order to track timestamps
    std::thread::sleep(std::time::Duration::from_millis(1));
    db.put(b"key1", b"FIRST").unwrap();

    std::thread::sleep(std::time::Duration::from_millis(1));
    db.put(b"key1", b"SECOND").unwrap();

    std::thread::sleep(std::time::Duration::from_millis(1));
    db.put(b"key1", b"THIRD_NEWEST").unwrap();

    // Scan should return only the newest version
    let results: Vec<_> = db
        .scan(Bound::Unbounded, Bound::Unbounded)
        .unwrap()
        .collect();
    assert_eq!(results.len(), 1, "Should have 1 unique key");
    assert_eq!(
        &results[0].1[..],
        b"THIRD_NEWEST",
        "Should return the newest version"
    );
}

#[test]
fn test_scan_with_mvcc() {
    use std::ops::Bound;
    let db = db_builder();

    // Write initial values
    db.put(b"key1", b"v1").unwrap();
    db.put(b"key2", b"v2").unwrap();

    // Update values (creates new versions)
    db.put(b"key1", b"v1-updated").unwrap();

    let results: Vec<_> = db
        .scan(Bound::Unbounded, Bound::Unbounded)
        .unwrap()
        .collect();
    assert_eq!(results.len(), 2);

    // Should see newest version
    assert_eq!(&results[0].0[..], b"key1");
    assert_eq!(&results[0].1[..], b"v1-updated");
    assert_eq!(&results[1].0[..], b"key2");
    assert_eq!(&results[1].1[..], b"v2");
}
