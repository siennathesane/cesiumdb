// Copyright (c) Sienna Meridian Satterwhite <sienna@r3t.io>
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

//! Property-based tests for the CesiumDB public interface.
//!
//! Uses model-based testing: execute random sequences of operations against
//! both the real Db and a simple in-memory HashMap, then verify they agree.

use std::{
    collections::HashMap,
    sync::Arc,
};

use cesiumdb::{
    Db,
    DbOptions,
};
use proptest::prelude::*;
use tempfile::TempDir;

/// Default proptest configuration for this test suite.
///
/// Uses fewer cases than the default (32 vs 256) because each case creates
/// a fresh database and may trigger background compaction, which makes
/// individual cases relatively expensive.
fn db_proptest_config() -> ProptestConfig {
    ProptestConfig {
        cases: 32,
        ..ProptestConfig::default()
    }
}

/// An operation in our state-machine model.
#[derive(Debug, Clone)]
enum Op {
    Put {
        ns: u64,
        key: Vec<u8>,
        value: Vec<u8>,
    },
    Get {
        ns: u64,
        key: Vec<u8>,
    },
    Delete {
        ns: u64,
        key: Vec<u8>,
    },
    Sync,
    Compact,
}

/// In-memory model of expected Db state.
struct Model {
    namespaces: HashMap<u64, HashMap<Vec<u8>, Vec<u8>>>,
}

impl Model {
    fn new() -> Self {
        Self {
            namespaces: HashMap::new(),
        }
    }

    fn put(&mut self, ns: u64, key: Vec<u8>, value: Vec<u8>) {
        self.namespaces.entry(ns).or_default().insert(key, value);
    }

    fn get(&self, ns: u64, key: &[u8]) -> Option<Vec<u8>> {
        self.namespaces.get(&ns)?.get(key).cloned()
    }

    fn delete(&mut self, ns: u64, key: &[u8]) {
        if let Some(map) = self.namespaces.get_mut(&ns) {
            map.remove(key);
        }
    }
}

fn arb_key() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 1usize..64)
}

fn arb_value() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 0usize..4096)
}

fn arb_ns() -> impl Strategy<Value = u64> {
    0u64..8
}

fn arb_op() -> impl Strategy<Value = Op> {
    prop_oneof![
        (arb_ns(), arb_key(), arb_value()).prop_map(|(ns, key, value)| Op::Put { ns, key, value }),
        (arb_ns(), arb_key()).prop_map(|(ns, key)| Op::Get { ns, key }),
        (arb_ns(), arb_key()).prop_map(|(ns, key)| Op::Delete { ns, key }),
        Just(Op::Sync),
        Just(Op::Compact),
    ]
}

fn open_test_db() -> (TempDir, Arc<Db>) {
    let tmp = TempDir::new().unwrap();
    let mut opts = DbOptions::default();
    opts.data_dir(tmp.path().to_path_buf())
        .memtable_size(64 * 1024)
        .max_memtables(4);
    let db = Db::open(opts).unwrap();
    (tmp, db)
}

/// Wait until the compaction queue is idle.
fn wait_for_compaction(db: &Db) {
    for _ in 0..200 {
        if let Ok(stats) = db.compaction_stats() {
            if stats.queued_jobs == 0 && stats.in_progress_jobs == 0 {
                return;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

proptest! {
    #![proptest_config(db_proptest_config())]
    #[test]
    fn prop_crud_matches_model(ops in prop::collection::vec(arb_op(), 1usize..100)) {
        let (_tmp, db) = open_test_db();
        let mut model = Model::new();

        for op in ops {
            match op {
                Op::Put { ns, key, value } => {
                    db.put_ns(ns, &key, &value).unwrap();
                    model.put(ns, key, value);
                }
                Op::Get { ns, key } => {
                    let actual = db.get_ns(ns, &key).unwrap();
                    let expected = model.get(ns, &key);
                    prop_assert_eq!(
                        actual.as_ref().map(|b| b.to_vec()),
                        expected,
                        "get mismatch for ns={}, key={:?}", ns, key
                    );
                }
                Op::Delete { ns, key } => {
                    db.delete_ns(ns, &key).unwrap();
                    model.delete(ns, &key);
                }
                Op::Sync => {
                    db.sync().unwrap();
                }

                Op::Compact => {
                    let _ = db.compact();
                    wait_for_compaction(&db);
                }
            }
        }

        // Final verification: every key in the model must match the db
        for (ns, map) in &model.namespaces {
            for (key, expected) in map {
                let actual = db.get_ns(*ns, key).unwrap();
                prop_assert_eq!(
                    actual.as_ref().map(|b| b.to_vec()),
                    Some(expected.clone()),
                    "final mismatch for ns={}, key={:?}", ns, key
                );
            }
        }
    }
}

proptest! {
    #![proptest_config(db_proptest_config())]
    /// Overwrites always return the latest value.
    #[test]
    fn prop_overwrite_latest((key, values) in (arb_key(), prop::collection::vec(arb_value(), 2usize..20))) {
        let (_tmp, db) = open_test_db();
        for value in &values {
            db.put(&key, value).unwrap();
        }
        let actual = db.get(&key).unwrap();
        prop_assert_eq!(
            actual.as_ref().map(|b| b.to_vec()),
            Some(values.last().unwrap().clone())
        );
    }
}

proptest! {
    #![proptest_config(db_proptest_config())]
    /// Deleted keys stay deleted.
    #[test]
    fn prop_delete_stays_deleted((key, value) in (arb_key(), arb_value())) {
        let (_tmp, db) = open_test_db();
        db.put(&key, &value).unwrap();
        db.delete(&key).unwrap();
        db.sync().unwrap();
        let actual = db.get(&key).unwrap();
        prop_assert!(actual.is_none(), "key should be deleted after sync");
    }
}

proptest! {
    #![proptest_config(db_proptest_config())]
    /// Namespace isolation: writes to one ns are invisible to others.
    #[test]
    fn prop_namespace_isolation(
        (ns1, ns2, key, value) in (arb_ns(), arb_ns(), arb_key(), arb_value())
    ) {
        prop_assume!(ns1 != ns2);
        let (_tmp, db) = open_test_db();
        db.put_ns(ns1, &key, &value).unwrap();
        let actual = db.get_ns(ns2, &key).unwrap();
        prop_assert!(actual.is_none(), "ns2 should not see ns1's data");
    }
}

proptest! {
    #![proptest_config(db_proptest_config())]
    /// Sync + reopen preserves all data.
    #[test]
    fn prop_sync_recover(
        ops in prop::collection::vec(
            (arb_ns(), arb_key(), arb_value()).prop_map(|(ns, key, value)| Op::Put { ns, key, value }),
            1usize..50
        )
    ) {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().to_path_buf();
        let mut model = Model::new();

        {
            let mut opts = DbOptions::default();
            opts.data_dir(path.clone())
                .memtable_size(64 * 1024)
                .max_memtables(4);
            let db = Db::open(opts).unwrap();
            for op in &ops {
                if let Op::Put { ns, key, value } = op {
                    db.put_ns(*ns, key, value).unwrap();
                    model.put(*ns, key.clone(), value.clone());
                }
            }
            db.sync().unwrap();
        }

        {
            let mut opts = DbOptions::default();
            opts.data_dir(path)
                .memtable_size(64 * 1024)
                .max_memtables(4);
            let db = Db::open(opts).unwrap();
            for (ns, map) in &model.namespaces {
                for (key, expected) in map {
                    let actual = db.get_ns(*ns, key).unwrap();
                    prop_assert_eq!(
                        actual.as_ref().map(|b| b.to_vec()),
                        Some(expected.clone()),
                        "recover mismatch for ns={}, key={:?}", ns, key
                    );
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(db_proptest_config())]
    /// Compaction must not lose data.
    #[test]
    fn prop_compaction_preserves_data(
        ops in prop::collection::vec(
            (arb_ns(), arb_key(), arb_value()).prop_map(|(ns, key, value)| Op::Put { ns, key, value }),
            1usize..100
        )
    ) {
        let (_tmp, db) = open_test_db();
        let mut model = Model::new();

        for op in &ops {
            if let Op::Put { ns, key, value } = op {
                db.put_ns(*ns, key, value).unwrap();
                model.put(*ns, key.clone(), value.clone());
            }
        }

        db.compact().unwrap();
        wait_for_compaction(&db);

        for (ns, map) in &model.namespaces {
            for (key, expected) in map {
                let actual = db.get_ns(*ns, key).unwrap();
                prop_assert_eq!(
                    actual.as_ref().map(|b| b.to_vec()),
                    Some(expected.clone()),
                    "compaction mismatch for ns={}, key={:?}", ns, key
                );
            }
        }
    }
}
