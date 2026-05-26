// Copyright (c) Sienna Meridian Satterwhite
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

#![no_main]

use std::collections::HashMap;

use cesiumdb::{Db, DbOptions};
use libfuzzer_sys::arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

// A single operation against the database.
#[derive(Arbitrary, Debug)]
enum Op {
    Put { ns: u8, key: Vec<u8>, value: Vec<u8> },
    Get { ns: u8, key: Vec<u8> },
    Delete { ns: u8, key: Vec<u8> },
    Sync,
    Compact,
}

/// Shadow model tracking expected state.
struct Model {
    namespaces: HashMap<u8, HashMap<Vec<u8>, Vec<u8>>>,
}

impl Model {
    fn new() -> Self {
        Self {
            namespaces: HashMap::new(),
        }
    }

    fn put(&mut self, ns: u8, key: Vec<u8>, value: Vec<u8>) {
        self.namespaces.entry(ns).or_default().insert(key, value);
    }

    fn get(&self, ns: u8, key: &[u8]) -> Option<Vec<u8>> {
        self.namespaces.get(&ns)?.get(key).cloned()
    }

    fn delete(&mut self, ns: u8, key: &[u8]) {
        if let Some(map) = self.namespaces.get_mut(&ns) {
            map.remove(key);
        }
    }
}

fn open_fuzz_db() -> (tempfile::TempDir, std::sync::Arc<Db>) {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut opts = DbOptions::default();
    opts.data_dir(tmp.path().to_path_buf())
        .memtable_size(16 * 1024)
        .max_memtables(2);
    let db = Db::open(opts).unwrap();
    (tmp, db)
}

/// Wait until compaction is idle.
fn wait_for_compaction(db: &Db) {
    for _ in 0..100 {
        if let Ok(stats) = db.compaction_stats() {
            if stats.queued_jobs == 0 && stats.in_progress_jobs == 0 {
                return;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

fuzz_target!(|ops: Vec<Op>| {
    let (_tmp, db) = open_fuzz_db();
    let mut model = Model::new();

    for op in ops {
        match op {
            Op::Put { ns, key, value } => {
                // libfuzzer can generate empty keys; skip them because
                // CesiumDB requires keys to be non-empty.
                if key.is_empty() {
                    continue;
                }
                let _ = db.put_ns(ns as u64, &key, &value);
                model.put(ns, key, value);
            }
            Op::Get { ns, key } => {
                if key.is_empty() {
                    continue;
                }
                let actual = db.get_ns(ns as u64, &key);
                let expected = model.get(ns, &key);
                match actual {
                    Ok(Some(v)) if !v.is_empty() => {
                        assert_eq!(
                            v.as_ref(),
                            expected.as_deref().unwrap_or_default(),
                            "get mismatch for ns={ns}, key={key:?}"
                        );
                    }
                    Ok(None) => {
                        assert!(
                            expected.is_none(),
                            "expected Some for ns={ns}, key={key:?}"
                        );
                    }
                    Ok(Some(v)) if v.is_empty() && expected == Some(Vec::new()) => {}
                    Err(_) => {
                        // Errors are acceptable during fuzzing (e.g. I/O issues)
                    }
                    _ => {
                        if let Some(ref exp) = expected {
                            assert_eq!(
                                actual.unwrap().unwrap().as_ref(),
                                exp.as_slice(),
                                "get mismatch for ns={ns}, key={key:?}"
                            );
                        }
                    }
                }
            }
            Op::Delete { ns, key } => {
                if key.is_empty() {
                    continue;
                }
                let _ = db.delete_ns(ns as u64, &key);
                model.delete(ns, &key);
            }
            Op::Sync => {
                let _ = db.sync();
            }
            Op::Compact => {
                let _ = db.compact();
                wait_for_compaction(&db);
            }
        }
    }

    // Final consistency check: every key in the model must match the db.
    for (ns, map) in &model.namespaces {
        for (key, expected) in map {
            let actual = db.get_ns(*ns as u64, key);
            match actual {
                Ok(Some(v)) => {
                    assert_eq!(
                        v.as_ref(),
                        expected.as_slice(),
                        "final mismatch for ns={ns}, key={key:?}"
                    );
                }
                Ok(None) => {
                    panic!("final mismatch for ns={ns}, key={key:?}: expected Some, got None");
                }
                Err(_) => {
                    // Errors are acceptable during fuzzing
                }
            }
        }
    }
});
