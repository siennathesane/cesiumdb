// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

use std::sync::Arc;

use parking_lot::{
    Mutex,
    RwLock,
};

use crate::memtable::{
    DEFAULT_MEMTABLE_SIZE_IN_BYTES,
    Memtable,
};
use crate::version::VersionManager;

pub const DEFAULT_BLOCK_SIZE: u64 = 4096;
pub const DEFAULT_TARGET_SST_SIZE: u64 = 4096;
pub const DEFAULT_NUM_MEMTABLES: u64 = 4;

/// The default set of database options.
#[derive(Clone, Copy)]
pub struct DbStorageBuilder {
    /// The size of a given disk block. It's recommended to leave the default
    /// for NVMe drives.
    pub block_size: u64,
    /// The target size of the disk files. This is a soft limit.
    pub target_sst_size: u64,
    /// The amount of tables to hold in-memory before flushing to disk.
    pub num_memtable_limit: u64,
}

impl DbStorageBuilder {
    pub fn new() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            target_sst_size: DEFAULT_TARGET_SST_SIZE,
            num_memtable_limit: DEFAULT_NUM_MEMTABLES,
        }
    }

    pub fn block_size(&mut self, block_size: u64) -> &mut Self {
        self.block_size = block_size;
        self
    }

    pub fn target_sst_size(&mut self, target_sst_size: u64) -> &mut Self {
        self.target_sst_size = target_sst_size;
        self
    }

    pub fn num_memtable_limit(&mut self, num_memtable_limit: u64) -> &mut Self {
        self.num_memtable_limit = num_memtable_limit;
        self
    }

    pub fn build(self) -> Mutex<DbStorageState> {
        Mutex::new(DbStorageState::new(self))
    }
}

impl Default for DbStorageBuilder {
    fn default() -> Self {
        DbStorageBuilder::new()
    }
}

/// Default number of LSM-tree levels (L1-L7)
pub const DEFAULT_NUM_LEVELS: usize = 7;

// TODO(@siennathesane): all universal ids (memtable, sstable, etc.) need to be
// monotonically increasing
pub struct DbStorageState {
    curr_memtable: RwLock<Arc<Memtable>>,
    frozen_memtables: Mutex<Vec<Arc<Memtable>>>,
    /// Version manager for LSM-tree level coordination
    pub version_manager: Arc<VersionManager>,
}

impl DbStorageState {
    fn new(_opts: DbStorageBuilder) -> Self {
        Self {
            // TODO(@siennathesane): add config hook here
            curr_memtable: RwLock::new(Arc::new(Memtable::new(0, DEFAULT_MEMTABLE_SIZE_IN_BYTES))),
            frozen_memtables: Mutex::new(vec![]),
            version_manager: Arc::new(VersionManager::new(DEFAULT_NUM_LEVELS)),
        }
    }

    pub fn current_memtable(&self) -> Arc<Memtable> {
        self.curr_memtable.read().clone()
    }

    /// This generates a new memtable and swaps the existing one.
    pub fn new_memtable(&mut self) {
        let next_id = self.curr_memtable.read().clone().id() + 1;
        let new_table = RwLock::new(Arc::new(Memtable::new(
            next_id,
            DEFAULT_MEMTABLE_SIZE_IN_BYTES,
        )));

        self.frozen_memtables
            .lock()
            .push(self.curr_memtable.read().clone());

        self.curr_memtable = new_table;
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use crate::{
        keypair::{
            DEFAULT_NS,
            KeyBytes,
            ValueBytes,
        },
        state::DbStorageBuilder,
    };

    #[test]
    fn test_new_memtable() {
        let state = DbStorageBuilder::default().build();

        assert!(state.lock().frozen_memtables.lock().is_empty());

        state.lock().new_memtable();
    }

    #[test]
    fn test_memtable_swap() {
        let state = DbStorageBuilder::default().build();

        let initial_id = state.lock().current_memtable().id();
        assert_eq!(initial_id, 0, "initial memtable should have id 0");

        // swap to new memtable
        state.lock().new_memtable();

        let new_id = state.lock().current_memtable().id();
        assert_eq!(new_id, 1, "new memtable should have id 1");

        // frozen memtables should contain the old one
        let frozen = state.lock().frozen_memtables.lock().clone();
        assert_eq!(frozen.len(), 1, "should have 1 frozen memtable");
        assert_eq!(frozen[0].id(), 0, "frozen memtable should have id 0");
    }

    #[test]
    fn test_multiple_memtable_swaps() {
        let state = DbStorageBuilder::default().build();

        const NUM_SWAPS: u64 = 5;

        for i in 0..NUM_SWAPS {
            let current_id = state.lock().current_memtable().id();
            assert_eq!(current_id, i);

            state.lock().new_memtable();

            let new_id = state.lock().current_memtable().id();
            assert_eq!(new_id, i + 1);
        }

        // verify all old memtables are frozen
        let frozen = state.lock().frozen_memtables.lock().clone();
        assert_eq!(frozen.len(), NUM_SWAPS as usize);

        // verify frozen memtables have correct ids
        for (idx, memtable) in frozen.iter().enumerate() {
            assert_eq!(memtable.id(), idx as u64);
        }
    }

    #[test]
    fn test_current_memtable_returns_same_instance() {
        let state = DbStorageBuilder::default().build();

        let mt1 = state.lock().current_memtable();
        let mt2 = state.lock().current_memtable();

        // should return the same Arc instance
        assert_eq!(mt1.id(), mt2.id());
    }

    #[test]
    fn test_frozen_memtables_preserve_data() {
        let state = DbStorageBuilder::default().build();

        // write data to first memtable
        let key = KeyBytes::new(DEFAULT_NS, Bytes::from("test-key"), 1000);
        let val = ValueBytes::new(DEFAULT_NS, Bytes::from("test-value"));
        {
            let current = state.lock().current_memtable();
            assert!(current.put(key.clone(), val.clone()).is_ok());
        }

        // swap to new memtable
        state.lock().new_memtable();

        // verify data is still accessible in frozen memtable
        let frozen = state.lock().frozen_memtables.lock().clone();
        assert_eq!(frozen.len(), 1);

        let retrieved = frozen[0].get(key);
        assert!(retrieved.is_some(), "data should be preserved in frozen memtable");
        assert_eq!(retrieved.unwrap().as_bytes(), val.as_bytes());
    }

    #[test]
    fn test_storage_builder_custom_config() {
        let custom_block_size = 8192;
        let custom_sst_size = 16384;
        let custom_memtable_limit = 8;

        let state = DbStorageBuilder::new()
            .block_size(custom_block_size)
            .target_sst_size(custom_sst_size)
            .num_memtable_limit(custom_memtable_limit)
            .build();

        // verify state is created successfully
        let current = state.lock().current_memtable();
        assert_eq!(current.id(), 0);
    }

    #[test]
    fn test_storage_builder_chain() {
        let mut builder = DbStorageBuilder::new();
        builder.block_size(4096).target_sst_size(8192).num_memtable_limit(6);

        let state = builder.build();
        assert_eq!(state.lock().current_memtable().id(), 0);
    }

    #[test]
    fn test_memtable_id_monotonic_increase() {
        let state = DbStorageBuilder::default().build();

        let mut prev_id = 0;
        for _ in 0..10 {
            state.lock().new_memtable();
            let current_id = state.lock().current_memtable().id();
            assert!(current_id > prev_id, "memtable ids should monotonically increase");
            prev_id = current_id;
        }
    }
}
