use std::sync::Arc;

use parking_lot::{
    Mutex,
    RwLock,
};

use crate::memtable::{
    Memtable,
    DEFAULT_MEMTABLE_SIZE_IN_BYTES,
};

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

pub struct DbStorageState {
    curr_memtable: RwLock<Arc<Memtable>>,
    frozen_memtables: Mutex<Vec<Arc<Memtable>>>,
}

impl DbStorageState {
    fn new(_opts: DbStorageBuilder) -> Self {
        Self {
            // TODO(@siennathesane): add config hook here
            curr_memtable: RwLock::new(Arc::new(Memtable::new(0, DEFAULT_MEMTABLE_SIZE_IN_BYTES))),
            frozen_memtables: Mutex::new(vec![]),
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
    use crate::state::DbStorageBuilder;

    #[test]
    fn test_new_memtable() {
        let state = DbStorageBuilder::default().build();
        
        assert!(state.lock().frozen_memtables.lock().is_empty());
        
        state.lock().new_memtable();
    }
}