use std::sync::Arc;

use crate::memtable::Memtable;

pub const DEFAULT_BLOCK_SIZE: usize = 4096;
pub const DEFAULT_TARGET_SST_SIZE: usize = 4096;
pub const DEFAULT_NUM_MEMTABLES: usize = 4;

/// The default set of database options.
#[derive(Clone, Copy)]
pub struct DbStorageBuilder {
    /// The size of a given disk block. It's recommended to leave the default
    /// for NVMe drives.
    pub block_size: usize,
    /// The target size of the disk files. This is a soft limit.
    pub target_sst_size: usize,
    /// The amount of tables to hold in-memory before flushing to disk.
    pub num_memtable_limit: usize,
}

impl DbStorageBuilder {
    pub fn new() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            target_sst_size: DEFAULT_TARGET_SST_SIZE,
            num_memtable_limit: DEFAULT_NUM_MEMTABLES,
        }
    }

    pub fn block_size(&mut self, block_size: usize) -> &mut Self {
        self.block_size = block_size;
        self
    }

    pub fn target_sst_size(&mut self, target_sst_size: usize) -> &mut Self {
        self.target_sst_size = target_sst_size;
        self
    }

    pub fn num_memtable_limit(&mut self, num_memtable_limit: usize) -> &mut Self {
        self.num_memtable_limit = num_memtable_limit;
        self
    }

    pub fn build(self) -> Arc<DbStorageState> {
        Arc::new(DbStorageState::new(self))
    }
}

impl Default for DbStorageBuilder {
    fn default() -> Self {
        DbStorageBuilder::new()
    }
}

pub struct DbStorageState {
    curr_memtable: Arc<Memtable>,
    _frozen_memtables: Vec<Arc<Memtable>>,
}

impl DbStorageState {
    fn new(_opts: DbStorageBuilder) -> Self {
        Self {
            curr_memtable: Arc::new(Memtable::new(0)),
            _frozen_memtables: vec![],
        }
    }

    pub fn current_memtable(&self) -> Arc<Memtable> {
        self.curr_memtable.clone()
    }
}
