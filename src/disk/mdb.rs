// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

const CORE_DBS: usize = 2;
const NUM_METAS: usize = 2;

pub(crate) const RESERVE: usize = 0x10000;
pub(crate) const APPEND: usize = 0x20000;

struct MapDB {
    padding: u32,
    flags: u16,
    /// The depth of the tree
    depth: u16,
    /// The number of branch pages
    branch_pages: u64,
    /// The number of leaf pages
    leaf_pages: u64,
    /// The number of overflow pages
    overflow_pages: u64,
    /// The number of entries in the tree
    entries: u64,
    /// The root page of this tree
    root: u64,
}

struct MapDBMeta {
    magic: u32,
    version: u32,
    address: u64, /* TODO(@siennathesane): address for fixed mapping? prolly a weird memory
                   * ownership workaround */
    map_size: usize,
    dbs: [MapDB; CORE_DBS],
    last_page: usize,
    last_txnid: usize, // volatile?
}

impl MapDBMeta {
    fn page_size(&self) -> u32 {
        self.dbs[0].padding
    }

    fn flags(&self) -> u16 {
        self.dbs[0].flags
    }
}
