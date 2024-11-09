// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

use std::sync::Arc;
use bitflags::bitflags;
use crate::disk::env::Environment;
use crate::disk::page::PageNum;
use crate::disk::txn::{TxnId};

const CORE_DBS: usize = 2;
const NUM_METAS: usize = 2;

pub(crate) const RESERVE: usize = 0x10000;
pub(crate) const APPEND: usize = 0x20000;

pub(crate) type DbId = u32;

struct Db {
    env: Arc<Environment>,
    id: DbId,
}

pub(crate) struct DbMetadata {
    txn_id: TxnId,
    last_page: PageNum,
    main_db_id: DbId,
    free_db_id: DbId,
    flags: MetadataFlags,
}

pub(crate) enum MetadataFlags {}

pub(crate) struct DbState {
    flags: DbFlags,
    key_cmp: Option<()>,
    data_cmp: Option<()>,
    root_page: PageNum,
    branch_pages: u64,
    lead_pages: u64,
    overflow_pages: u64,
    entries: u64,
}

bitflags! {
    pub (crate) struct DbFlags: u8 {
        const DIRTY = 0x01;
        const STALE = 0x02;
        const NEW = 0x04;
        const VALID = 0x08;
        const USR_VALID = 0x10;
        const DUP_DATA = 0x20;
    }
}

pub (crate) struct DbHandle {
    id: DbId,
    txn_id: TxnId,
}
