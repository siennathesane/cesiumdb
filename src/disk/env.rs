// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

use std::collections::HashMap;
use std::sync::Arc;
use memmap2::MmapMut;
use parking_lot::RwLock;
use crate::disk::cursor::CursorId;
use crate::disk::mdb::{DbId, DbMetadata, DbState};
use crate::disk::txn::{TxnId, TxnState};

pub(crate) struct Environment {
    inner: Arc<RwLock<EnvironmentInner>>,
}

struct EnvironmentInner {
    meta: DbMetadata,
    map: MmapMut,
    txns: HashMap<TxnId, TxnState>,
    dbs: HashMap<DbId, DbState>,
    next_txn_id: TxnId,
    next_db_id: DbId,
    next_cursor_id: CursorId,
    flags: EnvFlags,
    max_readers: u32,
    max_dbs: u32,
    page_size: u64,
    map_size: u64,
}

pub(crate) enum EnvFlags {
    
}