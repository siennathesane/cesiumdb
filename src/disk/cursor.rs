// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

use crate::disk::mdb::DbId;
use crate::disk::page::PageNum;
use crate::disk::txn::TxnId;

pub(crate) type CursorId = u64;

pub (crate) struct CursorState {
    txn_id: TxnId,
    db_id: DbId,
    
}

pub(crate) struct CursorStackEntry {
    page: PageNum,
    index: u16,
}