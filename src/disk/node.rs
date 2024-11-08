// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

use std::mem::offset_of;

use crate::disk::{
    mdb::{
        APPEND,
        RESERVE,
    },
    val::Value,
};

pub(crate) const BIG_DATA: usize = 0x01;
pub(crate) const SUB_DATA: usize = 0x02;
pub(crate) const DUP_DATA: usize = 0x04;

// TODO(@siennathesane): implement node size macro
const NODE_SIZE: usize = offset_of!(Node<&[u8]>, data);

const fn index_size(key_size: usize) -> usize {
    NODE_SIZE + key_size
}

#[repr(C)]
struct Node<T: AsRef<[u8]>> {
    // TODO(@siennathesane): make this a u32 once i have a better understanding of the layout
    lo: u16,
    hi: u16,
    flags: u16,
    key_size: u16,
    data: Value<T>, // TODO(@siennathesane): figure out what data type this should be
}

impl<T: AsRef<[u8]>> Node<T> {
    fn add_flags() -> usize {
        DUP_DATA | SUB_DATA | RESERVE | APPEND
    }

    #[inline]
    fn leaf_size(&self, key: &Value<T>, val: &Value<T>) -> usize {
        NODE_SIZE + key.size() as usize + val.size() as usize
    }

    #[inline]
    fn page_num(&self) {}
}
