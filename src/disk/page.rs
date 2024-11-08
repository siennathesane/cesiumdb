// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

use bitflags::bitflags;

/// Common header for all pages
pub(crate) struct Page {
    header: PageHeader,
    padding: u16,
    flags: PageFlags,
    bounds: PageBounds,
    ptrs: Vec<u16>, // TODO(@siennathesane): unfuck this later
}

#[repr(C)]
pub enum PageHeader {
    Number(u64),
    Next(*mut Page),
}

bitflags! {
    #[repr(C)]
    pub(crate) struct PageFlags: u16 {
        /// Branch page
        const BRANCH = 0x01;
        /// Leaf page
        const LEAF = 0x02;
        /// Overflow page
        const OVERFLOW = 0x04;
        /// Metadata page
        const METDATA = 0x08;
        /// Dirty page, also set for subpages?
        // TODO(@siennathesane): figure out what the documentation was getting at
        const DIRTY = 0x10;
        const LEAF2 = 0x20;
        const SUBPAGE = 0x40;
        /// Page was dirtied then freed, can be reused
        const LOOSE = 0x4000;
        /// Lead this page alone during spill
        const KEEP = 0x8000;
    }
}

#[repr(C)]
pub(crate) enum PageBounds {
    Regular { lower: u16, upper: u16 },
    Overflow { pages: u32 },
}
