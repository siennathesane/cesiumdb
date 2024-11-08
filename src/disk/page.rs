// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

use std::ptr::{
    null_mut,
    NonNull,
};

use bitflags::bitflags;

use crate::disk::page::PageBounds::{
    Overflow,
    Regular,
};

pub(crate) const PAGE_HEADER_SIZE: usize = std::mem::offset_of!(Page, ptrs);
pub(crate) const PAGE_BASE: usize = if cfg!(debug_assertions) {
    PAGE_HEADER_SIZE
} else {
    0
};

/// Common header for all pages
#[repr(C)]
pub(crate) struct Page {
    header: PageHeader,
    padding: u16,
    flags: PageFlags,
    bounds: PageBounds,
    ptrs: Vec<u16>, // TODO(@siennathesane): unfuck this later
}

impl Page {
    #[inline]
    fn num_keys(&self) -> usize {
        (self.bounds.lower() as usize - (PAGE_HEADER_SIZE - PAGE_BASE)) >> 1
    }

    #[inline]
    fn size_left(&self) -> u16 {
        match self.bounds {
            | Regular { upper, lower } => upper - lower,
            | Overflow { .. } => todo!(),
        }
    }

    #[inline]
    fn fill_percentage(&self, page_size: usize) -> u32 {
        (1000 * (page_size - PAGE_HEADER_SIZE - self.size_left() as usize) /
            (page_size - PAGE_HEADER_SIZE)) as u32
    }
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
        /// Leave this page alone during spill
        const KEEP = 0x8000;
    }
}

impl PageFlags {
    fn is_leaf(&self) -> bool {
        self.contains(PageFlags::LEAF)
    }

    fn is_leaf2(&self) -> bool {
        self.contains(PageFlags::LEAF2)
    }

    fn is_branch(&self) -> bool {
        self.contains(PageFlags::BRANCH)
    }

    fn is_overflow(&self) -> bool {
        self.contains(PageFlags::OVERFLOW)
    }

    fn is_subpage(&self) -> bool {
        self.contains(PageFlags::SUBPAGE)
    }
}

#[repr(C)]
pub(crate) enum PageBounds {
    Regular { lower: u16, upper: u16 },
    Overflow { pages: u32 },
}

// nb(@siennathesane): this is a "safer" abstraction on top of the
// `NEXT_LOOSE_PAGE` macro from the lmdb source. it retains the exact same
// memory layout while trying to maintain the same performance. generically,
// it's still unsafe, but it's much easier to understand and reason about.
// unfortunately this is one of the few places where we have to use unsafe code
// to retain the same performance characteristics.
pub(crate) struct LoosePage {
    page: NonNull<Page>,
    next: Option<NonNull<Page>>,
}

impl LoosePage {
    /// Create a new loose page wrapper, ensuring it's not null.
    #[inline(always)]
    pub(crate) fn new(page: NonNull<Page>) -> Self {
        Self {
            page,
            next: unsafe { Self::read_next_ptr(page.as_ptr()) },
        }
    }

    /// Get the next loose page pointer from the given page.
    #[inline(always)]
    pub(crate) fn page(&self) -> &Page {
        unsafe { self.page.as_ref() }
    }

    /// Get the next loose page pointer from the given page.
    #[inline(always)]
    pub(crate) fn page_mut(&mut self) -> &mut Page {
        unsafe { self.page.as_mut() }
    }

    /// Read the next loose page pointer from the given page.
    #[inline(always)]
    pub(crate) fn next(&self) -> Option<&Page> {
        self.next.map(|next| unsafe { &*next.as_ptr() })
    }

    /// Read the next loose page pointer from the given page.
    pub(crate) fn next_mut(&mut self) -> Option<&mut Page> {
        self.next.map(|next| unsafe { &mut *next.as_ptr() })
    }

    /// Set the next loose page pointer from the given page.
    #[inline(always)]
    pub(crate) fn set_next(&mut self, next: Option<&mut Page>) {
        self.next = next
            .map(|p| NonNull::new(p as *mut _))
            .expect("invalid page pointer, null page reference");

        unsafe {
            Self::write_next_ptr(
                self.page.as_ptr(),
                self.next.map(|p| p.as_ptr()).unwrap_or(null_mut()),
            );
        }
    }

    /// Read the next pointer. Maintains the LMDB 2-byte offset.
    #[inline(always)]
    unsafe fn read_next_ptr(page: *mut Page) -> Option<NonNull<Page>> {
        let ptr = (page as *mut u8).add(2);
        NonNull::new(ptr as *mut Page)
    }

    /// Write the next pointer. Maintains the 2-byte offset.
    #[inline(always)]
    unsafe fn write_next_ptr(page: *mut Page, next: *mut Page) {
        let ptr = (page as *mut u8).add(2);
        *(ptr as *mut *mut Page) = next;
    }
}
