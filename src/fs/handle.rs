use std::{
    ops::Range,
    ptr,
    slice,
    slice::from_raw_parts,
    sync::{
        atomic::{
            fence,
            AtomicU64,
            Ordering::SeqCst,
        },
        Arc,
    },
    time::{
        SystemTime,
        UNIX_EPOCH,
    },
};
use std::cmp::{max, min};
use bytes::{
    Buf,
    BufMut,
    Bytes,
    BytesMut,
};
use crossbeam_skiplist::SkipSet;
use getset::{
    CopyGetters,
    Getters,
    Setters,
};
use memmap2::MmapMut;

use crate::{
    errs::{
        FsError,
        FsError::{
            ReadOutOfBounds,
            WriteOutOfBounds,
        },
    },
    fs::{
        core::{
            Fs,
            FsHeader,
        },
    },
    utils::Deserializer,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::fs) struct OrderedRange {
    pub(in crate::fs) start: u64,
    pub(in crate::fs) end: u64,
}

impl OrderedRange {
    pub(crate) fn new(start: u64, end: u64) -> Self {
        Self { start, end }
    }
}

impl From<OrderedRange> for Range<u64> {
    fn from(r: OrderedRange) -> Self {
        r.start..r.end
    }
}

impl From<Range<u64>> for OrderedRange {
    fn from(r: Range<u64>) -> Self {
        Self {
            start: r.start,
            end: r.end,
        }
    }
}

impl<'a> From<&'a [u8]> for &'a OrderedRange {
    fn from(slice: &'a [u8]) -> &'a OrderedRange {
        // Safety: ensure alignment and size match
        unsafe { &*(slice.as_ptr() as *const OrderedRange) }
    }
}

impl Ord for OrderedRange {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.start.cmp(&other.start).then(self.end.cmp(&other.end))
    }
}

impl PartialOrd for OrderedRange {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// TODO(@siennathesane): when this supports extension beyond the current range
// we have to make sure it can't grow past 4000 extensions since it's size
// is defined
#[derive(Debug)]
pub(crate) struct FRangeMetadata {
    pub(in crate::fs) ranges: Vec<OrderedRange>,
    pub(in crate::fs) id: u64,
    pub(in crate::fs) length: AtomicU64, // Track actual bytes written
    pub(in crate::fs) size: u64, // Keep this as allocated size
    pub(in crate::fs) created_at: u64,
    pub(in crate::fs) modified_at: AtomicU64,
}

impl FRangeMetadata {
    pub(crate) fn new(
        range: OrderedRange,
        id: u64,
        length: u64,
        size: u64,
        created_at: u64,
        modified_at: u64,
    ) -> Self {
        let mut now = 0;
        if created_at == 0 {
            now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
        } else {
            now = created_at;
        }

        let mut mod_time = 0;
        if modified_at == 0 {
            mod_time = now;
        } else {
            mod_time = modified_at;
        }

        let mut ranges = Vec::new();
        ranges.push(range);

        Self {
            ranges,
            id,
            length: AtomicU64::new(length),
            size,
            created_at: now,
            modified_at: AtomicU64::new(mod_time),
        }
    }

    pub(in crate::fs) fn serialize(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(size_of_val(self));
        buf.put_u64_le(self.id);
        buf.put_u64_le(self.length.load(SeqCst));
        buf.put_u64_le(self.size);
        buf.put_u64_le(self.created_at);
        buf.put_u64_le(self.modified_at.load(SeqCst));
        buf.put_u16_le(self.ranges.len() as u16);
        for range in self.ranges.iter() {
            buf.put_u64_le(range.start);
            buf.put_u64_le(range.end);
        }
        buf.freeze()
    }

    pub(in crate::fs) fn deserialize(bytes: Bytes) -> Self {
        let mut buf = Bytes::from(bytes);
        let id = buf.get_u64_le();
        let length = buf.get_u64_le();
        let size = buf.get_u64_le();
        let created_at = buf.get_u64_le();
        let modified_at = buf.get_u64_le();
        let num_ranges = buf.get_u16_le();
        let mut ranges = Vec::with_capacity(num_ranges as usize);

        for _ in 0..num_ranges {
            let start = buf.get_u64_le();
            let end = buf.get_u64_le();
            ranges.push(OrderedRange::new(start, end));
        }

        Self {
            ranges,
            id,
            length: AtomicU64::new(length),
            size,
            created_at,
            modified_at: AtomicU64::new(modified_at),
        }
    }
}

impl Clone for FRangeMetadata {
    fn clone(&self) -> Self {
        Self {
            ranges: self.ranges.clone(),
            id: self.id,
            length: AtomicU64::new(self.length.load(SeqCst)),
            size: self.size,
            created_at: self.created_at,
            modified_at: AtomicU64::new(self.modified_at.load(SeqCst)),
        }
    }
}

// TODO(@siennathesane): change `OrderedRange` to `Vec<OrderedRange>` to support
// multiple ranges
pub struct FRangeHandle {
    mmap: Arc<MmapMut>,
    pub(in crate::fs) ranges: Vec<OrderedRange>,
    metadata: FRangeMetadata,
    fs: Arc<Fs>,
}

impl FRangeHandle {
    pub(crate) fn new(
        mmap: Arc<MmapMut>,
        ranges: Vec<OrderedRange>,
        metadata: FRangeMetadata,
        fs: Arc<Fs>,
    ) -> Self {
        Self {
            mmap,
            ranges,
            metadata,
            fs,
        }
    }

    /// Add a new range to this handle
    pub(crate) fn add_range(&mut self, range: OrderedRange) {
        self.ranges.push(range);
        self.ranges.sort_by_key(|r| r.start);
    }

    /// Write data at the given offset.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it allows writing to "arbitrary" memory
    /// locations.
    /// - There is pointer arithmetic to calculate the destination pointer.
    /// - There is a `memcpy` with a raw pointer.
    pub fn write_at(&self, offset: u64, data: &[u8]) -> Result<(), FsError> {
        // Check if the write would exceed total capacity
        if offset as u64 + data.len() as u64 > self.capacity() {
            return Err(ReadOutOfBounds);
        }

        let mut data_written = 0;
        let mut current_logical_offset = offset;

        while data_written < data.len() {
            let (range_idx, physical_offset) = match self.map_offset(current_logical_offset)
                .ok_or(ReadOutOfBounds) {
                    | Ok(v) => (v.0, v.1),
                    | Err(_) => return Err(ReadOutOfBounds),
            };
            let range = &self.ranges[range_idx];

            // calculate how much we can write in this range
            let range_remaining = (range.end - range.start - physical_offset) as usize;
            let write_size = min(range_remaining, data.len() - data_written);

            let base = range.start as usize + physical_offset as usize;

            // calculate page-aligned boundaries for this chunk
            let start_page = (base / self.fs.page_size) * self.fs.page_size;
            let end_page = (base + write_size).div_ceil(self.fs.page_size) * self.fs.page_size;

            // ,ark affected pages as dirty
            {
                let dirty = self.fs.dirty_pages.write();
                for page in (start_page..end_page).step_by(self.fs.page_size) {
                    let page_num = page / self.fs.page_size;
                    dirty.insert(page_num);
                }
            }

            // SAFETY: bounds already checked
            unsafe {
                let dst = self.mmap.as_ptr().add(base).cast::<u8>() as *mut u8;
                ptr::copy_nonoverlapping(
                    data[data_written..].as_ptr(),
                    dst,
                    write_size
                );
            }

            data_written += write_size;
            current_logical_offset += write_size as u64;
        }

        fence(SeqCst);
        
        self.metadata
            .length
            .store(max(offset + data.len() as u64, self.len()), SeqCst);

        match self.fs.maybe_flush(false) {
            Ok(_) => {},
            Err(e) => return Err(e),
        };

        self.fs
            .metadata_changes
            .write()
            .mark_frange_modified(self.metadata.id);

        Ok(())
    }

    /// Read data at the given offset.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it allows reading from "arbitrary"
    /// memory locations.
    /// - Builds a slice from a raw pointer.
    /// - Performs pointer arithmetic to calculate the source pointer.
    pub fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FsError> {
        if offset + buf.len() as u64 > self.capacity() {
            return Err(ReadOutOfBounds);
        }

        let mut bytes_read = 0;
        let mut current_logical_offset = offset;

        while bytes_read < buf.len() {
            let (range_idx, physical_offset) = match self.map_offset(current_logical_offset)
                .ok_or(ReadOutOfBounds) {
                    | Ok(v) => (v.0, v.1),
                    | Err(_) => return Err(ReadOutOfBounds),
            };
            let range = &self.ranges[range_idx];

            // calculate how much we can read from this range
            let range_remaining = (range.end - range.start - physical_offset) as usize;
            let read_size = min(range_remaining, buf.len() - bytes_read);

            let base = range.start as usize + physical_offset as usize;

            // SAFETY: Bounds already checked
            unsafe {
                let src = self.mmap.as_ptr().add(base);
                let verify = from_raw_parts(src, read_size);

                buf[bytes_read..bytes_read + read_size]
                    .copy_from_slice(from_raw_parts(src, read_size));
            }

            bytes_read += read_size;
            current_logical_offset += read_size as u64;
        }

        fence(SeqCst);
        Ok(())
    }

    /// Maps a logical offset to the corresponding physical range and offset
    fn map_offset(&self, logical_offset: u64) -> Option<(usize, u64)> {
        let mut current_offset = 0;

        for (idx, range) in self.ranges.iter().enumerate() {
            let range_size = range.end - range.start;
            if logical_offset >= current_offset && logical_offset < current_offset + range_size {
                return Some((idx, logical_offset - current_offset));
            }
            current_offset += range_size;
        }
        None
    }

    pub(crate) fn capacity(&self) -> u64 {
        self.ranges.iter().map(|r| r.end - r.start).sum()
    }

    pub(crate) fn len(&self) -> u64 {
        self.metadata.length.load(SeqCst)
    }

    pub(crate) fn metadata(&self) -> &FRangeMetadata {
        &self.metadata
    }
}

impl Drop for FRangeHandle {
    fn drop(&mut self) {
        // we basically remove ourselves from the open ranges so that when this
        // is dropped you can theoretically delete the frange
        {
            let franges = self.fs.franges.write();
            match franges.get(&self.metadata.id) {
                | None => {},
                | Some(entry) => {
                    let updated = entry.value().clone();
                    updated.modified_at.store(
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        SeqCst,
                    );
                    updated
                        .length
                        .store(self.metadata.length.load(SeqCst), SeqCst);
                    franges.insert(self.metadata.id, updated);
                },
            };
        }

        {
            let mut open_franges = self.fs.open_franges.write();
            open_franges.remove(&self.metadata.id);
        }
    }
}
