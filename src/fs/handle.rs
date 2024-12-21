use std::ops::Range;
use std::ptr;
use std::slice::from_raw_parts;
use std::sync::Arc;
use std::sync::atomic::{fence, AtomicU64};
use std::sync::atomic::Ordering::SeqCst;
use std::time::{SystemTime, UNIX_EPOCH};
use getset::{CopyGetters, Getters, Setters};
use memmap2::MmapMut;
use crate::errs::FsError;
use crate::errs::FsError::{ReadOutOfBounds, WriteOutOfBounds};
use crate::fs::core::Fs;
use crate::utils::Deserializer;

#[derive(Clone, Debug, Eq, PartialEq, CopyGetters, Setters)]
#[getset(get_copy = "pub(crate)")]
pub(crate) struct OrderedRange {
    start: u64,
    #[getset(set = "pub(crate)")]
    end: u64,
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

#[derive(Debug, Getters, CopyGetters, Setters)]
pub(crate) struct FRangeMetadata {
    #[getset(get = "pub(crate)", set = "pub(crate)")]
    range: OrderedRange,
    #[getset(get_copy = "pub(crate)")]
    id: u64,
    #[getset(get = "pub(crate)")]
    length: AtomicU64, // Track actual bytes written
    #[getset(get_copy = "pub(crate)")]
    size: u64,         // Keep this as allocated size
    #[getset(get_copy = "pub(crate)")]
    created_at: u64,
    #[getset(get = "pub(crate)")]
    modified_at: AtomicU64,
}

impl FRangeMetadata {
    pub(crate) fn new(range: OrderedRange, id: u64, length: u64, size: u64, created_at: u64, modified_at: u64) -> Self {
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

        Self {
            range,
            id,
            length: AtomicU64::new(length),
            size,
            created_at: now,
            modified_at: AtomicU64::new(mod_time),
        }
    }
}

impl Clone for FRangeMetadata {
    fn clone(&self) -> Self {
        Self {
            range: self.range.clone(),
            id: self.id,
            length: AtomicU64::new(self.length.load(SeqCst)),
            size: self.size,
            created_at: self.created_at,
            modified_at: AtomicU64::new(self.modified_at.load(SeqCst)),
        }
    }
}

pub struct FRangeHandle {
    mmap: Arc<MmapMut>,
    range: OrderedRange,
    metadata: FRangeMetadata,
    fs: Arc<Fs>,
}

impl FRangeHandle {
    pub(crate) fn new(mmap: Arc<MmapMut>, range: OrderedRange, metadata: FRangeMetadata, fs: Arc<Fs>) -> Self {
        Self {
            mmap,
            range,
            metadata,
            fs,
        }
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
        if offset as usize + data.len() > (self.range.end - self.range.start) as usize {
            return Err(ReadOutOfBounds);
        }

        let base = self.range.start as usize + offset as usize;

        if base + data.len() > self.range.end as usize {
            return Err(WriteOutOfBounds);
        }

        // Mark affected pages as dirty
        let start_page = base / self.fs.page_size;
        let end_page = (base + data.len()).div_ceil(self.fs.page_size);

        {
            let dirty = self.fs.dirty_pages.write();
            for page in start_page..end_page {
                dirty.insert(page);
            }
        }

        // SAFETY: we have already checked that the write is within bounds
        unsafe {
            // Get mutable pointer for destination
            let dst = self.mmap.as_ptr().add(base).cast::<u8>() as *mut u8;
            ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }

        fence(SeqCst);

        self.metadata.length.fetch_add(data.len() as u64, SeqCst);

        self.metadata.modified_at.store(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            SeqCst,
        );

        match self.fs.maybe_flush(false) {
            | Ok(_) => {},
            | Err(e) => return Err(e),
        };

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
        // First check if the read would be out of bounds of our allocated range
        if offset as usize + buf.len() > (self.range.end - self.range.start) as usize {
            return Err(ReadOutOfBounds);
        }

        // Then handle EOF case
        if offset >= self.metadata.size {
            return Ok(());
        }

        // Calculate how many bytes we can actually read based on written data
        let available_bytes = (self.metadata.size - offset) as usize;
        let bytes_to_read = buf.len().min(available_bytes);

        let base = self.range.start as usize + offset as usize;

        // mark affected pages as dirty
        let start_page = base / self.fs.page_size;
        let end_page = (base + buf.len()).div_ceil(self.fs.page_size);

        {
            for page in start_page..end_page {
                self.fs.dirty_pages.write().insert(page);
            }
        }

        // SAFETY: we have already checked that the read is within bounds
        unsafe {
            // Only copy the actually written bytes
            buf[..bytes_to_read]
                .copy_from_slice(from_raw_parts(self.mmap.as_ptr().add(base), bytes_to_read));
        }

        fence(SeqCst);
        Ok(())
    }

    pub(crate) fn capacity(&self) -> u64 {
        self.range.end - self.range.start
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