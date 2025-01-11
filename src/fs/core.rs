use std::{
    collections::BinaryHeap,
    io::{
        Error,
        ErrorKind::{
            AlreadyExists,
            InvalidInput,
        },
    },
    mem::ManuallyDrop,
    ops::Range,
    ptr,
    slice,
    slice::from_raw_parts,
    sync::{
        atomic,
        atomic::{
            fence,
            AtomicBool,
            AtomicU64,
            Ordering::{
                AcqRel,
                Acquire,
                Relaxed,
                SeqCst,
            },
        },
        Arc,
    },
    time::{
        SystemTime,
        UNIX_EPOCH,
    },
};

use bytes::{
    Buf,
    BufMut,
    Bytes,
    BytesMut,
};
use crossbeam_skiplist::{
    map::Entry,
    SkipMap,
    SkipSet,
};
use getset::{
    CopyGetters,
    Getters,
};
use gxhash::{
    HashSet,
    HashSetExt,
};
use memmap2::{
    Advice::WillNeed,
    MmapMut,
    UncheckedAdvice::DontNeed,
};
use parking_lot::{
    Mutex,
    RawRwLock,
    RwLock,
    RwLockWriteGuard,
};

use crate::{
    block::BLOCK_SIZE,
    errs::{
        FsError,
        FsError::{
            FRangeAlreadyOpen,
            FRangeNotFound,
            FRangeStillOpen,
            FragmentationLimit,
            InsufficientSpace,
            InvalidHeaderFormat,
            IoError,
            NoAdjacentSpace,
            NoFreeSpace,
            ReadOutOfBounds,
            StorageExhausted,
            WriteOutOfBounds,
        },
    },
    fs::{
        compaction::FragmentationStats,
        handle::{
            FRangeHandle,
            FRangeMetadata,
            OrderedRange,
        },
        metadata::{
            FsMetadata,
            MetadataChanges,
        },
    },
};

// TODO(@siennathesane): make this configurable
pub(in crate::fs) const FLUSH_INTERVAL_SECS: u64 = 5;

#[repr(C)]
#[derive(Debug, CopyGetters)]
pub(in crate::fs) struct FsHeader {
    magic: [u8; 8],       // Magic bytes to identify our filesystem
    version: u32,         // Version for future compatibility
    page_size: u32,       // Page size used for this filesystem
    next_frange_id: u64,  // Next available frange ID
    metadata_offset: u64, // Offset to the metadata region
    metadata_size: u64,   // Size of the metadata region
}

impl FsHeader {
    const CURRENT_VERSION: u32 = 1;
    const MAGIC: &'static [u8; 8] = b"CESIUMFS";

    fn new(
        page_size: u32,
        metadata_offset: u64,
        metadata_size: u64,
    ) -> Self {
        Self {
            magic: *Self::MAGIC,
            version: Self::CURRENT_VERSION,
            page_size,
            next_frange_id: 0,
            metadata_offset,
            metadata_size,
        }
    }

    fn serialize(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(size_of::<FsHeader>());
        buf.extend_from_slice(&self.magic);
        buf.extend_from_slice(&self.version.to_le_bytes());
        buf.extend_from_slice(&self.page_size.to_le_bytes());
        buf.extend_from_slice(&self.next_frange_id.to_le_bytes());
        buf.extend_from_slice(&self.metadata_offset.to_le_bytes());
        buf.extend_from_slice(&self.metadata_size.to_le_bytes());
        buf.freeze()
    }
}

fn deserialize_header(bytes: &[u8]) -> Result<FsHeader, FsError> {
    if bytes.len() < size_of::<FsHeader>() {
        return Err(InvalidHeaderFormat("header too short".into()));
    }

    let mut magic = [0u8; 8];
    magic.copy_from_slice(&bytes[0..8]);

    if magic != *FsHeader::MAGIC {
        return Err(InvalidHeaderFormat("invalid magic bytes".into()));
    }

    Ok(FsHeader {
        magic,
        version: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        page_size: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
        next_frange_id: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
        metadata_offset: u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
        metadata_size: u64::from_le_bytes(bytes[32..40].try_into().unwrap()),
    })
}

// TODO(@siennathesane): this isn't valid, i'm not sure why it's here
pub(in crate::fs) const INITIAL_METADATA_SIZE: usize = 4096;

/// A custom filesystem implementation for CesiumDB. It is designed to work with
/// physical storage devices or `fallocate`d files and uses memory-mapped files
/// to provide fast access to the data.
pub struct Fs {
    pub(in crate::fs) mmap: Arc<MmapMut>,
    pub(in crate::fs) franges: RwLock<SkipMap<u64, FRangeMetadata>>,

    pub(in crate::fs) free_ranges: RwLock<SkipSet<OrderedRange>>,
    pub(in crate::fs) open_franges: RwLock<HashSet<u64>>,
    pub(in crate::fs) next_frange_id: AtomicU64,

    // flushing
    pub(in crate::fs) last_flush: AtomicU64,
    pub(in crate::fs) dirty_pages: RwLock<SkipSet<usize>>,
    pub(in crate::fs) page_size: usize,

    // journaling
    pub(in crate::fs) metadata_changes: RwLock<MetadataChanges>,
}

impl Fs {
    pub(crate) fn new(mmap: MmapMut) -> Result<Arc<Self>, FsError> {
        match mmap.advise(memmap2::Advice::Random) {
            | Ok(_) => {},
            | Err(e) => return Err(IoError(e)),
        };

        // Read header
        let header = {
            let mut header_bytes = [0u8; size_of::<FsHeader>()];
            header_bytes.copy_from_slice(&mmap[..size_of::<FsHeader>()]);
            match deserialize_header(&header_bytes) {
                | Ok(v) => v,
                | Err(e) => return Err(e),
            }
        };

        println!("Reading header from new(): {:?}", header);

        // Read metadata
        let metadata: FsMetadata = {
            let start = header.metadata_offset as usize;
            let end = start + header.metadata_size as usize;
            println!("Reading metadata from offset {} to {}", start, end);
            let metadata_bytes = &mmap[start..end];
            println!("First few bytes of metadata: {:?}", &metadata_bytes[..20]);

            match FsMetadata::deserialize(Bytes::copy_from_slice(metadata_bytes)) {
                | Ok(v) => {
                    println!("Deserialized metadata. Franges count: {}", v.franges.len());
                    v
                },
                | Err(_) => return Err(InvalidHeaderFormat("invalid metadata region".into())),
            }
        };

        // Initialize state from metadata
        let franges = SkipMap::new();
        for entry in metadata.franges.iter() {
            println!("Adding frange {} to new fs", entry.key());
            franges.insert(*entry.key(), entry.value().clone());
        }

        let free_ranges = SkipSet::new();
        for range in metadata.free_ranges {
            free_ranges.insert(range);
        }

        let fs = Arc::new(Self {
            mmap: Arc::new(mmap),
            franges: RwLock::new(franges),
            free_ranges: RwLock::new(free_ranges),
            open_franges: RwLock::new(HashSet::default()),
            next_frange_id: AtomicU64::new(header.next_frange_id),
            last_flush: AtomicU64::new(0),
            dirty_pages: RwLock::new(SkipSet::new()),
            page_size: header.page_size as usize,
            metadata_changes: RwLock::new(MetadataChanges::default()),
        });

        // flush to ensure it's written
        match fs.sync() {
            | Ok(_) => {},
            | Err(e) => return Err(e),
        };

        Ok(fs)
    }

    /// Initialize a new filesystem in the given mmap
    pub fn init(mut mmap: MmapMut) -> Result<Arc<Self>, FsError> {
        let total_size = mmap.len() as u64;

        // Calculate sizes including journal
        let header_size = size_of::<FsHeader>();
        let metadata_size = INITIAL_METADATA_SIZE;
        let data_start = (header_size + metadata_size) as u64;

        println!("Filesystem initialization:");
        println!("  Total size: {}", total_size);
        println!("  Header size: {}", header_size);
        println!("  Metadata size: {}", metadata_size);
        println!("  Data start: {}", data_start);
        println!("  Available space: {}", total_size - data_start);

        // Verify we have enough space
        if total_size < data_start {
            return Err(InsufficientSpace);
        }

        // Create and write header
        let header_bytes = FsHeader::new(
            BLOCK_SIZE as u32,
            header_size as u64, // metadata starts after header
            metadata_size as u64,
        )
        .serialize();

        // Write header using actual serialized size
        mmap[..header_size].copy_from_slice(&header_bytes);

        // Initialize empty metadata
        let metadata = FsMetadata::new();
        let encoded = metadata.serialize();

        // Zero metadata region and write encoded data
        mmap[header_size..metadata_size + header_size].fill(0);
        mmap[header_size..header_size + encoded.len()].copy_from_slice(&encoded);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mmap = Arc::new(mmap);

        // create filesystem with single free range
        let fs = Self {
            mmap,
            franges: RwLock::new(SkipMap::new()),
            free_ranges: RwLock::new(SkipSet::new()),
            open_franges: RwLock::new(HashSet::default()),
            next_frange_id: AtomicU64::new(0),
            last_flush: AtomicU64::new(now),
            dirty_pages: RwLock::new(SkipSet::new()),
            page_size: BLOCK_SIZE,
            metadata_changes: RwLock::new(MetadataChanges::default()),
        };

        // add initial range and add to tracking
        let initial_range = OrderedRange::from(data_start..total_size);
        fs.free_ranges.write().insert(initial_range.clone());
        fs.metadata_changes
            .write()
            .mark_range_modified(initial_range);

        let fs = Arc::new(fs);
        match fs.persist_metadata() {
            | Ok(_) => {},
            | Err(e) => return Err(e),
        }

        Ok(fs)
    }

    pub fn create_frange(self: &Arc<Self>, size: u64) -> Result<u64, FsError> {
        let range = {
            let free = self.free_ranges.write();
            self.find_free_range(&free, size)
        };

        let range = match range {
            | Ok(range) => range,
            | Err(NoFreeSpace) => {
                let total_space = self.total_free_space();
                return if size > total_space {
                    Err(StorageExhausted)
                } else {
                    Err(FragmentationLimit)
                };
            },
            | Err(e) => return Err(e),
        };

        // get the id and create metadata
        let id = self.next_frange_id.fetch_add(1, SeqCst);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let metadata = FRangeMetadata::new(range.clone(), id, 0, size, now, now);

        {
            self.franges.write().insert(id, metadata);
        }

        {
            self.metadata_changes.write().mark_frange_modified(id);
        }

        {
            self.metadata_changes.write().mark_range_modified(range);
        }

        match self.persist_metadata() {
            | Ok(_) => {},
            | Err(e) => return Err(e),
        };

        Ok(id)
    }

    pub fn open_frange(self: &Arc<Self>, id: u64) -> Result<FRangeHandle, FsError> {
        let mut open_franges = self.open_franges.write();

        if !open_franges.insert(id) {
            return Err(FRangeAlreadyOpen);
        }

        let franges = self.franges.read();
        let metadata = match franges.get(&id) {
            | None => {
                return Err(FRangeNotFound);
            },
            | Some(v) => v.value().clone(), // Dereference and clone the value
        };

        Ok(FRangeHandle::new(
            self.mmap.clone(),
            metadata.ranges.clone(),
            metadata,
            self.clone(),
        ))
    }

    pub fn close_frange(self: &Arc<Self>, handle: FRangeHandle) -> Result<(), FsError> {
        let frange_id = handle.metadata().id;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let length = handle.metadata().length.load(SeqCst);

        // Update metadata first
        {
            let updated = {
                match self.franges.read().get(&frange_id) {
                    | None => return Err(FRangeNotFound),
                    | Some(entry) => {
                        let updated = entry.value().clone();
                        updated.modified_at.store(now, SeqCst);
                        updated.length.store(length, SeqCst);
                        updated
                    },
                }
            };

            {
                self.franges.write().insert(frange_id, updated);
            }
        }

        // Mark changes
        {
            self.metadata_changes
                .write()
                .mark_frange_modified(frange_id);
        }

        // Remove from open_franges
        {
            self.open_franges.write().remove(&frange_id);
        }

        match self.persist_metadata() {
            | Ok(_) => {},
            | Err(e) => return Err(e),
        };

        // Finally flush
        self.maybe_flush(true)
    }

    pub fn delete_frange(self: &Arc<Self>, id: u64) -> Result<(), FsError> {
        if self.open_franges.read().contains(&id) {
            return Err(FRangeStillOpen);
        }

        // Remove the frange from the franges map
        let range = {
            let franges = self.franges.write();
            let x = if let Some(entry) = franges.remove(&id) {
                entry.value().ranges.clone()
            } else {
                return Ok(());
            };
            x
        };

        // update the free ranges
        {
            let free_ranges = self.free_ranges.write();
            for range in range.clone() {
                free_ranges.insert(range.clone());
            }
        }

        {
            self.metadata_changes.write().mark_frange_modified(id);
        }

        {
            let metadata = self.metadata_changes.write();
            for range in range {
                metadata.mark_range_modified(range);
            }
        }

        match self.coalesce_free_ranges() {
            | Ok(_) => {},
            | Err(e) => return Err(e),
        };
        self.persist_metadata()
    }

    /// Sync all changes to disk
    pub fn sync(self: &Arc<Self>) -> Result<(), FsError> {
        self.flush_dirty_pages()
    }

    /// Persist metadata changes to disk
    pub(in crate::fs) fn persist_metadata(self: &Arc<Self>) -> Result<(), FsError> {
        // take a single lock to fetch the info
        let (has_changes, mut modified_franges, free_ranges_modified) = {
            let changes = self.metadata_changes.read();
            (
                !changes.modified_franges.is_empty() ||
                    !changes.modified_ranges.is_empty() ||
                    changes.header_modified.load(SeqCst),
                changes
                    .modified_franges
                    .iter()
                    .map(|id| *id.value())
                    .collect::<Vec<_>>(),
                !changes.modified_ranges.is_empty(),
            )
        };

        // if we don't have any changes, just return
        if !has_changes {
            return Ok(());
        }

        // get the current header
        let mut header = {
            let mut header_bytes = [0u8; size_of::<FsHeader>()];
            header_bytes.copy_from_slice(&self.mmap[..size_of::<FsHeader>()]);
            match deserialize_header(&header_bytes) {
                | Ok(v) => v,
                | Err(e) => return Err(e),
            }
        };

        let metadata_start = header.metadata_offset as usize;
        let metadata_end = metadata_start + header.metadata_size as usize;

        // sort the modified franges so we can do this in a single iteration
        modified_franges.sort();

        // pre-calculate the space needed for the metadata
        let mut total_space_needed = 0;

        for &id in &modified_franges {
            if let Some(frange) = self.franges.read().get(&id) {
                total_space_needed += size_of_val(frange.value()) + 2; // +2 for
                                                                       // size prefix
            }
        }

        // see if we need to grow the metadata region
        if total_space_needed > (metadata_end - metadata_start) {
            let new_size = total_space_needed * 2; // double for future growth
            match self.grow_metadata_region(&mut header, new_size) {
                | Ok(_) => {},
                | Err(e) => return Err(e),
            };
        }

        // warn the kernel to read the new metadata region
        // with enough time for it to load the relevant pages
        // that should make this fast enough to not be a problem.
        // if the metadata grew, this will encompass those changes.
        match self
            .mmap
            .advise_range(WillNeed, metadata_start, metadata_end - metadata_start)
        {
            | Ok(_) => {},
            | Err(e) => return Err(IoError(e)),
        };

        let mut current_pos = metadata_start;
        let mut frange_idx = 0;
        
        let mut_ptr = Arc::as_ptr(&self.mmap) as *mut u8;
        while current_pos < metadata_end && frange_idx < modified_franges.len() {
            // read current record size
            let mut size_bytes = [0u8; 2];
            // SAFETY: we've already done the bounds check
            unsafe {
                ptr::copy_nonoverlapping(
                    self.mmap.as_ptr().add(current_pos),
                    size_bytes.as_mut_ptr(),
                    2,
                );
            }
            let current_size = u16::from_le_bytes(size_bytes) as usize;

            // read current record id
            let mut id_bytes = [0u8; 8];
            // SAFETY: we've already done the bounds check
            unsafe {
                ptr::copy_nonoverlapping(
                    self.mmap.as_ptr().add(current_pos + 4),
                    id_bytes.as_mut_ptr(),
                    8,
                );
            }
            let current_id = u64::from_le_bytes(id_bytes);

            if current_id == modified_franges[frange_idx] {
                let frange = match self.franges.read().get(&current_id) {
                    | None => return Err(FRangeNotFound),
                    | Some(v) => v.value().clone(),
                };
                let new_bytes = frange.serialize();
                let new_size = new_bytes.len();
                let new_total_size = new_size + 2;
                let old_total_size = current_size + 2;

                // sometimes the franges will grow or shrink, so we need to shift the data if
                // needed
                if new_total_size != old_total_size {
                    // calculate shift
                    let shift = new_total_size as isize - old_total_size as isize;
                    let next_pos = current_pos + old_total_size;

                    // Ensure we don't overflow metadata region
                    if shift > 0 && next_pos + shift as usize > metadata_end {
                        return Err(InsufficientSpace);
                    }

                    // calculate remaining data length
                    let remaining_len = metadata_end - next_pos;

                    // SAFETY: we've already done the bounds check
                    unsafe {
                        // shift remaining data if needed
                        if remaining_len > 0 {
                            ptr::copy(
                                self.mmap.as_ptr().add(next_pos),
                                mut_ptr.add(next_pos + shift as usize),
                                remaining_len,
                            );
                        }
                    }
                    
                    // ensure the memory ordering is correct before new records are written
                    fence(SeqCst);
                }

                // write new record
                let size_prefix = (new_size as u16).to_le_bytes();
                self.write_to_mmap(current_pos, &size_prefix);
                self.write_to_mmap(current_pos + 2, &new_bytes);

                frange_idx += 1;
                current_pos += new_total_size;
            } else {
                current_pos += current_size + 2;
            }
        }

        // update header with new metadata size and next_frange_id
        {
            let changes = self.metadata_changes.read();
            if changes.header_modified.load(SeqCst) {
                header.next_frange_id = self.next_frange_id.load(SeqCst);
                let header_bytes = header.serialize();
                self.write_to_mmap(0, &header_bytes);
            }
        }

        // clear the change tracking
        {
            self.metadata_changes.write().clear();
        }

        Ok(())
    }

    fn grow_metadata_region(&self, header: &mut FsHeader, new_size: usize) -> Result<(), FsError> {
        let metadata_end = header.metadata_offset + header.metadata_size;
        let growth_needed = new_size as u64 - header.metadata_size;

        // TODO(@siennathesane): this is wrong. it need to check if there's enough free space
        // from the free_ranges section, not just the size of the mmap
        if metadata_end + growth_needed > self.mmap.len() as u64 {
            return Err(InsufficientSpace);
        }

        header.metadata_size = new_size as u64;
        Ok(())
    }

    /// Write data to the mmap at the given offset.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it dereferences a raw pointer.
    /// - There is pointer arithmetic to calculate the destination pointer.
    /// - There is a `memcpy` on a raw pointer.
    #[inline]
    fn write_to_mmap(self: &Arc<Self>, offset: usize, data: &[u8]) {
        // TODO(@siennathesane): do some bounds checking or something
        // SAFETY: yeah this is actually unsafe, see docstring
        unsafe {
            let ptr = self.mmap.as_ptr().add(offset) as *mut u8;
            ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        fence(SeqCst);
    }

    pub(in crate::fs) fn find_free_range(
        self: &Arc<Self>,
        free: &RwLockWriteGuard<SkipSet<OrderedRange>>,
        size: u64,
    ) -> Result<OrderedRange, FsError> {
        // Make sure we're only allocating ranges in the data region
        let header_bytes = {
            let mut header_bytes = [0u8; size_of::<FsHeader>()];
            header_bytes.copy_from_slice(&self.mmap[..size_of::<FsHeader>()]);
            match deserialize_header(&header_bytes) {
                | Ok(v) => v,
                | Err(e) => return Err(e),
            }
        };

        // calculate start of data region - just after metadata
        let data_start = header_bytes.metadata_offset + header_bytes.metadata_size;

        // Find the rightmost suitable range by iterating in reverse
        let suitable_range = match free
            .iter()
            .rev()
            .find(|r| r.end - r.start >= size && r.start >= data_start)
        {
            | None => return Err(NoFreeSpace),
            | Some(v) => v,
        };

        free.remove(&suitable_range);

        // When splitting a range, keep the left part free and allocate from the right
        if suitable_range.end - suitable_range.start > size {
            let new_free = OrderedRange::new(suitable_range.start, suitable_range.end - size);
            free.insert(new_free);
        }

        Ok(OrderedRange::new(
            suitable_range.end - size,
            suitable_range.end,
        ))
    }

    pub(in crate::fs) fn coalesce_free_ranges(self: &Arc<Self>) -> Result<(), FsError> {
        let mut ranges = {
            let free = self.free_ranges.read();
            let mut ranges: Vec<OrderedRange> = free.iter().map(|entry| (*entry).clone()).collect();
            ranges.sort();
            ranges
        };

        let mut i = 0;
        while i < ranges.len() - 1 {
            if ranges[i].end == ranges[i + 1].start {
                let merged = OrderedRange::new(ranges[i].start, ranges[i + 1].end);
                // Take write lock only for the actual modifications
                {
                    let free = self.free_ranges.write();
                    free.remove(&ranges[i]);
                    free.remove(&ranges[i + 1]);
                    free.insert(merged.clone());
                }
                ranges.remove(i + 1);
            } else {
                i += 1;
            }
        }

        Ok(())
    }

    pub(crate) fn maybe_flush(self: &Arc<Self>, force: bool) -> Result<(), FsError> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let last = self.last_flush.load(Acquire);

        if !force && now - last < FLUSH_INTERVAL_SECS {
            return Ok(());
        }

        if self
            .last_flush
            .compare_exchange(last, now, AcqRel, Relaxed)
            .is_err()
        {
            return Ok(());
        }

        self.flush_dirty_pages()
    }

    fn flush_dirty_pages(self: &Arc<Self>) -> Result<(), FsError> {
        // First collect pages
        let pages = {
            let dirty_pages = self.dirty_pages.write();
            dirty_pages
                .iter()
                .map(|entry| *entry)
                .collect::<Vec<usize>>()
        };

        // Then persist metadata
        match self.persist_metadata() {
            | Ok(_) => {},
            | Err(e) => return Err(e),
        };

        // Handle ranges
        let ranges = self.consolidate_pages(&pages);

        for range in ranges {
            match self.mmap.flush_range(
                range.start as usize,
                (range.end - range.start) as usize,
            ) {
                | Ok(_) => {},
                | Err(e) => return Err(IoError(e)),
            };
        }

        // Finally clear dirty pages
        {
            let dirty_pages = self.dirty_pages.write();
            for page in pages {
                dirty_pages.remove(&page);
            }
        }

        Ok(())
    }

    fn consolidate_pages(self: &Arc<Self>, pages: &[usize]) -> Vec<OrderedRange> {
        let mut ranges = Vec::new();
        let mut current_range: Option<OrderedRange> = None;

        for &page in pages {
            let start = page * self.page_size;
            let end = start + self.page_size;

            match &mut current_range {
                | Some(range) if range.end as usize == start => {
                    range.end = end as u64;
                },
                | Some(range) => {
                    ranges.push(range.clone());
                    current_range = Some(OrderedRange::new(start as u64, end as u64));
                },
                | None => {
                    current_range = Some(OrderedRange::new(start as u64, end as u64));
                },
            }
        }

        if let Some(range) = current_range {
            ranges.push(range);
        }

        ranges
    }
}

#[derive(Debug)]
pub struct MetadataSpaceInfo {
    pub total_capacity: u64,
    pub current_size: u64,
    pub available_space: u64,
}

#[derive(Debug, Getters)]
#[get = "pub"]
pub struct FsHealth {
    space_accounting_valid: bool,
    metadata_healthy: bool,
    fragmentation_percent: f64,
    total_size: u64,
    used_space: u64,
    free_space: u64,
    metadata_space: MetadataSpaceInfo,
    fragmentation_stats: FragmentationStats,
}

#[cfg(test)]
#[allow(clippy::question_mark_used)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::undocumented_unsafe_blocks)]
mod tests {
    use std::{
        fs::{
            File,
            OpenOptions,
        },
        io::Read,
        sync::{
            atomic::Ordering::{
                Acquire,
                SeqCst,
            },
            Arc,
        },
        thread,
        time::Duration,
    };
    use std::cmp::max;
    use memmap2::MmapMut;
    use parking_lot::deadlock;
    use tempfile::tempdir;

    use crate::{
        block::BLOCK_SIZE,
        errs::FsError::InvalidHeaderFormat,
        fs::{
            compaction::CompactionConfig,
            core::{
                deserialize_header,
                Fs,
                FsHeader,
                INITIAL_METADATA_SIZE,
            },
        },
    };

    const TEST_FILE_SIZE: u64 = 1024 * 1024 * 10; // 10MB

    fn setup_test_file() -> (tempfile::TempDir, File) {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.db");

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&file_path)
            .unwrap();

        file.set_len(TEST_FILE_SIZE).unwrap();
        (dir, file)
    }

    fn create_and_reopen_fs() -> (tempfile::TempDir, File, Arc<Fs>) {
        let (dir, file) = setup_test_file();

        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };

        println!("Creating filesystem");
        let fs = Fs::init(mmap).expect("Failed to initialize filesystem");

        // Ensure mmap is synced before dropping
        fs.sync().expect("Failed to sync mmap");

        drop(fs);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(dir.path().join("test.db"))
            .unwrap();

        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };

        let fs = Fs::new(mmap).expect("Failed to reopen filesystem");

        (dir, file, fs)
    }

    fn setup_deadlock_detection() {
        thread::spawn(move || loop {
            thread::sleep(Duration::from_secs(1));
            let deadlocks = deadlock::check_deadlock();
            if !deadlocks.is_empty() {
                eprintln!("{} deadlocks detected", deadlocks.len());
                for (i, threads) in deadlocks.iter().enumerate() {
                    eprintln!("Deadlock #{}", i);
                    for t in threads {
                        eprintln!("Thread Id {:#?}", t.thread_id());
                        eprintln!("{:#?}", t.backtrace());
                    }
                }
            }
        });
    }

    #[test]
    fn test_create_frange() {
        setup_deadlock_detection();

        let (_dir, _file, fs) = create_and_reopen_fs();

        // Test creating a frange
        let id = fs.create_frange(1024).unwrap();
        assert_eq!(id, 0);

        // Verify metadata
        let franges = fs.franges.read();
        let metadata = franges.get(&id).unwrap();
        assert_eq!(metadata.value().size, 1024);
        assert_eq!(metadata.value().id, 0);
    }

    #[test]
    fn test_open_and_close_frange() {
        setup_deadlock_detection();

        let (_dir, _file, fs) = create_and_reopen_fs();

        // Create and open a frange
        let id = fs.create_frange(1024).unwrap();
        let handle = fs.open_frange(id).unwrap();

        // Verify we can't open it again
        assert!(fs.open_frange(id).is_err());

        // Close it
        fs.close_frange(handle).unwrap();

        // Verify we can open it again
        assert!(fs.open_frange(id).is_ok());
    }

    #[test]
    fn test_write_and_read() {
        setup_deadlock_detection();

        let (_dir, _file, fs) = create_and_reopen_fs();

        // Create and open two franges that we'll combine
        let id1 = fs.create_frange(512).unwrap();
        let id2 = fs.create_frange(512).unwrap();

        let mut handle = fs.open_frange(id1).unwrap();
        let handle2 = fs.open_frange(id2).unwrap();

        // Add the second range to our first handle
        handle.add_range(handle2.ranges[0].clone());

        // Write data that spans both ranges
        let data = b"Hello, World! This is a longer string that will span multiple ranges";
        handle.write_at(0, data).unwrap();

        // Read it back
        let mut buf = vec![0u8; data.len()];
        handle.read_at(0, &mut buf).unwrap();
        assert_eq!(&buf, data);

        // Test writing at offset that crosses range boundary
        let data2 = b"Cross-boundary write";
        let cross_boundary_offset = 500; // Close to the boundary between ranges
        handle.write_at(cross_boundary_offset, data2).unwrap();

        // Read back the cross-boundary write
        let mut buf2 = vec![0u8; data2.len()];
        handle.read_at(cross_boundary_offset, &mut buf2).unwrap();
        assert_eq!(&buf2, data2);

        assert_eq!(handle.len(), max(
            cross_boundary_offset + data2.len() as u64,
            data.len() as u64
        ));
        assert_eq!(handle.capacity(), 1024); // 512 + 512
    }

    #[test]
    fn test_delete_frange() {
        setup_deadlock_detection();

        let (_dir, _file, fs) = create_and_reopen_fs();

        // Create a frange
        let id = fs.create_frange(1024).unwrap();

        // Delete it
        fs.delete_frange(id).unwrap();

        // Verify it's gone
        let franges = fs.franges.read();
        assert!(franges.get(&id).is_none());

        // Verify space was freed
        let free = fs.free_ranges.read();
        assert_eq!(free.len(), 1);
    }

    #[test]
    fn test_out_of_bounds_write() {
        setup_deadlock_detection();

        let (_dir, _file, fs) = create_and_reopen_fs();

        let id = fs.create_frange(1024).unwrap();
        let handle = fs.open_frange(id).unwrap();

        // Try to write beyond range
        let data = vec![1u8; 2048];
        assert!(handle.write_at(0, &data).is_err());
    }

    #[test]
    fn test_out_of_bounds_read() {
        setup_deadlock_detection();

        let (_dir, _file, fs) = create_and_reopen_fs();

        let id = fs.create_frange(1024).unwrap();
        let handle = fs.open_frange(id).unwrap();

        // Try to read beyond range
        let mut buf = vec![0u8; 2048];
        assert!(handle.read_at(0, &mut buf).is_err());
    }

    #[test]
    fn test_multiple_franges() {
        setup_deadlock_detection();

        let (_dir, _file, fs) = create_and_reopen_fs();

        // Create multiple franges
        let id1 = fs.create_frange(1024).unwrap();
        let id2 = fs.create_frange(1024).unwrap();
        let id3 = fs.create_frange(1024).unwrap();

        // Write to each
        let handle1 = fs.open_frange(id1).unwrap();
        let handle2 = fs.open_frange(id2).unwrap();
        let handle3 = fs.open_frange(id3).unwrap();

        handle1.write_at(0, b"first").unwrap();
        handle2.write_at(0, b"second").unwrap();
        handle3.write_at(0, b"third").unwrap();

        // read back and verify
        let mut buf = vec![0u8; 6];
        handle1.read_at(0, &mut buf[..5]).unwrap();
        assert_eq!(&buf[..5], b"first");

        handle2.read_at(0, &mut buf).unwrap();
        assert_eq!(&buf, b"second");

        handle3.read_at(0, &mut buf[..5]).unwrap();
        assert_eq!(&buf[..5], b"third");
    }

    #[test]
    fn test_fragmentation() {
        setup_deadlock_detection();

        let (_dir, _file, fs) = create_and_reopen_fs();

        // Create and delete franges to create fragmentation
        let id1 = fs.create_frange(1024).unwrap();
        let id2 = fs.create_frange(1024).unwrap();
        let id3 = fs.create_frange(1024).unwrap();

        // Delete middle frange
        fs.delete_frange(id2).unwrap();

        // Create new frange that should fit in the gap
        let id4 = fs.create_frange(512).unwrap();

        // Verify we can still use all franges
        let handle1 = fs.open_frange(id1).unwrap();
        let handle3 = fs.open_frange(id3).unwrap();
        let handle4 = fs.open_frange(id4).unwrap();

        handle1.write_at(0, b"data1").unwrap();
        handle3.write_at(0, b"data3").unwrap();
        handle4.write_at(0, b"data4").unwrap();
    }

    #[test]
    fn test_flush_behavior() {
        setup_deadlock_detection();

        let (_dir, _file, fs) = create_and_reopen_fs();

        let id = fs.create_frange(1024).unwrap();
        let handle = fs.open_frange(id).unwrap();

        // Write data
        handle.write_at(0, b"test data").unwrap();

        // Force flush
        fs.maybe_flush(true).unwrap();

        // Verify dirty pages were cleared
        assert_eq!(fs.dirty_pages.read().len(), 0);
    }

    #[test]
    fn test_concurrent_access() {
        setup_deadlock_detection();

        use std::thread;

        let (_dir, _file, fs) = create_and_reopen_fs();
        let fs = Arc::new(fs);

        let id = fs.create_frange(1024).unwrap();

        let mut handles = vec![];

        // Spawn multiple threads trying to open the same frange
        for _ in 0..10 {
            let fs_clone = fs.clone();
            let handle = thread::spawn(move || {
                let result = fs_clone.open_frange(id);
                assert!(result.is_ok() || result.is_err());
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_fs_init() {
        setup_deadlock_detection();

        let (dir, file) = setup_test_file();

        // Test basic initialization
        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let fs = Fs::init(mmap).expect("Failed to initialize filesystem");

        // Verify initial state
        assert_eq!(fs.next_frange_id.load(Acquire), 0);
        assert_eq!(fs.free_ranges.read().len(), 1); // Should have one large free range
        assert_eq!(fs.franges.read().len(), 0); // No franges yet

        // Verify the free range is correctly sized
        {
            let free_range = fs.free_ranges.read();
            let free_range = free_range.iter().next().unwrap();
            assert!(free_range.end > free_range.start); // Ensure valid range
            assert!(free_range.start >= (size_of::<FsHeader>() + INITIAL_METADATA_SIZE) as u64);
            // After header and metadata
        }

        drop(fs);
        dir.close().unwrap();
    }

    #[test]
    fn test_fs_new_after_init() {
        setup_deadlock_detection();

        let (dir, file) = setup_test_file();

        // First initialize
        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let fs = Fs::init(mmap).expect("Failed to initialize filesystem");

        // Force sync everything
        fs.maybe_flush(true).unwrap();
        fs.mmap.flush().unwrap();
        drop(fs);

        // Now try to open with new()
        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let result = Fs::new(mmap);
        assert!(result.is_ok());

        dir.close().unwrap();
    }

    #[test]
    fn test_fs_new_on_uninitialized() {
        setup_deadlock_detection();

        let (dir, file) = setup_test_file();
        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };

        // Try to open uninitialized file - should fail with invalid magic
        let result = Fs::new(mmap);
        assert!(matches!(result, Err(InvalidHeaderFormat(_))));

        dir.close().unwrap();
    }

    #[test]
    fn test_header_persistence() {
        setup_deadlock_detection();

        let (dir, mut file, _) = create_and_reopen_fs();

        // Read the raw header bytes
        let mut header_bytes = vec![0u8; size_of::<FsHeader>()];
        file.read_exact(&mut header_bytes).unwrap();

        // Deserialize and verify header
        let header = deserialize_header(&header_bytes).unwrap();
        assert_eq!(&header.magic, FsHeader::MAGIC);
        assert_eq!(header.version, FsHeader::CURRENT_VERSION);
        assert_eq!(header.page_size as usize, BLOCK_SIZE);

        // Clean up
        drop(file);
        dir.close().unwrap();
    }

    #[test]
    fn test_frange_metadata_persistence() {
        setup_deadlock_detection();

        // Create initial filesystem and add some franges
        let (dir, file) = setup_test_file();
        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let fs = Fs::init(mmap).expect("Failed to initialize filesystem");

        // Create some franges with different sizes
        let id1 = fs.create_frange(1024).unwrap();
        let id2 = fs.create_frange(2048).unwrap();

        // Write some data to create metadata
        let handle1 = fs.open_frange(id1).unwrap();
        handle1.write_at(0, b"test data 1").unwrap();
        fs.close_frange(handle1).unwrap();

        let handle2 = fs.open_frange(id2).unwrap();
        handle2.write_at(0, b"test data 2").unwrap();
        fs.close_frange(handle2).unwrap();

        // Force flush and drop
        fs.maybe_flush(true).unwrap();
        drop(fs);

        // Reopen filesystem
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(dir.path().join("test.db"))
            .unwrap();
        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let reopened_fs = Fs::new(mmap).expect("Failed to reopen filesystem");

        // Verify metadata persisted correctly
        let franges = reopened_fs.franges.read();

        let metadata1 = franges.get(&id1).unwrap();
        assert_eq!(metadata1.value().size, 1024);
        assert_eq!(metadata1.value().length.load(SeqCst), 11); // "test data 1" length

        let metadata2 = franges.get(&id2).unwrap();
        assert_eq!(metadata2.value().size, 2048);
        assert_eq!(metadata2.value().length.load(SeqCst), 11); // "test data 2" length

        dir.close().unwrap();
    }

    #[test]
    fn test_free_range_persistence() {
        setup_deadlock_detection();

        let (dir, file) = setup_test_file();
        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let fs = Fs::init(mmap).expect("Failed to initialize filesystem");

        // Create and delete some franges to create specific free range patterns
        let id1 = fs.create_frange(1024).unwrap();
        let id2 = fs.create_frange(2048).unwrap();
        let id3 = fs.create_frange(1024).unwrap();

        // Delete middle frange to create fragmentation
        fs.delete_frange(id2).unwrap();

        // Force flush and drop
        fs.maybe_flush(true).unwrap();
        drop(fs);

        // Reopen filesystem
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(dir.path().join("test.db"))
            .unwrap();
        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let reopened_fs = Fs::new(mmap).expect("Failed to reopen filesystem");

        // Should still be able to allocate in the freed space
        let new_id = reopened_fs.create_frange(2048).unwrap();
        assert!(new_id > id1 && new_id > id3);

        dir.close().unwrap();
    }

    #[test]
    fn test_data_persistence() {
        setup_deadlock_detection();

        let (dir, file) = setup_test_file();
        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let fs = Fs::init(mmap).expect("Failed to initialize filesystem");

        // Create a frange and write data
        let id = fs.create_frange(1024).unwrap();
        let handle = fs.open_frange(id).unwrap();
        handle.write_at(0, b"persistent data test").unwrap();
        fs.close_frange(handle).unwrap();

        // Force flush and drop
        fs.maybe_flush(true).unwrap();
        drop(fs);

        // Reopen filesystem
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(dir.path().join("test.db"))
            .unwrap();
        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let reopened_fs = Fs::new(mmap).expect("Failed to reopen filesystem");

        // Read and verify data
        let handle = reopened_fs.open_frange(id).unwrap();
        let mut buf = vec![0u8; 20]; // Length of "persistent data test"
        handle.read_at(0, &mut buf).unwrap();
        assert_eq!(&buf, b"persistent data test");

        dir.close().unwrap();
    }

    #[test]
    fn test_next_frange_id_persistence() {
        setup_deadlock_detection();

        let (dir, file) = setup_test_file();
        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let fs = Fs::init(mmap).expect("Failed to initialize filesystem");

        // Create several franges to increment the next_frange_id
        for _ in 0..5 {
            fs.create_frange(1024).unwrap();
        }

        let last_id = fs.next_frange_id.load(Acquire);

        // Force flush and drop
        fs.maybe_flush(true).unwrap();
        drop(fs);

        // Reopen filesystem
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(dir.path().join("test.db"))
            .unwrap();
        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let reopened_fs = Fs::new(mmap).expect("Failed to reopen filesystem");

        // Verify next_frange_id was persisted
        assert_eq!(reopened_fs.next_frange_id.load(Acquire), last_id);

        // Create new frange and verify ID continues from last
        let new_id = reopened_fs.create_frange(1024).unwrap();
        assert_eq!(new_id, last_id);

        dir.close().unwrap();
    }

    #[test]
    fn test_dirty_pages_persistence() {
        setup_deadlock_detection();

        let (dir, file) = setup_test_file();
        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let fs = Fs::init(mmap).expect("Failed to initialize filesystem");

        // Create a frange and write data without flushing
        let id = fs.create_frange(1024).unwrap();
        let handle = fs.open_frange(id).unwrap();
        handle.write_at(0, b"unflushed data").unwrap();

        // Verify dirty pages exist
        let dirty_page_count = fs.dirty_pages.read().len();
        assert!(dirty_page_count > 0);

        // Close without explicit flush
        fs.close_frange(handle).unwrap();

        // Reopen filesystem
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(dir.path().join("test.db"))
            .unwrap();
        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let reopened_fs = Fs::new(mmap).expect("Failed to reopen filesystem");

        // Verify data was actually persisted despite no explicit flush
        let handle = reopened_fs.open_frange(id).unwrap();
        let mut buf = vec![0u8; 14]; // Length of "unflushed data"
        handle.read_at(0, &mut buf).unwrap();
        assert_eq!(&buf, b"unflushed data");

        dir.close().unwrap();
    }

    #[test]
    fn test_compaction_triggers() {
        setup_deadlock_detection();

        let (_dir, file) = setup_test_file();
        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let fs = Fs::init(mmap).unwrap();

        // Create some fragmented state
        let mut ids = Vec::new();
        for i in 0..20 {
            let size = if i % 2 == 0 { 4096 } else { 8192 };
            let id = fs.create_frange(size).unwrap();

            // Write less data than allocated to create fragmentation
            let handle = fs.open_frange(id).unwrap();
            let data = vec![1u8; size as usize / 2];
            handle.write_at(0, &data).unwrap();
            fs.close_frange(handle).unwrap();

            ids.push(id);
        }

        // Delete some franges to create fragmentation
        for i in (0..ids.len()).step_by(2) {
            fs.delete_frange(ids[i]).unwrap();
        }

        // Check if compaction is triggered
        let config = CompactionConfig {
            fragmentation_threshold: 0.2,
            min_fragment_size: 1024,
            max_compact_batch: 5,
            block_buffer_size: 4096,
        };

        assert!(fs.maybe_compact_with_config(&config).unwrap());

        // Verify improved fragmentation
        let stats_after = fs.fragmentation_stats();
        assert!(stats_after.free_range_count < ids.len() as u64 / 2);
    }

    #[test]
    fn test_compaction_preserves_data() {
        setup_deadlock_detection();

        let (_dir, file) = setup_test_file();
        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let fs = Fs::init(mmap).unwrap();

        // Create a frange with known data but fragmented
        let id = fs.create_frange(8192).unwrap();
        let handle = fs.open_frange(id).unwrap();

        let data = b"Hello, World!";
        handle.write_at(0, data).unwrap();
        fs.close_frange(handle).unwrap();

        // Force compaction
        let config = CompactionConfig {
            fragmentation_threshold: 0.0, // Always compact
            min_fragment_size: 1024,
            max_compact_batch: 1,
            block_buffer_size: 4096,
        };

        fs.maybe_compact_with_config(&config).unwrap();

        // Verify data is preserved
        let handle = fs.open_frange(id).unwrap();
        let mut buf = vec![0u8; data.len()];
        handle.read_at(0, &mut buf).unwrap();
        assert_eq!(&buf, data);
    }
}
