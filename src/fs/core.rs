use std::{
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
    slice::from_raw_parts,
    sync::{
        atomic,
        atomic::{
            fence,
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
use getset::{CopyGetters, Getters};
use gxhash::HashSet;
use memmap2::{
    MmapMut,
    UncheckedAdvice::DontNeed,
};
use parking_lot::{
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
    fs::handle::{
        FRangeHandle,
        FRangeMetadata,
        OrderedRange,
    },
};
use crate::fs::compaction::FragmentationStats;

// TODO(@siennathesane): make this configurable
const FLUSH_INTERVAL_SECS: u64 = 5;

#[repr(C)]
#[derive(Debug)]
struct FsHeader {
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

    fn new(page_size: u32, metadata_offset: u64, metadata_size: u64) -> Self {
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

#[derive(Debug)]
struct FsMetadata {
    franges: SkipMap<u64, FRangeMetadata>,
    free_ranges: SkipSet<OrderedRange>,
}

impl FsMetadata {
    fn new() -> Self {
        Self {
            franges: SkipMap::new(),
            free_ranges: SkipSet::new(),
        }
    }

    fn from_fs(fs: &Fs) -> Self {
        let metadata = Self::new();

        {
            let franges = fs.franges.read();
            for entry in franges.iter() {
                metadata.franges.insert(*entry.key(), entry.value().clone());
            }
        }

        {
            let free_ranges = fs.free_ranges.read();
            for range in free_ranges.iter() {
                metadata.free_ranges.insert(range.value().clone());
            }
        }

        metadata
    }

    fn serialize(&self) -> Bytes {
        let mut buf = BytesMut::new();

        // Write franges count
        buf.put_u64_le(self.franges.len() as u64);

        // Write each frange entry
        for entry in self.franges.iter() {
            // Write key
            buf.put_u64_le(*entry.key());

            // Write FRangeMetadata
            let metadata = entry.value();
            buf.put_u64_le(metadata.range().start());
            buf.put_u64_le(metadata.range().end());
            buf.put_u64_le(metadata.id());
            buf.put_u64_le(metadata.length().load(SeqCst));
            buf.put_u64_le(metadata.size());
            buf.put_u64_le(metadata.created_at());
            buf.put_u64_le(metadata.modified_at().load(SeqCst));
        }

        // Write free ranges count
        buf.put_u64_le(self.free_ranges.len() as u64);

        // Write each free range
        for range in self.free_ranges.iter() {
            buf.put_u64_le(range.start());
            buf.put_u64_le(range.end());
        }

        buf.freeze()
    }

    fn deserialize(mut bytes: Bytes) -> Result<Self, FsError> {
        let fs_metadata = Self::new();

        // Read franges
        let frange_count = bytes.get_u64_le() as usize;
        for _ in 0..frange_count {
            let key = bytes.get_u64_le();

            // // Read FRangeMetadata
            let range = OrderedRange::new(bytes.get_u64_le(), bytes.get_u64_le());

            let metadata = FRangeMetadata::new(
                range,
                bytes.get_u64_le(),
                bytes.get_u64_le(),
                bytes.get_u64_le(),
                bytes.get_u64_le(),
                bytes.get_u64_le(),
            );

            fs_metadata.franges.insert(key, metadata);
        }

        // Read free ranges
        let free_range_count = bytes.get_u64_le() as usize;
        for _ in 0..free_range_count {
            let range = OrderedRange::new(bytes.get_u64_le(), bytes.get_u64_le());
            fs_metadata.free_ranges.insert(range);
        }

        Ok(fs_metadata)
    }

    // Calculate total serialized size for pre-allocation
    fn serialized_size(&self) -> usize {
        // 8 bytes for franges count
        // For each frange: 8 (key) + 7*8 (metadata fields)
        // 8 bytes for free ranges count
        // For each free range: 2*8 (start/end)
        8 + (self.franges.len() * (8 + 56)) + 8 + (self.free_ranges.len() * 16)
    }
}

// For calculating size during filesystem init
const INITIAL_METADATA_SIZE: usize = 4096; // Or calculate based on expected initial capacity

/// A custom filesystem implementation for CesiumDB. It is designed to work with
/// physical storage devices or `fallocate`d files and uses memory-mapped files
/// to provide fast access to the data.
pub struct Fs {
    pub(crate) mmap: Arc<MmapMut>,
    pub(crate) franges: RwLock<SkipMap<u64, FRangeMetadata>>,

    pub(crate) free_ranges: RwLock<SkipSet<OrderedRange>>,
    pub(crate) open_franges: RwLock<HashSet<u64>>,
    pub(crate) next_frange_id: AtomicU64,

    // flushing
    pub(crate) last_flush: AtomicU64,
    pub(crate) dirty_pages: RwLock<SkipSet<usize>>,
    pub(crate) page_size: usize,
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

        // Read metadata
        let metadata: FsMetadata = {
            let start = header.metadata_offset as usize;
            let end = start + header.metadata_size as usize;
            match FsMetadata::deserialize(Bytes::copy_from_slice(&mmap[start..end])) {
                | Ok(v) => v,
                | Err(_) => return Err(InvalidHeaderFormat("invalid metadata region".into())),
            }
        };

        // Initialize state from metadata
        let franges = SkipMap::new();
        for (id, frange) in metadata.franges {
            franges.insert(id, frange);
        }

        let free_ranges = SkipSet::new();
        for range in metadata.free_ranges {
            free_ranges.insert(range);
        }

        Ok(Arc::new(Self {
            mmap: Arc::new(mmap),
            franges: RwLock::new(franges),
            free_ranges: RwLock::new(free_ranges),
            open_franges: RwLock::new(HashSet::default()),
            next_frange_id: AtomicU64::new(header.next_frange_id),
            last_flush: AtomicU64::new(0),
            dirty_pages: RwLock::new(SkipSet::new()),
            page_size: header.page_size as usize,
        }))
    }

    /// Initialize a new filesystem in the given mmap
    pub fn init(mut mmap: MmapMut) -> Result<Arc<Self>, FsError> {
        let total_size = mmap.len() as u64;

        // Calculate sizes
        let header_bytes = FsHeader::new(
            BLOCK_SIZE as u32,
            size_of::<FsHeader>() as u64,
            INITIAL_METADATA_SIZE as u64,
        )
        .serialize();

        let header_size = header_bytes.len();
        let data_start = header_size + INITIAL_METADATA_SIZE;

        // Write header using actual serialized size
        mmap[..header_size].copy_from_slice(&header_bytes);

        // Initialize empty metadata
        let metadata = FsMetadata::new();
        let encoded = metadata.serialize();

        // Zero metadata region and write encoded data
        mmap[header_size..data_start].fill(0);
        mmap[header_size..header_size + encoded.len()].copy_from_slice(&encoded);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Create filesystem with single free range
        let fs = Self {
            mmap: Arc::new(mmap),
            franges: RwLock::new(SkipMap::new()),
            free_ranges: RwLock::new(SkipSet::new()),
            open_franges: RwLock::new(HashSet::default()),
            next_frange_id: AtomicU64::new(0),
            last_flush: AtomicU64::new(now),
            dirty_pages: RwLock::new(SkipSet::new()),
            page_size: BLOCK_SIZE,
        };

        // Add initial free range (excluding header and metadata)
        fs.free_ranges
            .write()
            .insert(OrderedRange::from(data_start as u64..total_size));

        Ok(Arc::new(fs))
    }

    pub fn create_frange(self: &Arc<Self>, size: u64) -> Result<u64, FsError> {
        // Attempt allocation with a limited lock scope
        let range = {
            let free = self.free_ranges.write();
            self.find_free_range(&free, size)
        };

        // Now lock is released, handle the result
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

        // Get the ID and create metadata
        let id = self.next_frange_id.fetch_add(1, SeqCst);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let metadata = FRangeMetadata::new(range, id, size, 0, now, now);

        // Insert metadata with minimal lock duration
        {
            let ranges = self.franges.write();
            ranges.insert(id, metadata);
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
            metadata.range().clone(),
            metadata,
            self.clone(),
        ))
    }

    pub fn close_frange(self: &Arc<Self>, handle: FRangeHandle) -> Result<(), FsError> {
        // Update metadata first
        {
            let franges = self.franges.write();
            match franges.get(&handle.metadata().id()) {
                | None => {
                    return Err(FRangeNotFound);
                },
                | Some(entry) => {
                    let updated = entry.value().clone();
                    updated.modified_at().store(
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        SeqCst,
                    );
                    updated
                        .length()
                        .store(handle.metadata().length().load(SeqCst), SeqCst);
                    franges.insert(handle.metadata().id(), updated);
                },
            };
        }

        // Then handle open_franges
        {
            let mut open_franges = self.open_franges.write();
            open_franges.remove(&handle.metadata().id());
        } // Release open_franges lock

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
                entry.value().range().clone()
            } else {
                return Ok(());
            };
            x
        };

        {
            let free_ranges = self.free_ranges.write();
            free_ranges.insert(range);
        }

        // Coalesce free ranges
        match self.coalesce_free_ranges() {
            | Ok(_) => {},
            | Err(e) => return Err(e),
        };

        // Persist changes
        self.persist_metadata()
    }

    /// Sync all changes to disk
    pub fn sync(self: &Arc<Self>) -> Result<(), FsError> {
        self.flush_dirty_pages()
    }

    /// Persist metadata changes to disk
    pub(in crate::fs) fn persist_metadata(self: &Arc<Self>) -> Result<(), FsError> {
        let metadata = {
            let metadata = FsMetadata::new();

            // Take one lock at a time and immediately release
            {
                let franges = self.franges.read();
                for entry in franges.iter() {
                    metadata.franges.insert(*entry.key(), entry.value().clone());
                }
            }

            {
                let free_ranges = self.free_ranges.read();
                for range in free_ranges.iter() {
                    metadata.free_ranges.insert(range.value().clone());
                }
            }

            metadata
        };

        // Then handle header and serialization without holding locks
        let mut header = {
            let mut header_bytes = [0u8; size_of::<FsHeader>()];
            header_bytes.copy_from_slice(&self.mmap[..size_of::<FsHeader>()]);
            match deserialize_header(&header_bytes) {
                | Ok(v) => v,
                | Err(e) => return Err(e),
            }
        };

        let encoded = metadata.serialize();

        // If we need more space, try to grow metadata region
        if encoded.len() > header.metadata_size as usize {
            // Calculate new size with some room for growth
            let new_size = encoded.len() * 2;

            // Make sure we have space to grow
            let data_start = header.metadata_offset + header.metadata_size;
            let free_ranges = self.free_ranges.write();

            // Collect matching ranges first to avoid borrowing conflict
            let matching_ranges: Vec<_> = free_ranges
                .iter()
                .filter(|r| r.start() == data_start)
                .collect();

            // Now we can safely modify free_ranges if we found a match
            if let Some(range) = matching_ranges.first() {
                let growth_needed = new_size as u64 - header.metadata_size;
                if range.end() - range.start() >= growth_needed {
                    // Update the free range
                    free_ranges.remove(range);
                    if range.end() - range.start() > growth_needed {
                        free_ranges
                            .insert(OrderedRange::new(data_start + growth_needed, range.end()));
                    }

                    // Update header with new metadata size
                    header.metadata_size = new_size as u64;
                } else {
                    return Err(InsufficientSpace);
                }
            } else {
                return Err(NoAdjacentSpace);
            }
        }

        // Write metadata using our safe wrapper
        let start = header.metadata_offset as usize;
        self.write_to_mmap(start, &encoded);

        // Update header with new metadata size and next_frange_id
        header.next_frange_id = self.next_frange_id.load(SeqCst);
        let header_bytes = header.serialize();
        self.write_to_mmap(0, &header_bytes);

        Ok(())
    }

    /// Write data to the mmap at the given offset.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it dereferences a raw pointer.
    /// - There is pointer arithmetic to calculate the destination pointer.
    /// - There is a `memcpy` on a raw pointer.
    fn write_to_mmap(self: &Arc<Self>, offset: usize, data: &[u8]) {
        // TODO(@siennathesane): do some bounds checking or something
        // SAFETY: yeah this is actually unsafe
        unsafe {
            let ptr = self.mmap.as_ptr().add(offset) as *mut u8;
            ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        fence(SeqCst);
    }

    fn find_free_range(
        self: &Arc<Self>,
        free: &RwLockWriteGuard<SkipSet<OrderedRange>>,
        size: u64,
    ) -> Result<OrderedRange, FsError> {
        // Find the rightmost suitable range by iterating in reverse
        let suitable_range = match free.iter().rev().find(|r| r.end() - r.start() >= size) {
            | None => {
                return Err(NoFreeSpace);
            },
            | Some(v) => v,
        };

        free.remove(&suitable_range);

        // When splitting a range, keep the left part free and allocate from the right
        if suitable_range.end() - suitable_range.start() > size {
            let new_free = OrderedRange::new(suitable_range.start(), suitable_range.end() - size);
            free.insert(new_free);
        }

        Ok(OrderedRange::new(
            suitable_range.end() - size,
            suitable_range.end(),
        ))
    }

    pub(in crate::fs) fn coalesce_free_ranges(self: &Arc<Self>) -> Result<(), FsError> {
        let free = self.free_ranges.write();
        let mut ranges: Vec<OrderedRange> = free.iter().map(|entry| (*entry).clone()).collect();
        ranges.sort();

        let mut i = 0;
        while i < ranges.len() - 1 {
            if ranges[i].end() == ranges[i + 1].start() {
                let merged = OrderedRange::new(ranges[i].start(), ranges[i + 1].end());
                free.remove(&ranges[i]);
                free.remove(&ranges[i + 1]);
                free.insert(merged);
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
                range.start() as usize,
                (range.end() - range.start()) as usize,
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
                | Some(range) if range.end() as usize == start => {
                    range.set_end(end as u64);
                },
                | Some(range) => {
                    ranges.push(range.clone());
                    current_range = Some(OrderedRange::new(start as u64, end as u64));
                },
                | None => {
                    current_range = Some(OrderedRange::new(
                        start as u64,
                        end as u64,
                    ));
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
    };

    use memmap2::MmapMut;
    use tempfile::tempdir;

    use crate::{
        block::BLOCK_SIZE,
        errs::FsError::InvalidHeaderFormat,
    };
    use crate::fs::compaction::CompactionConfig;
    use crate::fs::core::{deserialize_header, Fs, FsHeader, INITIAL_METADATA_SIZE};

    const TEST_FILE_SIZE: u64 = 1024 * 1024 * 10; // 10MB

    fn setup_test_file() -> (tempfile::TempDir, File) {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.db");

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&file_path)
            .unwrap();

        file.set_len(TEST_FILE_SIZE).unwrap();
        (dir, file)
    }

    fn create_and_reopen_fs() -> (tempfile::TempDir, File, Arc<Fs>) {
        let (dir, file) = setup_test_file();

        println!("Creating initial filesystem...");
        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let fs = Fs::init(mmap).expect("Failed to initialize filesystem");

        // Ensure mmap is synced before dropping
        println!("Syncing mmap...");
        fs.sync().expect("Failed to sync mmap");

        println!("Dropping initial filesystem...");
        drop(fs);

        println!("Reopening file...");
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(dir.path().join("test.db"))
            .unwrap();

        println!("Creating new mmap...");
        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };

        println!("Reopening filesystem...");
        let fs = Fs::new(mmap).expect("Failed to reopen filesystem");
        println!("Filesystem reopened successfully");

        (dir, file, fs)
    }

    #[test]
    fn test_create_frange() {
        let (_dir, _file, fs) = create_and_reopen_fs();

        // Test creating a frange
        let id = fs.create_frange(1024).unwrap();
        assert_eq!(id, 0);

        // Verify metadata
        let franges = fs.franges.read();
        let metadata = franges.get(&id).unwrap();
        assert_eq!(metadata.value().size(), 1024);
        assert_eq!(metadata.value().id(), 0);
    }

    #[test]
    fn test_open_and_close_frange() {
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
        let (_dir, _file, fs) = create_and_reopen_fs();

        // Create and open a frange
        let id = fs.create_frange(1024).unwrap();
        let handle = fs.open_frange(id).unwrap();

        // Write some data
        let data = b"Hello, World!";
        handle.write_at(0, data).unwrap();

        // Read it back
        let mut buf = vec![0u8; data.len()];
        handle.read_at(0, &mut buf).unwrap();
        assert_eq!(&buf, data);

        // Verify size matches written data
        assert_eq!(handle.len(), data.len() as u64);

        // If we want to verify allocation size:
        assert_eq!(
            handle.metadata().range().end() - handle.metadata().range().start(),
            1024
        );
    }

    #[test]
    fn test_delete_frange() {
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
        let (_dir, _file, fs) = create_and_reopen_fs();

        let id = fs.create_frange(1024).unwrap();
        let handle = fs.open_frange(id).unwrap();

        // Try to write beyond range
        let data = vec![1u8; 2048];
        assert!(handle.write_at(0, &data).is_err());
    }

    #[test]
    fn test_out_of_bounds_read() {
        let (_dir, _file, fs) = create_and_reopen_fs();

        let id = fs.create_frange(1024).unwrap();
        let handle = fs.open_frange(id).unwrap();

        // Try to read beyond range
        let mut buf = vec![0u8; 2048];
        assert!(handle.read_at(0, &mut buf).is_err());
    }

    #[test]
    fn test_multiple_franges() {
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
            assert!(free_range.end() > free_range.start()); // Ensure valid range
            assert!(free_range.start() >= (size_of::<FsHeader>() + INITIAL_METADATA_SIZE) as u64);
            // After header and metadata
        }

        drop(fs);
        dir.close().unwrap();
    }

    #[test]
    fn test_fs_new_after_init() {
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
        let (dir, file) = setup_test_file();
        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };

        // Try to open uninitialized file - should fail with invalid magic
        let result = Fs::new(mmap);
        assert!(matches!(result, Err(InvalidHeaderFormat(_))));

        dir.close().unwrap();
    }

    #[test]
    fn test_header_persistence() {
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
        assert_eq!(metadata1.value().size(), 1024);
        assert_eq!(metadata1.value().length().load(SeqCst), 11); // "test data 1" length

        let metadata2 = franges.get(&id2).unwrap();
        assert_eq!(metadata2.value().size(), 2048);
        assert_eq!(metadata2.value().length().load(SeqCst), 11); // "test data 2" length

        dir.close().unwrap();
    }

    #[test]
    fn test_free_range_persistence() {
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
        println!("Starting data persistence test");
        let (dir, file) = setup_test_file();
        println!("Test file created");
        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        println!("Memory map created");
        let fs = Fs::init(mmap).expect("Failed to initialize filesystem");
        println!("Filesystem initialized");

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

#[cfg(test)]
#[allow(clippy::question_mark_used)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::undocumented_unsafe_blocks)]
mod e2e_tests {
    use std::{
        fs::OpenOptions,
        sync::Arc,
    };

    use proptest::{
        collection::vec,
        proptest,
    };
    use rand::{
        thread_rng,
        Rng,
        RngCore,
    };
    use tempfile::tempdir;
    use tokio::task;

    use super::*;

    const TEST_FILE_SIZE: u64 = 1024 * 1024 * 100; // 100MB for testing

    async fn setup_test_fs() -> (Arc<Fs>, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)
            .unwrap();

        file.set_len(TEST_FILE_SIZE).unwrap();
        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let fs = Fs::init(mmap).unwrap();

        (fs, dir)
    }

    proptest! {
        #[test]
        fn test_random_frange_operations(
            operations in vec(0..4u8, 1..100),
            sizes in vec(1024u64..1024*1024, 1..20),
            write_positions in vec(0u64..1024*1024, 1..50),
            write_sizes in vec(128u64..4096, 1..50)
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let (fs, _dir) = setup_test_fs().await;
                let mut active_franges = Vec::new();
                let mut rng = thread_rng();

                for op in operations {
                    match op {
                        // Create new frange
                        0 if !sizes.is_empty() => {
                            let size = sizes[rng.gen_range(0..sizes.len())];
                            if let Ok(id) = fs.create_frange(size) {
                                active_franges.push((id, size));
                            }
                        },
                        // Delete existing frange
                        1 if !active_franges.is_empty() => {
                            let idx = rng.gen_range(0..active_franges.len());
                            let (id, _) = active_franges.remove(idx);
                            fs.delete_frange(id).unwrap();
                        },
                        // Write to existing frange
                        2 if !active_franges.is_empty() && !write_positions.is_empty() && !write_sizes.is_empty() => {
                            let (id, max_size) = active_franges[rng.gen_range(0..active_franges.len())];
                            let pos = write_positions[rng.gen_range(0..write_positions.len())] % max_size;
                            let size = write_sizes[rng.gen_range(0..write_sizes.len())].min(max_size - pos);

                            let handle = fs.open_frange(id).unwrap();
                            let data = vec![rng.gen::<u8>(); size as usize];
                            handle.write_at(pos, &data).unwrap();
                            fs.close_frange(handle).unwrap();
                        },
                        // Read and verify
                        3 if !active_franges.is_empty() && !write_positions.is_empty() && !write_sizes.is_empty() => {
                            let (id, max_size) = active_franges[rng.gen_range(0..active_franges.len())];
                            let pos = write_positions[rng.gen_range(0..write_positions.len())] % max_size;
                            let size = write_sizes[rng.gen_range(0..write_sizes.len())].min(max_size - pos);

                            let handle = fs.open_frange(id).unwrap();
                            let mut buf = vec![0u8; size as usize];
                            handle.read_at(pos, &mut buf).unwrap();
                            fs.close_frange(handle).unwrap();
                        },
                        _ => {}
                    }
                }

                // Final verification
                for (id, _) in active_franges {
                    fs.delete_frange(id).unwrap();
                }

                // Verify filesystem is still in a consistent state
                fs.sync().unwrap();
                assert!(fs.coalesce_free_ranges().is_ok());
            });
        }

        #[test]
        fn test_frange_size_boundaries(
            size in 1024u64..1024*1024*10,
            operations in vec(0u8..4, 1..20)
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let (fs, _dir) = setup_test_fs().await;

                // Try to create a frange of the given size
                if let Ok(id) = fs.create_frange(size) {
                    let handle = fs.open_frange(id).unwrap();

                    // Test operations at boundaries
                    let data = vec![1u8; 1024];

                    // Write at start
                    handle.write_at(0, &data).unwrap();

                    // Write at end - size
                    if size >= data.len() as u64 {
                        handle.write_at(size - data.len() as u64, &data).unwrap();
                    }

                    // Verify writes beyond size fail
                    assert!(handle.write_at(size, &data).is_err());

                    fs.close_frange(handle).unwrap();
                    fs.delete_frange(id).unwrap();
                }
            });
        }
    }

    #[tokio::test]
    async fn test_stress_with_mixed_sizes() {
        let (fs, _dir) = setup_test_fs().await;
        let fs = Arc::new(fs);

        // Create mixed workload with different sizes
        let sizes = vec![
            1024,        // 1KB
            1024 * 1024, // 1MB
            1024 * 16,   // 16KB
            1024 * 256,  // 256KB
            1024 * 64,   // 64KB
        ];

        let mut handles = vec![];

        for size in sizes {
            let fs_clone = fs.clone();
            handles.push(task::spawn(async move {
                let mut frange_ids = vec![];

                // Create multiple franges of this size
                for _ in 0..5 {
                    if let Ok(id) = fs_clone.create_frange(size) {
                        frange_ids.push(id);

                        // Write some data
                        let handle = fs_clone.open_frange(id).unwrap();
                        let data = vec![thread_rng().gen::<u8>(); (size / 2) as usize];
                        handle.write_at(0, &data).unwrap();
                        fs_clone.close_frange(handle).unwrap();
                    }
                }

                // Delete some randomly
                let mut rng = thread_rng();
                while !frange_ids.is_empty() {
                    let idx = rng.gen_range(0..frange_ids.len());
                    let id = frange_ids.swap_remove(idx);
                    fs_clone.delete_frange(id).unwrap();
                }
            }));
        }

        futures::future::join_all(handles).await;

        // Verify filesystem is still consistent
        fs.sync().unwrap();
        assert!(fs.coalesce_free_ranges().is_ok());
    }

    #[tokio::test]
    async fn test_simulated_crash_recovery() {
        let (fs, dir) = setup_test_fs().await;
        let mut frange_data = Vec::new();

        // Create some initial state
        for size in [1024, 2048, 4096] {
            let id = fs.create_frange(size).unwrap();
            let handle = fs.open_frange(id).unwrap();

            let data = vec![thread_rng().gen::<u8>(); size as usize];
            handle.write_at(0, &data).unwrap();

            fs.close_frange(handle).unwrap();
            frange_data.push((id, data));
        }

        // Force a flush
        fs.sync().unwrap();

        // Simulate crash by dropping the fs and reopening
        drop(fs);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(dir.path().join("test.db"))
            .unwrap();

        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let recovered_fs = Fs::new(mmap).unwrap();

        // Verify all data survived
        for (id, original_data) in frange_data {
            let handle = recovered_fs.open_frange(id).unwrap();
            let mut buf = vec![0u8; original_data.len()];
            handle.read_at(0, &mut buf).unwrap();
            assert_eq!(buf, original_data);
            recovered_fs.close_frange(handle).unwrap();
        }
    }

    #[tokio::test]
    async fn test_concurrent_frange_operations() {
        let (fs, _dir) = setup_test_fs().await;
        let fs = Arc::new(fs);

        let mut handles = vec![];

        // Create multiple concurrent writers
        for _ in 0..10 {
            let fs_clone = fs.clone();
            handles.push(task::spawn(async move {
                // Create a frange
                let id = fs_clone.create_frange(1024 * 1024).unwrap(); // 1MB each

                // Open it
                let handle = fs_clone.open_frange(id).unwrap();

                // Write random data
                let mut rng = thread_rng();
                let mut data = vec![0u8; 1024 * 512]; // 512KB
                rng.fill_bytes(&mut data);

                // Write multiple times
                for offset in (0..1024 * 1024).step_by(1024 * 512) {
                    handle.write_at(offset as u64, &data).unwrap();
                }

                // Close it
                fs_clone.close_frange(handle).unwrap();

                id
            }));
        }

        // Wait for all operations to complete
        let results = futures::future::join_all(handles).await;
        let frange_ids: Vec<u64> = results.into_iter().map(|r| r.unwrap()).collect();

        // Verify all franges
        for id in frange_ids {
            let handle = fs.open_frange(id).unwrap();
            let mut buf = vec![0u8; 1024 * 512];
            handle.read_at(0, &mut buf).unwrap();
            fs.close_frange(handle).unwrap();
        }
    }

    #[tokio::test]
    async fn test_fragmentation_and_coalescing() {
        let (fs, _dir) = setup_test_fs().await;

        // Create a sequence of franges with gaps
        let mut frange_ids = vec![];
        for _ in 0..5 {
            let id = fs.create_frange(1024 * 1024).unwrap(); // 1MB each
            frange_ids.push(id);
        }

        // Delete alternate franges to create fragmentation
        for i in (0..frange_ids.len()).step_by(2) {
            fs.delete_frange(frange_ids[i]).unwrap();
        }

        // Try to create a large frange that should fit in coalesced space
        let large_id = fs.create_frange(1024 * 1024 * 2).unwrap(); // 2MB

        // Verify the large frange is usable
        let handle = fs.open_frange(large_id).unwrap();
        let data = vec![42u8; 1024 * 1024 * 2];
        handle.write_at(0, &data).unwrap();
        fs.close_frange(handle).unwrap();
    }

    #[tokio::test]
    async fn test_stress_with_many_small_franges() {
        let (fs, _dir) = setup_test_fs().await;
        let fs = Arc::new(fs);

        // Create many small franges concurrently
        let mut handles = vec![];
        for _ in 0..100 {
            let fs_clone = fs.clone();
            handles.push(task::spawn(async move {
                let id = fs_clone.create_frange(1024).unwrap(); // 1KB each

                // Write small amounts of data repeatedly
                let handle = fs_clone.open_frange(id).unwrap();
                let data = vec![1u8; 128]; // 128 bytes
                for i in 0..8 {
                    handle.write_at(i * 128, &data).unwrap();
                }
                fs_clone.close_frange(handle).unwrap();

                // Reopen and verify
                let handle = fs_clone.open_frange(id).unwrap();
                let mut buf = vec![0u8; 128];
                handle.read_at(0, &mut buf).unwrap();
                assert_eq!(buf, data);
                fs_clone.close_frange(handle).unwrap();

                id
            }));
        }

        let results = futures::future::join_all(handles).await;
        assert_eq!(results.len(), 100);
    }

    #[tokio::test]
    async fn test_metadata_persistence() {
        let (fs, _dir) = setup_test_fs().await;

        // Create some initial state
        let id1 = fs.create_frange(1024 * 1024).unwrap();
        let id2 = fs.create_frange(1024 * 512).unwrap();

        let handle1 = fs.open_frange(id1).unwrap();
        let data1 = vec![1u8; 1024 * 1024];
        handle1.write_at(0, &data1).unwrap();
        fs.close_frange(handle1).unwrap();

        // Force a flush
        fs.sync().unwrap();

        // "Crash" and recover
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(_dir.path().join("test.db"))
            .unwrap();

        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let recovered_fs = Fs::new(mmap).unwrap();

        // Verify state was recovered
        let handle1 = recovered_fs.open_frange(id1).unwrap();
        let mut buf = vec![0u8; 1024 * 1024];
        handle1.read_at(0, &mut buf).unwrap();
        assert_eq!(buf, data1);
        recovered_fs.close_frange(handle1).unwrap();

        // Verify second frange exists
        let handle2 = recovered_fs.open_frange(id2).unwrap();
        recovered_fs.close_frange(handle2).unwrap();
    }

    #[tokio::test]
    async fn test_edge_cases() {
        let (fs, _dir) = setup_test_fs().await;

        // Calculate available space accounting for header and metadata
        let header_size = size_of::<FsHeader>();
        let metadata_size = INITIAL_METADATA_SIZE;
        let available_space = TEST_FILE_SIZE - (header_size + metadata_size) as u64;

        // Test creating a frange at the maximum available size
        let max_id = fs.create_frange(available_space).unwrap();

        // Attempt to create another frange should fail
        assert!(matches!(fs.create_frange(1024), Err(StorageExhausted)));

        // Delete the large frange
        fs.delete_frange(max_id).unwrap();

        // Should now be able to create a small frange
        let small_id = fs.create_frange(1024).unwrap();

        // Test reading/writing at frange boundaries
        let handle = fs.open_frange(small_id).unwrap();
        let data = vec![255u8; 1024];
        handle.write_at(0, &data).unwrap();

        // Reading past end should fail
        let mut buf = vec![0u8; 128];
        assert!(handle.read_at(1024, &mut buf).is_err());

        fs.close_frange(handle).unwrap();
    }
}
