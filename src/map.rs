use std::{
    cell::SyncUnsafeCell,
    fs::{
        File,
        OpenOptions,
    },
    ops::Range,
    path::PathBuf,
    sync::atomic::{
            AtomicPtr,
            AtomicU64,
            Ordering::{
                AcqRel,
                Acquire,
                Release,
            },
        },
};

use memmap2::MmapMut;
use parking_lot::{
    Mutex,
    RwLock,
};

use crate::errs::{
    SegmentError,
    SegmentError::IoError,
};

/// The maximum amount of disk space that can be allocated at once.
pub const MAX_GROWTH_INCREMENT: u64 = 8 * 1024 * 1024;

#[derive(Debug)]
pub struct Map {
    inner: AtomicPtr<SyncUnsafeCell<MmapMut>>,
    file: Mutex<File>,
    current_size: AtomicU64,
    resize_lock: RwLock<()>,
    read_only: bool,
}

impl Map {
    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn new(path: PathBuf, initial_size: u64) -> Result<Self, SegmentError> {
        let file = match OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path.clone())
        {
            | Ok(v) => v,
            | Err(e) => return Err(IoError(e)),
        };

        match file.set_len(initial_size) {
            | Ok(_) => {},
            | Err(e) => return Err(IoError(e)),
        };

        let size_metadata = match file.metadata() {
            | Ok(v) => v.len(),
            | Err(e) => return Err(IoError(e)),
        };

        // SAFETY: none, this is an unsafe operation
        let mmap = unsafe {
            match MmapMut::map_mut(&file) {
                | Ok(v) => v,
                | Err(e) => return Err(IoError(e)),
            }
        };

        Ok(Self {
            inner: AtomicPtr::new(Box::into_raw(Box::new(SyncUnsafeCell::new(mmap)))),
            file: Mutex::new(file),
            current_size: AtomicU64::new(size_metadata),
            resize_lock: RwLock::new(()),
            read_only: false,
        })
    }

    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn open(path: PathBuf) -> Result<Self, SegmentError> {
        let file = match OpenOptions::new().read(true).write(true).open(path.clone()) {
            | Ok(v) => v,
            | Err(e) => return Err(IoError(e)),
        };

        let size_metadata = match file.metadata() {
            | Ok(v) => v.len(),
            | Err(e) => return Err(IoError(e)),
        };

        // SAFETY: none, this is an unsafe operation
        let mmap = unsafe {
            match MmapMut::map_mut(&file) {
                | Ok(v) => v,
                | Err(e) => return Err(IoError(e)),
            }
        };

        Ok(Self {
            inner: AtomicPtr::new(Box::into_raw(Box::new(SyncUnsafeCell::new(mmap)))),
            file: Mutex::new(file),
            current_size: AtomicU64::new(size_metadata),
            resize_lock: RwLock::new(()),
            read_only: false,
        })
    }

    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn open_read_only(path: PathBuf) -> Result<Self, SegmentError> {
        // Note: We open with write permissions to allow mmap creation,
        // but enforce read-only behavior at the API level
        let file = match OpenOptions::new().read(true).write(true).open(path.clone()) {
            | Ok(v) => v,
            | Err(e) => return Err(IoError(e)),
        };

        let size_metadata = match file.metadata() {
            | Ok(v) => v.len(),
            | Err(e) => return Err(IoError(e)),
        };

        // SAFETY: none, this is an unsafe operation
        let mmap = unsafe {
            match MmapMut::map_mut(&file) {
                | Ok(v) => v,
                | Err(e) => return Err(IoError(e)),
            }
        };

        Ok(Self {
            inner: AtomicPtr::new(Box::into_raw(Box::new(SyncUnsafeCell::new(mmap)))),
            file: Mutex::new(file),
            current_size: AtomicU64::new(size_metadata),
            resize_lock: RwLock::new(()),
            read_only: true,
        })
    }

    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn grow(&self, new_size: u64) -> Result<(), SegmentError> {
        // Check if map is read-only
        if self.read_only {
            return Err(IoError(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "Cannot grow read-only map",
            )));
        }

        let _guard = self.resize_lock.write();

        {
            let file = self.file.lock();
            // Sync file before truncating to ensure all writes are persisted
            match file.sync_all() {
                | Ok(_) => {},
                | Err(e) => return Err(IoError(e)),
            };
            match file.set_len(new_size) {
                | Ok(_) => {},
                | Err(e) => return Err(IoError(e)),
            };

            let size_metadata = match file.metadata() {
                | Ok(v) => v.len(),
                | Err(e) => return Err(IoError(e)),
            };

            self.current_size.store(size_metadata, Release);
        }

        let new_mmap = {
            let file = self.file.lock();

            // SAFETY: none, this is an unsafe operation
            SyncUnsafeCell::new(unsafe {
                match MmapMut::map_mut(&*file) {
                    | Ok(v) => v,
                    | Err(e) => return Err(IoError(e)),
                }
            })
        };

        // swap the pointers and remove the old one to prevent use after free
        let old_ptr = self.inner.swap(Box::into_raw(Box::new(new_mmap)), AcqRel);
        // SAFETY: this is just a swap
        unsafe {
            drop(Box::from_raw(old_ptr));
        }

        Ok(())
    }

    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn shrink(&self, new_size: u64) -> Result<(), SegmentError> {
        let _guard = self.resize_lock.write();

        {
            let file = self.file.lock();
            match file.set_len(new_size) {
                | Ok(_) => {},
                | Err(e) => return Err(IoError(e)),
            };

            let size_metadata = match file.metadata() {
                | Ok(v) => v.len(),
                | Err(e) => return Err(IoError(e)),
            };

            self.current_size.store(size_metadata, Release);
        }

        let new_mmap = {
            let file = self.file.lock();

            // SAFETY: none, this is an unsafe operation
            SyncUnsafeCell::new(unsafe {
                match MmapMut::map_mut(&*file) {
                    | Ok(v) => v,
                    | Err(e) => return Err(IoError(e)),
                }
            })
        };

        // swap the pointers and remove the old one to prevent use after free
        let old_ptr = self.inner.swap(Box::into_raw(Box::new(new_mmap)), AcqRel);
        // SAFETY: this is just a drop
        unsafe {
            drop(Box::from_raw(old_ptr));
        }

        Ok(())
    }

    /// Write to a specific range.
    pub fn write_to_range(
        &self,
        range: Range<usize>,
        writer: impl FnOnce(&mut [u8]),
    ) -> Result<(), SegmentError> {
        // Check if map is read-only
        if self.read_only {
            return Err(IoError(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "Cannot write to read-only map",
            )));
        }

        // Hold write lock for entire operation to prevent races with shrink/grow
        let _guard = self.resize_lock.write();

        let len = self.len();
        let end = range.end;

        // Grow the map if needed (inline to avoid lock recursion)
        if end > len {
            tracing::debug!("Growing map from {} to {}", len, end);
            let new_size = end as u64;

            {
                let file = self.file.lock();
                match file.set_len(new_size) {
                    | Ok(_) => {},
                    | Err(e) => return Err(IoError(e)),
                };

                // CRITICAL: Sync metadata to ensure mmap sees the new size
                match file.sync_all() {
                    | Ok(_) => {},
                    | Err(e) => return Err(IoError(e)),
                };

                let size_metadata = match file.metadata() {
                    | Ok(v) => v.len(),
                    | Err(e) => return Err(IoError(e)),
                };

                self.current_size.store(size_metadata, Release);
            }

            let new_mmap = {
                let file = self.file.lock();

                // SAFETY: none, this is an unsafe operation
                SyncUnsafeCell::new(unsafe {
                    match MmapMut::map_mut(&*file) {
                        | Ok(v) => v,
                        | Err(e) => return Err(IoError(e)),
                    }
                })
            };

            // swap the pointers and remove the old one to prevent use after free
            let old_ptr = self.inner.swap(Box::into_raw(Box::new(new_mmap)), AcqRel);
            // SAFETY: this is just a swap
            unsafe {
                drop(Box::from_raw(old_ptr));
            }
        }

        // now get the reference from the potentially new mapping
        let ptr = self.inner.load(Acquire);

        // SAFETY: We're ensuring synchronized access through the Map's methods
        unsafe {
            let mmap = &*ptr;
            let inner = &mut *mmap.get();

            // Defensive check: ensure the range is valid
            if range.end > inner.len() {
                return Err(IoError(std::io::Error::other(
                    format!(
                        "write_to_range: range.end ({}) > map.len ({}) after potential grow to {}",
                        range.end,
                        inner.len(),
                        end
                    ),
                )));
            }

            let slice = &mut inner[range.clone()];
            writer(slice);
        }

        Ok(())
    }

    #[inline]
    pub fn warn(&self, range: Range<usize>) {
        let _guard = self.resize_lock.read(); // Hold read lock during access
        let ptr = self.inner.load(Acquire);
        // SAFETY: none, this is an unsafe operation as we are dereferencing a pointer
        unsafe {
            let mmap = &*ptr;
            let inner = &*mmap.get();
            let _ = inner.advise_range(
                memmap2::Advice::WillNeed,
                range.start,
                range.end - range.start,
            );
        }
    }

    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn close(&self) -> Result<(), SegmentError> {
        let _guard = self.resize_lock.write();

        let ptr = self.inner.load(Acquire);
        // SAFETY: none, this is an unsafe operation as we are dereferencing a pointer
        unsafe {
            let mmap = &*ptr;
            let inner = &mut *mmap.get();
            match inner.flush().map_err(IoError) {
                | Ok(_) => {},
                | Err(e) => return Err(e),
            };
        }

        Ok(())
    }

    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn len(&self) -> usize {
        let fh = self.file.lock();
        match fh.metadata() {
            | Ok(v) => v.len() as usize,
            | Err(_) => 0,
        }
    }

    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Safely read from a range by holding a read lock and executing a closure
    ///
    /// This is the safe replacement for the Index trait - it holds the read
    /// lock for the duration of the closure, preventing concurrent
    /// resize/unmap.
    pub fn read_range<F, R>(&self, range: Range<usize>, f: F) -> Result<R, SegmentError>
    where
        F: FnOnce(&[u8]) -> R, {
        // Hold read lock for entire duration including closure execution
        // The guard MUST NOT be dropped until after the closure completes
        let _guard = self.resize_lock.read();
        let ptr = self.inner.load(Acquire);

        // Execute everything in one expression to ensure guard lifetime
        // SAFETY: ptr is valid while guard is held, preventing resize/deallocation
        let result = unsafe {
            let mmap = &*ptr;
            let inner = &*mmap.get();

            // Bounds check
            if range.end > inner.len() {
                return Err(IoError(std::io::Error::other(
                    format!(
                        "read_range: range.end ({}) > map.len ({})",
                        range.end,
                        inner.len()
                    ),
                )));
            }

            // Execute closure while holding guard
            f(&inner[range])
        };

        // Guard is explicitly kept alive until here
        drop(_guard);
        Ok(result)
    }

    /// Hint to kernel that we'll need this range soon (prefetch)
    pub fn advise_willneed(&self) -> Result<(), SegmentError> {
        let _guard = self.resize_lock.read();
        let ptr = self.inner.load(Acquire);

        // SAFETY: ptr is valid for the lifetime of the guard
        unsafe {
            let inner = &*(*ptr).get();
            if let Err(e) = inner.advise(memmap2::Advice::WillNeed).map_err(IoError) {
                return Err(e);
            }
        }
        Ok(())
    }

    /// Hint to kernel that we'll read sequentially (aggressive readahead)
    pub fn advise_sequential(&self) -> Result<(), SegmentError> {
        let _guard = self.resize_lock.read();
        let ptr = self.inner.load(Acquire);

        // SAFETY: ptr is valid for the lifetime of the guard
        unsafe {
            let inner = &*(*ptr).get();
            if let Err(e) = inner.advise(memmap2::Advice::Sequential).map_err(IoError) {
                return Err(e);
            }
        }
        Ok(())
    }

    /// Hint to kernel that access is random (disable readahead)
    pub fn advise_random(&self) -> Result<(), SegmentError> {
        let _guard = self.resize_lock.read();
        let ptr = self.inner.load(Acquire);

        // SAFETY: ptr is valid for the lifetime of the guard
        unsafe {
            let inner = &*(*ptr).get();
            if let Err(e) = inner.advise(memmap2::Advice::Random).map_err(IoError) {
                return Err(e);
            }
        }
        Ok(())
    }
}

// REMOVED: IndexMut, Deref, DerefMut impls
// These are fundamentally unsafe for concurrent access because they return
// references without holding locks. Use read_range() and write_to_range()
// instead.

impl Drop for Map {
    fn drop(&mut self) {
        let ptr = self.inner.load(Acquire);
        // SAFETY: none, this is an unsafe operation
        unsafe {
            drop(Box::from_raw(ptr));
        }
    }
}

/// SAFETY: trait impl
unsafe impl Send for Map {}

/// SAFETY: trail impl
unsafe impl Sync for Map {}

#[cfg(test)]
mod tests {
    

    use std::sync::atomic::Ordering::Relaxed;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn test_new_map() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_new.segment");

        let initial_size = 1024;
        let map = Map::new(file_path, initial_size).unwrap();

        assert_eq!(map.len(), initial_size as usize);
        assert!(!map.is_empty());
        assert_eq!(map.current_size.load(Relaxed), initial_size);
    }

    #[test]
    fn test_grow_map() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_grow.segment");

        let initial_size = 1024;
        let map = Map::new(file_path, initial_size).unwrap();

        assert_eq!(map.len(), initial_size as usize);

        let new_size = 2048;
        map.grow(new_size).unwrap();

        assert_eq!(map.len(), new_size as usize);
    }

    #[test]
    fn test_read_range() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_read.segment");

        let initial_size = 1024;
        let map = Map::new(file_path, initial_size).unwrap();

        // Read a range from the map using the new safe API
        let result = map
            .read_range(0..10, |slice| {
                assert_eq!(slice.len(), 10);
                // New mmap should be zeroed
                assert_eq!(slice, &[0u8; 10]);
                slice.to_vec()
            })
            .unwrap();

        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_write_to_range() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_write.segment");

        let initial_size = 1024;
        let map = Map::new(file_path, initial_size).unwrap();

        // Write to a range in the map using the new safe API
        let data: Vec<u8> = (0..10).map(|i| i as u8).collect();
        map.write_to_range(0..10, |slice| {
            slice.copy_from_slice(&data);
        })
        .unwrap();

        // Check that the changes persisted
        let result = map.read_range(0..10, |slice| slice.to_vec()).unwrap();

        for i in 0..10 {
            assert_eq!(result[i], i as u8);
        }
    }

    #[test]
    fn test_write_grow() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_write_grow.segment");

        let initial_size = 100;
        let map = Map::new(file_path, initial_size).unwrap();

        assert_eq!(map.len(), initial_size as usize);

        // Write beyond the current size to trigger grow
        let data: Vec<u8> = (100..200).map(|i| i as u8).collect();
        map.write_to_range(100..200, |slice| {
            slice.copy_from_slice(&data);
        })
        .unwrap();

        assert!(map.len() >= 200);

        // Check that the changes persisted
        let result = map.read_range(100..200, |slice| slice.to_vec()).unwrap();

        for i in 0..100 {
            assert_eq!(result[i], (i + 100) as u8);
        }
    }

    #[test]
    fn test_read_full() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_read_full.segment");

        let initial_size = 10;
        let map = Map::new(file_path, initial_size).unwrap();

        // Read the entire map using the new safe API
        let all_zeros = map
            .read_range(0..initial_size as usize, |slice| {
                slice.iter().all(|&b| b == 0)
            })
            .unwrap();

        assert!(all_zeros);
    }

    #[test]
    fn test_write_full() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_write_full.segment");

        let initial_size = 10;
        let map = Map::new(file_path, initial_size).unwrap();

        // Write pattern to the entire map using the new safe API
        let data: Vec<u8> = (0..initial_size as usize).map(|i| (i * 2) as u8).collect();
        map.write_to_range(0..initial_size as usize, |slice| {
            slice.copy_from_slice(&data);
        })
        .unwrap();

        // Check that the changes persisted
        for i in 0..initial_size as usize {
            let byte = map.read_range(i..i + 1, |slice| slice[0]).unwrap();
            assert_eq!(byte, (i * 2) as u8);
        }
    }

    #[test]
    fn test_concurrent_access() {
        use std::{
            sync::Arc,
            thread,
        };

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_concurrent.segment");

        let initial_size = 1024;
        let map = Arc::new(Map::new(file_path, initial_size).unwrap());

        let mut handles = vec![];

        // Spawn multiple threads to read from the map concurrently using the safe API
        for i in 0..4 {
            let map_clone = Arc::clone(&map);
            let handle = thread::spawn(move || {
                let offset = i * 10;
                let range = offset..offset + 10;
                let len = map_clone.read_range(range, |slice| slice.len()).unwrap();
                assert_eq!(len, 10);
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_is_empty() {
        let dir = tempdir().unwrap();

        // Test with zero size
        let file_path = dir.path().join("test_empty.segment");
        let map = Map::new(file_path, 0).unwrap();
        assert!(map.is_empty());

        // Test with non-zero size
        let file_path = dir.path().join("test_not_empty.segment");
        let map = Map::new(file_path, 1).unwrap();
        assert!(!map.is_empty());
    }

    #[test]
    fn test_persistence() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_persistence.segment");

        // First, create and modify the map using the safe API
        {
            let map = Map::new(file_path.clone(), 100).unwrap();
            let data: Vec<u8> = (1..11).map(|i| i as u8).collect();
            map.write_to_range(0..10, |slice| {
                slice.copy_from_slice(&data);
            })
            .unwrap();
            // map will be dropped here, which should flush changes
        }

        // Now open the same file again and verify data persisted
        {
            let map = Map::new(file_path, 100).unwrap();
            let result = map.read_range(0..10, |slice| slice.to_vec()).unwrap();

            for i in 0..10 {
                assert_eq!(result[i], (i + 1) as u8);
            }
        }
    }

    #[test]
    fn test_read_only_map() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_readonly.segment");

        // First, create and write some data to the file
        {
            let map = Map::new(file_path.clone(), 100).unwrap();
            let data: Vec<u8> = (1..11).map(|i| i as u8).collect();
            map.write_to_range(0..10, |slice| {
                slice.copy_from_slice(&data);
            })
            .unwrap();
        }

        // Now open it in read-only mode
        let readonly_map = Map::open_read_only(file_path).unwrap();

        // Reading should work
        let result = readonly_map
            .read_range(0..10, |slice| slice.to_vec())
            .unwrap();

        for i in 0..10 {
            assert_eq!(result[i], (i + 1) as u8);
        }

        // Writing should fail
        let data = vec![99u8; 10];
        let write_result = readonly_map.write_to_range(0..10, |slice| {
            slice.copy_from_slice(&data);
        });

        assert!(write_result.is_err());
        if let Err(IoError(e)) = write_result {
            assert_eq!(e.kind(), std::io::ErrorKind::PermissionDenied);
        } else {
            panic!("Expected PermissionDenied error");
        }
    }
}
